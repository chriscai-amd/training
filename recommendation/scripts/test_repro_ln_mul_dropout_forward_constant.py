#!/usr/bin/env python3
"""CPU contracts for the separate nonzero constant forward control."""
import copy
from contextlib import redirect_stdout
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
import repro_ln_mul_dropout_forward_constant as repro
from test_repro_ln_mul_dropout_forward_zero import backing, tiny_elf


def config(**changes):
    value={**repro.DEFAULT_CONFIG,'rows':5,'cols':4,'stride_x':7,'stride_u':9,'guard_elements':16}
    value.update(changes);return value


def mathematical_reference(state,geometry):
    inputs=state['inputs'];x=inputs['X'].double();u=inputs['U'].double()
    mean=x.mean(1);centered=x-mean[:,None]
    rstd=(centered.square().mean(1)+repro.zero.fp32(geometry['eps'])).rsqrt()
    normalized=(centered*rstd[:,None])*inputs['W'].double()+inputs['B'].double()
    scale=1/(1-geometry['dropout_ratio'])
    out=state['outputs']
    out['Y'].copy_(torch.cat((u*scale,x*scale,normalized*u*scale),1))
    out['Mean'].copy_(mean);out['Rstd'].copy_(rstd)


class ConstantForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);cls.rng=torch.get_rng_state().clone()
        cls.zero_hash=repro.require_reviewed_helper()

    @classmethod
    def tearDownClass(cls):
        assert torch.equal(cls.rng,torch.get_rng_state())
        assert not torch.cuda.is_initialized()
        assert repro.require_reviewed_helper()==cls.zero_hash

    def run_fixture(self,c,path,**kwargs):
        return repro.run_loop(c,failure_path=path,device='cpu',max_resident_bytes=4<<20,
                              max_capture_bytes=4<<20,chunk_elements=7,chunk_bytes=31,**kwargs)

    def test_power_of_two_and_explicit_dropout_are_required(self):
        for changes in ({'cols':3},{'dropout_ratio':.3},{'dropout_ratio':.5000000000001},
                        {'rows':2**31},{'eps':1e-100},{'guard_elements':0}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):repro.geometry_plan(config(**changes))
        plan=repro.geometry_plan(repro.DEFAULT_CONFIG)
        self.assertEqual(plan['grid'],[88956,1,1])
        self.assertEqual(plan['exact_oracle'],{'Y_sections':[1.,2.,0.],'Mean':1.})
        self.assertEqual(plan['control_design']['dropout_intervention']['source_default'],.3)
        self.assertEqual(plan['control_design']['dropout_intervention']['control'],.5)
        self.assertFalse(any('Exact zero is required for Y and Mean' in line for line in repro.LIMITS))

    def test_nonzero_recipe_and_distinct_complete_backing_canaries(self):
        c=config();state=repro.make_buffers(c)
        for name,tensor in state['inputs'].items():
            self.assertTrue(bool((tensor==repro.INPUT_RECIPE[name]).all()))
            raw=repro.zero.flat_storage(tensor);canary=85 if name=='RANDOM_MASK' else 17
            self.assertEqual(float(raw[0]),canary);self.assertEqual(float(raw[-1]),canary)
        self.assertEqual(state['inputs']['X'].stride(),(7,1));self.assertEqual(state['inputs']['U'].stride(),(9,1))
        self.assertFalse(repro.check_inputs(state['inputs'],3)['failed'])
        for tensor in state['outputs'].values():
            self.assertTrue(bool(torch.isnan(tensor).all()))
            self.assertEqual(float(repro.zero.flat_storage(tensor)[0]),23)
            self.assertEqual(float(repro.zero.flat_storage(tensor)[-1]),23)

    def test_independent_mathematical_reference_gives_exact_expected_sections(self):
        for width in (1,2,4,8,512):
            c=config(cols=width,stride_x=width+3,stride_u=width+5)
            state=repro.make_buffers(c);mathematical_reference(state,c)
            y=state['outputs']['Y']
            self.assertTrue(torch.equal(y[:,:width],torch.ones_like(y[:,:width])))
            self.assertTrue(torch.equal(y[:,width:2*width],torch.full_like(y[:,width:2*width],2)))
            self.assertEqual(int(torch.count_nonzero(y[:,2*width:])),0)
            self.assertTrue(bool((state['outputs']['Mean']==1).all()))
            y[:,2*width:].fill_(-0.)
            self.assertFalse(repro.check_outputs(state['outputs'],c,7)['failed'])

    def test_zero_outputs_or_stale_zero_inputs_cannot_pass_nonzero_sections(self):
        c=config()
        for stale in ('all_outputs','X','U'):
            state=repro.make_buffers(c)
            if stale=='all_outputs':
                mathematical_reference(state,c);state['outputs']['Y'].zero_();state['outputs']['Mean'].zero_()
            else:
                state['inputs'][stale].zero_();mathematical_reference(state,c)
            checked=repro.check_outputs(state['outputs'],c,3)
            self.assertTrue(checked['failed'])
            if stale=='U':self.assertGreater(checked['y_sections'][0]['mismatched_elements'],0)
            else:self.assertGreater(checked['y_sections'][1]['mismatched_elements'],0)

    def test_each_exact_output_region_rejects_one_bit_fault(self):
        c=config()
        for name,col in (('Y',0),('Y',4),('Y',8),('Mean',None)):
            state=repro.make_buffers(c);mathematical_reference(state,c)
            tensor=state['outputs'][name];bits=tensor.view(repro.zero.integer_dtype(tensor))
            index=(-1,col) if col is not None else (-1,)
            bits[index]^=1
            checked=repro.check_outputs(state['outputs'],c,1)
            self.assertTrue(checked['failed']);self.assertEqual(checked['tensors'][name]['mismatched_elements'],1)
            self.assertEqual(checked['tensors'][name]['nonfinite_elements'],0)

    def test_nonfinite_statistics_and_finite_output_guards_are_separate(self):
        c=config()
        for name in ('Y','Mean','Rstd'):
            for fault in ('logical_nan','prefix_zero','suffix_nan'):
                state=repro.make_buffers(c);mathematical_reference(state,c)
                tensor=state['outputs'][name]
                if fault=='logical_nan':tensor[(0,)*tensor.ndim]=float('nan')
                else:repro.zero.flat_storage(tensor)[0 if fault=='prefix_zero' else -1]=0 if fault=='prefix_zero' else float('nan')
                checked=repro.check_outputs(state['outputs'],c,3)
                self.assertTrue(checked['failed'])
                self.assertEqual(checked['tensors'][name]['changed_guard_elements'],0 if fault=='logical_nan' else 1)
        state=repro.make_buffers(c);mathematical_reference(state,c);state['outputs']['Rstd'][-1]=1001.
        self.assertTrue(repro.check_outputs(state['outputs'],c,1)['failed'])

    def test_all_input_backing_regions_are_checked_against_nonzero_recipe(self):
        c=config()
        for name in repro.INPUT_RECIPE:
            for region in ('logical','prefix','suffix'):
                state=repro.make_buffers(c);tensor=state['inputs'][name]
                if region=='logical':tensor[(0,)*tensor.ndim]+=1
                else:repro.zero.flat_storage(tensor)[0 if region=='prefix' else -1]=0
                self.assertTrue(repro.check_inputs(state['inputs'],3)['failed'])
        for name in ('X','U'):
            state=repro.make_buffers(c);tensor=state['inputs'][name]
            repro.zero.flat_storage(tensor)[tensor.storage_offset()+tensor.shape[1]]=0
            checked=repro.check_inputs(state['inputs'],3)
            self.assertTrue(checked['failed']);self.assertEqual(checked['tensors'][name]['different_logical_elements'],0)
            self.assertEqual(checked['tensors'][name]['different_guard_or_padding_elements'],1)

    def test_resident_success_repoisons_only_logical_outputs_each_call(self):
        c=config();calls=[];pointers=[]
        def launch(state,actual):
            calls.append(1);pointers.append(tuple(t.data_ptr() for t in state['outputs'].values()))
            for tensor in state['outputs'].values():
                self.assertTrue(bool(torch.isnan(tensor).all()));self.assertEqual(float(repro.zero.flat_storage(tensor)[0]),23)
            mathematical_reference(state,actual)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'fail.pt';report=self.run_fixture(c,path,repeats=3,launch=launch)
            self.assertFalse(path.exists())
        self.assertEqual(report['status'],'PASS');self.assertEqual(len(calls),3);self.assertEqual(len(set(pointers)),1)
        self.assertEqual(report['runtime']['synthetic_recipe'],repro.INPUT_RECIPE)
        self.assertTrue(report['final_input_hashes_match_initial'])

    def test_first_fault_retains_all_outputs_inputs_and_recipe_without_rerun(self):
        c=config()
        for fault in ('Y0','Y1','Y2','Mean','Rstd','X','U_padding','RANDOM_MASK'):
            calls=[];observed={};configuration={'state':'before'}
            def launch(state,actual):
                calls.append(1);mathematical_reference(state,actual)
                if len(calls)==3:
                    configuration['state']='after'
                    if fault.startswith('Y'):state['outputs']['Y'][-1,int(fault[1])*4]+=1
                    elif fault in ('Mean','Rstd'):state['outputs'][fault][-1]+=1
                    elif fault=='U_padding':
                        u=state['inputs']['U'];repro.zero.flat_storage(u)[u.storage_offset()+u.shape[1]]=0
                    else:state['inputs'][fault][0,0]=0
                    observed.update({(group,name):backing(value).clone() for group,tensors in state.items() for name,value in tensors.items()})
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'fail.pt'
                report=self.run_fixture(c,path,repeats=9,launch=launch,configuration=configuration)
                artifact=torch.load(path,map_location='cpu',weights_only=False)
            self.assertEqual(len(calls),3);self.assertEqual(report['status'],'FAIL')
            self.assertEqual(artifact['format'],'ln_mul_dropout_forward_constant_failure_v1')
            self.assertEqual(artifact['runtime']['configuration'],{'state':'before'})
            self.assertEqual(artifact['runtime']['control_design']['expected_y_sections'],[1.,2.,0.])
            self.assertEqual(artifact['runtime']['synthetic_recipe'],repro.INPUT_RECIPE)
            for (group,name),raw in observed.items():self.assertTrue(torch.equal(backing(artifact['current_at_failure'][group][name]),raw))

    def test_missing_write_first_call_and_failed_save_do_not_rerun(self):
        for save_error in (False,True):
            calls=[]
            def launch(state,actual):calls.append(1)
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'fail.pt'
                if save_error:
                    with mock.patch.object(torch,'save',side_effect=RuntimeError('failed write')),self.assertRaisesRegex(RuntimeError,'failed write'):
                        self.run_fixture(config(),path,repeats=5,launch=launch)
                    self.assertFalse(path.exists());self.assertFalse(Path(str(path)+'.tmp').exists())
                else:
                    report=self.run_fixture(config(),path,repeats=5,launch=launch);self.assertEqual(report['iterations_completed'],1)
            self.assertEqual(len(calls),1)

    def test_budget_existing_paths_and_changed_frozen_helper_reject_before_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'fail.pt'
            with mock.patch.object(repro,'make_buffers') as allocate:
                for budget in ('max_resident_bytes','max_capture_bytes'):
                    with self.assertRaisesRegex(ValueError,'budget'):repro.run_loop(config(),failure_path=path,launch=lambda *_:None,**{budget:1})
                path.write_bytes(b'preserved')
                with self.assertRaises(FileExistsError):self.run_fixture(config(),path,repeats=1,launch=lambda *_:None)
                self.assertEqual(path.read_bytes(),b'preserved');path.unlink()
                with mock.patch.object(repro,'REVIEWED_ZERO_SHA256','changed'),self.assertRaisesRegex(ValueError,'frozen zero'):
                    self.run_fixture(config(),path,repeats=1,launch=lambda *_:None)
                allocate.assert_not_called()

    def test_original_launcher_archives_constant_driver_and_passes_explicit_half_dropout(self):
        @dataclass
        class Target:backend:str='hip';arch:str='gfx1250';warp_size:int=32
        c=config();state=repro.make_buffers(c);binary=tiny_elf()
        runner=mock.Mock(side_effect=lambda *args:mathematical_reference(state,c))
        class Compiled:
            name=repro.zero.KERNEL;hash='same-cache-key';metadata={'target':Target(),'shared':64,'num_warps':1}
            asm={'hsaco':binary,'amdgcn':'assembly','ttir':'IR'};kernel=binary;function=12345;n_regs=1024;n_spills=174
            def __getitem__(self,grid):return runner
        jit=SimpleNamespace(arg_names=repro.zero.ARGUMENTS,warmup=mock.Mock(return_value=Compiled()))
        module=SimpleNamespace(**{repro.zero.KERNEL:SimpleNamespace(fn=jit)})
        with tempfile.TemporaryDirectory() as directory:
            code=Path(directory)/'code'
            with mock.patch.object(repro.zero.importlib,'import_module',return_value=module),mock.patch.object(repro.zero,'original_inner_jit',return_value=jit):
                launcher=repro.OriginalLauncher(state,c,code)
            runner.assert_not_called()
            self.assertEqual((code/Path(repro.__file__).name).read_bytes(),Path(repro.__file__).read_bytes())
            self.assertEqual((code/Path(repro.zero.__file__).name).read_bytes(),Path(repro.zero.__file__).read_bytes())
            archived=json.loads((code/'constant_control.json').read_text())
            self.assertEqual(archived['constant_control_design']['dropout_intervention']['control'],.5)
            self.assertEqual(archived['hsaco_text_sha256'],hashlib.sha256(b'actual-machine-code').hexdigest())
            self.assertIn('matches_historical_candidate_text',archived)
            launcher(state,c)
            self.assertEqual(len(runner.call_args.args),23)
            self.assertEqual(runner.call_args.args[8:],(5,4,1e-6,.5,7,9,12,4,False,4,16,True,True,True,'none'))
            self.assertEqual(jit.warmup.call_args.kwargs,{'grid':(1,1,1),'num_warps':1,'num_stages':1})

    def test_cpu_cli_inspects_without_allocation_and_records_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'report.json'
            with mock.patch.object(repro,'make_buffers') as allocate,redirect_stdout(io.StringIO()):
                self.assertEqual(repro.main(['--report',str(path)]),0);allocate.assert_not_called()
            report=json.loads(path.read_text());self.assertEqual(report['status'],'CPU_INSPECTED_NO_ALLOCATION_OR_DISPATCH')
            self.assertEqual(report['control_design']['expected_y_sections'],[1.,2.,0.])
            self.assertTrue(report['source_files_unchanged_during_run']);before=path.read_bytes()
            with self.assertRaises(FileExistsError):repro.main(['--report',str(path)])
            self.assertEqual(path.read_bytes(),before)
            failed=Path(directory)/'error.json'
            with mock.patch.object(repro,'require_reviewed_helper',side_effect=RuntimeError('test error')),self.assertRaisesRegex(RuntimeError,'test error'):
                repro.main(['--report',str(failed)])
            self.assertEqual(json.loads(failed.read_text())['status'],'ERROR')


if __name__=='__main__':unittest.main()
