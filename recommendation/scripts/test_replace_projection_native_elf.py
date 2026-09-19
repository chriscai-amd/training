"""CPU-only ABI and fail-closed tests using real pinned ELF bytes and mock HIP."""
from pathlib import Path
import hashlib
import json
import os
import re
import subprocess
import tempfile
import unittest

LOGS = Path('/home/chcai/mi450_logs/root_cause_20260919')
DATA = Path('/home/chcai/dlrm_data/root_cause_20260919')
ORIGINAL_SHA = 'b323216507600606e076ab3546e81a931809008627c77d7f0d5a85ffa6cf1542'
DERIVED_SHA = '7d3fe969381c072e1c28d18491ab1f5ea1de51b186c73cbad3cda32c406db69d'
FROZEN_SHA = '07d962f71ad7dc1b4f996fc029b21ba2e1133951bb0701e25ef2da9ca9866195'

MOCK = r'''
#include <atomic>
#include <cassert>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <mutex>
using H=void*;
static std::atomic<int> altered{0};
static std::atomic<int> target_lookups{0};
static std::mutex file_lock;
static bool is(const char*s){const char*p=std::getenv("MOCK_CASE");return p&&std::strcmp(p,s)==0;}
static void event(const char*name){std::lock_guard<std::mutex> g(file_lock);FILE*f=std::fopen(std::getenv("MOCK_HIP_RECORD"),"a");assert(f);std::fprintf(f,"%s\n",name);std::fclose(f);}
extern "C" void mock_change(int value){altered=value;}
extern "C" int hipModuleGetFunction(H*f,H m,const char*n){
 bool target=n&&std::strcmp(n,std::getenv("MOCK_TARGET"))==0;
 if(m==(H)0x99){event("get_replacement");assert(target);if(is("replacement_get_error"))return 13;*f=is("replacement_null")?nullptr:is("replacement_alias")?(H)0x1234:(H)0x5678;errno=ENOTTY;return 0;}
 assert(m==(H)0x88||m==(H)0x89);event(target?"get_original":"get_other");
 if(target&&is("original_error"))return 11;
 if(target&&is("lookup_miss_then_success")&&target_lookups++==0){assert(m==(H)0x89);errno=ENOENT;return 500;}
 *f=target?(altered==2?(H)0x2345:(H)0x1234):(H)0x321;
 if(target&&is("original_null"))*f=nullptr;
 errno=EAGAIN;return 0;
}
extern "C" int hipModuleLoadData(H*m,const void*image){event("load");assert(image&&std::memcmp(image,"\177ELF",4)==0);
 if(is("load_error"))return 12;
 if(is("load_mutates"))static_cast<unsigned char*>(const_cast<void*>(image))[100]^=1;
 if(is("reentrant")){H f=nullptr;hipModuleGetFunction(&f,(H)0x88,"loader_dependency");}
 *m=is("load_null")?nullptr:is("load_alias")?(H)0x88:(H)0x99;return 0;
}
extern "C" int hipCtxGetCurrent(H*c){event("context");if(is("context_error"))return 14;*c=altered==1?(H)0xbb:(H)0xaa;return 0;}
extern "C" int hipGetDevice(int*d){event("device");if(is("device_error"))return 15;*d=altered==3?1:0;return 0;}
static void check_buffer(void**p,void**e){assert(!p&&e&&e[0]==(H)1&&e[2]==(H)2&&e[4]==(H)3);assert(*static_cast<size_t*>(e[3])==212);
 auto*b=static_cast<unsigned char*>(e[1]);for(size_t i=0;i<212;++i)assert(b[i]==static_cast<unsigned char>(i*7+3));}
extern "C" int hipModuleLaunchKernel(H f,unsigned gx,unsigned gy,unsigned gz,unsigned bx,unsigned by,unsigned bz,unsigned sh,H st,void**p,void**e){
 event("launch");assert(f==(std::getenv("PROJECTION_NATIVE_OVERRIDE")?(H)0x5678:(H)0x1234));assert(gx==2&&gy==3&&gz==4&&bx==5&&by==6&&bz==7&&sh==8&&st==(H)0xab);check_buffer(p,e);errno=EDOM;return 17;}
extern "C" int hipExtModuleLaunchKernel(H f,uint32_t gx,uint32_t gy,uint32_t gz,uint32_t bx,uint32_t by,uint32_t bz,size_t sh,H st,void**p,void**e,H start,H stop,uint32_t flags){
 event("ext_launch");assert(f==(std::getenv("PROJECTION_NATIVE_OVERRIDE")?(H)0x5678:(H)0x1234));assert(gx==20&&gy==30&&gz==40&&bx==50&&by==60&&bz==70&&sh==((size_t(1)<<32)+8)&&st==(H)0xab&&start==(H)0xcc&&stop==(H)0xdd&&flags==11);check_buffer(p,e);errno=ERANGE;return 23;}
extern "C" int hipMemcpyAsync(void*d,const void*s,size_t n,int kind,H st){event("copy");assert(d==(H)0xde&&s&&n==8&&kind==1&&st==(H)0xab);errno=ERANGE;return 53;}
extern "C" int hipModuleUnload(H){event("unload");return 0;}
extern "C" int hipDeviceReset(){event("reset");return 0;}
'''

MAIN = r'''
#include <cassert>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>
using H=void*;
extern "C" int hipModuleGetFunction(H*,H,const char*);
extern "C" int hipModuleLaunchKernel(H,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,H,void**,void**);
extern "C" int hipExtModuleLaunchKernel(H,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,size_t,H,void**,void**,H,H,uint32_t);
extern "C" int hipMemcpyAsync(void*,const void*,size_t,int,H);
extern "C" int hipModuleUnload(H);
extern "C" int hipDeviceReset();
extern "C" void mock_change(int);
int main(int argc,char**argv){assert(argc==2);std::string test=argv[1],target=std::getenv("MOCK_TARGET");H f=nullptr;
 if(test=="original_error"){assert(hipModuleGetFunction(&f,(H)0x88,target.c_str())==11&&!f);return 0;}
 if(test=="lookup_miss_then_success"){
  assert(hipModuleGetFunction(&f,(H)0x89,target.c_str())==500&&!f);assert(errno==ENOENT);
 }
 if(test=="concurrent"){
  std::vector<std::thread> threads;for(int i=0;i<12;++i)threads.emplace_back([&]{H t=nullptr;assert(hipModuleGetFunction(&t,(H)0x88,target.c_str())==0&&t==(H)0x5678);});
  for(auto&t:threads)t.join();return 0;
 }
 if(test=="success"||test=="disabled"||test=="near_only"){
  for(const auto& n:{target.substr(0,target.size()-1),target+"suffix",std::string("other")}){
   assert(hipModuleGetFunction(&f,(H)0x88,n.c_str())==0&&f==(H)0x321);
  }
  if(test=="near_only")return 0;
 }
 assert(hipModuleGetFunction(&f,(H)0x88,target.c_str())==0);assert(errno==EAGAIN);
 if(test=="env_changed")setenv("PROJECTION_NATIVE_OVERRIDE","a_reuse_off",1);
 if(test=="context_changed")mock_change(1);
 if(test=="device_changed")mock_change(3);
 if(test=="handle_changed")mock_change(2);
 if(test=="control_changed")setenv("TRITON_FULL_AUTOTUNE","1",1);
 if(test=="unload")return hipModuleUnload((H)0x99);
 if(test=="reset")return hipDeviceReset();
 if(test=="fork"){
  pid_t pid=fork();assert(pid>=0);if(pid==0){hipModuleGetFunction(&f,(H)0x88,target.c_str());_exit(99);}
  int status=0;assert(waitpid(pid,&status,0)==pid&&WIFEXITED(status)&&WEXITSTATUS(status)==125);return 0;
 }
 if(test=="env_changed"||test=="handle_changed"||test=="module_changed"){
  hipModuleGetFunction(&f,test=="module_changed"?(H)0x89:(H)0x88,target.c_str());return 99;
 }
 if(test=="success"||test=="disabled")assert(hipModuleGetFunction(&f,(H)0x88,target.c_str())==0);
 unsigned char b[212];for(size_t i=0;i<sizeof(b);++i)b[i]=i*7+3;size_t size=sizeof(b);void*extra[]={(H)1,b,(H)2,&size,(H)3};
 if(test=="bypass")f=(H)0x1234;
 assert(hipModuleLaunchKernel(f,2,3,4,5,6,7,8,(H)0xab,nullptr,extra)==17);assert(errno==EDOM);
 assert(hipExtModuleLaunchKernel(f,20,30,40,50,60,70,(size_t(1)<<32)+8,(H)0xab,nullptr,extra,(H)0xcc,(H)0xdd,11)==23);assert(errno==ERANGE);
 unsigned char small[8]={0,1,2,3,4,5,6,255};assert(hipMemcpyAsync((H)0xde,small,8,1,(H)0xab)==53);assert(errno==ERANGE);
 return 0;
}
'''


class NativeOverrideTests(unittest.TestCase):
    def test_all_CPU_cases(self):
        source = Path(__file__).with_name('replace_projection_native_elf.cpp')
        frozen = source.with_name('trace_projection_hip_launches_v2.cpp')
        manifest = json.loads((LOGS / 'native_projection_disassembly_full_v1/manifest.json').read_text())
        target = manifest['records'][0]['name']
        self.assertEqual(re.search(r'constexpr char target_symbol\[\] =\s*"([^"]+)";', source.read_text())[1], target)
        self.assertEqual(hashlib.sha256(frozen.read_bytes()).hexdigest(), FROZEN_SHA)
        original = Path(manifest['source_elf'])
        derived = LOGS / 'wmma_reuse_static_v1/MT128x240x128.matrix_A_reuse_off.static_only.elf'
        self.assertEqual(hashlib.sha256(original.read_bytes()).hexdigest(), ORIGINAL_SHA)
        self.assertEqual(hashlib.sha256(derived.read_bytes()).hexdigest(), DERIVED_SHA)
        keep = os.environ.get('PROJECTION_NATIVE_OVERRIDE_TEST_DIR')
        temporary = None if keep else tempfile.TemporaryDirectory()
        root = Path(keep or temporary.name)
        root.mkdir(parents=True, exist_ok=True)
        for name, value in [('mock.cpp', MOCK), ('main.cpp', MAIN)]:
            (root / name).write_text(value)
        commands = [
            ['g++', '-shared', '-fPIC', '-O2', '-std=c++17', '-Wall', '-Wextra', '-Werror', str(source), '-ldl', '-pthread', '-lcrypto', '-o', str(root / 'replace_projection_native_elf.so')],
            ['g++', '-shared', '-fPIC', '-O2', '-std=c++17', str(root / 'mock.cpp'), '-pthread', '-o', str(root / 'mock.so')],
            ['g++', '-O2', '-std=c++17', str(root / 'main.cpp'), str(root / 'mock.so'), '-pthread', '-o', str(root / 'test')],
        ]
        for command in commands:
            subprocess.run(command, check=True, capture_output=True, text=True)
        cases = []
        def run(case, *, mode='original', path=original, expected=0, changes=None, test=None):
            trace, calls_path = root / (case + '.trace.jsonl'), root / (case + '.calls.txt')
            self.assertFalse(trace.exists() or calls_path.exists())
            env = os.environ.copy()
            for key in ('HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL'):
                env[key] = ''
            env.update(LD_PRELOAD=str(root / 'replace_projection_native_elf.so'),
                PROJECTION_NATIVE_OVERRIDE=mode, PROJECTION_NATIVE_OVERRIDE_ELF=str(path),
                PROJECTION_HIP_LAUNCH_TRACE=str(trace), MOCK_HIP_RECORD=str(calls_path),
                MOCK_TARGET=target, MOCK_CASE=case,
                AMDGCN_USE_BUFFER_OPS='0', TRITON_FULL_AUTOTUNE='0', TRITON_ALLOW_PIPELINING='0')
            for key, value in (changes or {}).items():
                if value is None:
                    env.pop(key, None)
                else:
                    env[key] = value
            result = subprocess.run([str(root / 'test'), test or case], env=env, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, expected, (case, result.stdout, result.stderr))
            records = [json.loads(s) for s in trace.read_text().splitlines()] if trace.exists() else []
            calls = calls_path.read_text().splitlines() if calls_path.exists() else []
            cases.append({'case':case,'exit_code':result.returncode,'mock_calls':calls,'stderr':result.stderr,
                          'trace':str(trace),'trace_sha256':hashlib.sha256(trace.read_bytes()).hexdigest() if trace.exists() else None})
            return records, calls
        for mode, path, sha in [('original', original, ORIGINAL_SHA), ('a_reuse_off', derived, DERIVED_SHA)]:
            records, calls = run('success_' + mode, mode=mode, path=path, test='success')
            self.assertEqual(calls.count('get_other'), 3)
            self.assertEqual(calls.count('get_original'), 2)
            self.assertEqual(calls.count('load'), 1)
            self.assertEqual(calls.count('get_replacement'), 1)
            for r in records:
                if r['event'] == 'native_override_function_return':
                    self.assertEqual((r['original_function'], r['replacement_function'], r['sha256']), ('0x1234', '0x5678', sha))
            launches = [r for r in records if r['event'] == 'launch_begin']
            self.assertEqual(len(launches), 2)
            expected_hex = bytes((i * 7 + 3) & 255 for i in range(212)).hex()
            for r in launches:
                self.assertEqual((r['function'], r['name']), ('0x5678', target))
                self.assertEqual(r['argument_capture']['buffer_hex'], expected_hex)
                self.assertEqual(r['argument_capture']['buffer_size'], 212)
            self.assertEqual(launches[1]['shared_bytes'], (1 << 32) + 8)
            self.assertEqual([r['hip_error'] for r in records if r['event'] == 'launch_return'], [17, 23])
            self.assertEqual([r['source_capture']['source_hex'] for r in records if r['event'] == 'h2d_copy_begin'], ['00010203040506ff'])
        records, calls = run('disabled', changes={'PROJECTION_NATIVE_OVERRIDE':None, 'PROJECTION_NATIVE_OVERRIDE_ELF':None})
        self.assertNotIn('load', calls)
        self.assertNotIn('context', calls)
        self.assertFalse(any(r['event'].startswith('native_override') for r in records))
        _, calls = run('near_only', path=root / 'nonexistent.elf')
        self.assertEqual(calls, ['get_other'] * 3)
        records, calls = run('concurrent')
        self.assertEqual((calls.count('get_original'), calls.count('load'), calls.count('get_replacement')), (12, 1, 1))
        self.assertEqual([r['retrieval_id'] for r in records if r['event'] == 'native_override_function_return'], list(range(1,13)))
        records, calls = run('original_error')
        self.assertEqual(calls, ['get_original'])
        self.assertEqual([r['hip_error'] for r in records if r['event'] == 'native_override_original_lookup_error'], [11])
        records, calls = run('lookup_miss_then_success')
        self.assertEqual((calls.count('get_original'), calls.count('load'), calls.count('get_replacement')), (2, 1, 1))
        self.assertEqual(calls[:2], ['get_original', 'get_original'])
        errors = [r for r in records if r['event'] == 'native_override_original_lookup_error']
        self.assertEqual([(r['original_module'], r['hip_error'], r['forwarded_as_original_error']) for r in errors], [('0x89', 500, True)])
        replacements = [r for r in records if r['event'] == 'native_override_function_return']
        self.assertEqual([(r['original_module'], r['retrieval_id']) for r in replacements], [('0x88', 2)])
        self.assertEqual([r['function'] for r in records if r['event'] == 'launch_begin'], ['0x5678'] * 2)
        for case in ('original_null','load_error','load_null','load_alias','load_mutates',
                     'replacement_get_error','replacement_null','replacement_alias','context_error','device_error','reentrant',
                     'env_changed','context_changed','device_changed','control_changed','handle_changed','module_changed','bypass','unload','reset'):
            records, calls = run(case, expected=125)
            self.assertTrue(any(r['event'] == 'native_override_fatal' for r in records), case)
            self.assertNotIn('launch', calls, case)
            self.assertNotIn('ext_launch', calls, case)
            self.assertLessEqual(calls.count('load'), 1)
        _, calls = run('fork')
        self.assertEqual(calls.count('get_original'), 1)
        self.assertEqual(calls.count('load'), 1)
        wrong = root / 'wrong_sha.elf'
        raw = bytearray(original.read_bytes()); raw[100] ^= 1; wrong.write_bytes(raw)
        for case, path in [('bad_hash', wrong), ('wrong_arm', derived), ('missing_file', root / 'absent.elf')]:
            _, calls = run(case, path=path, expected=125)
            self.assertEqual(calls.count('get_original'), 1)
            self.assertNotIn('load', calls)
        small = root / 'truncated.elf'; small.write_bytes(b'\x7fELF')
        _, calls = run('wrong_size', path=small, expected=125)
        self.assertNotIn('load', calls)
        for case, changes in [
            ('bad_mode', {'PROJECTION_NATIVE_OVERRIDE':'typo'}),
            ('missing_mode', {'PROJECTION_NATIVE_OVERRIDE':None}),
            ('missing_path', {'PROJECTION_NATIVE_OVERRIDE_ELF':None}),
            ('missing_trace', {'PROJECTION_HIP_LAUNCH_TRACE':None}),
            ('missing_zero', {'AMDGCN_USE_BUFFER_OPS':None}),
        ]:
            _, calls = run(case, changes=changes, expected=125)
            self.assertEqual(calls, [])
        self.assertEqual(hashlib.sha256(frozen.read_bytes()).hexdigest(), FROZEN_SHA)
        self.assertEqual(hashlib.sha256(original.read_bytes()).hexdigest(), ORIGINAL_SHA)
        self.assertEqual(hashlib.sha256(derived.read_bytes()).hexdigest(), DERIVED_SHA)
        report = {'format':'native_projection_override_CPU_tests_v2','status':'PASS','GPU_initialized':False,
                  'unsuccessful_original_lookups_forwarded_without_module_load':True,
                  'HIP_provider':'CPU mock only; real pinned ELF bytes hashed but never loaded by a GPU runtime',
                  'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                  'frozen_tracer_sha256':FROZEN_SHA,'test_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'compiled_sha256':hashlib.sha256((root/'replace_projection_native_elf.so').read_bytes()).hexdigest(),
                  'commands':commands,'case_count':len(cases),'cases':cases}
        (root / 'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if temporary:
            temporary.cleanup()


if __name__ == '__main__':
    unittest.main()
