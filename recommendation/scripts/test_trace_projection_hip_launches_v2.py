"""CPU-only ABI/transparency and bounded-capture tests; links a mock HIP API."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import tempfile
import unittest

MOCK = r'''
#include <cstdint>
#include <cstddef>
#include <cerrno>
#include <cstring>
#include <vector>
using H=void*;
static int calls[4]={};
static void **ep=nullptr,**ee=nullptr;
static void *ed=nullptr; static const void *es=nullptr;
static size_t en=0; static int ek=0;
static unsigned char* watched=nullptr; static std::vector<unsigned char> expected;
static bool mutate=false;
extern "C" void expect_launch(void**p,void**e){ep=p;ee=e;}
extern "C" void expect_copy(void*d,const void*s,size_t n,int k){ed=d;es=s;en=n;ek=k;}
extern "C" void watch(unsigned char*p,size_t n,bool change){watched=p;expected.assign(p,p+n);mutate=change;}
static bool check(){
 bool good=!watched||std::memcmp(watched,expected.data(),expected.size())==0;
 if(good&&watched&&mutate&&!expected.empty())watched[0]=0xee;
 watched=nullptr;expected.clear();return good;
}
extern "C" int hipModuleGetFunction(H*f,H m,const char*n){++calls[0];if(m!=(H)0x88||!n)return 9;*f=(H)0x123400;return 0;}
extern "C" int hipModuleLaunchKernel(H f,unsigned gx,unsigned gy,unsigned gz,unsigned bx,unsigned by,unsigned bz,unsigned s,H stream,void**p,void**e){
 ++calls[1];bool ok=f==(H)0x123400&&gx==2&&gy==3&&gz==4&&bx==5&&by==6&&bz==7&&s==8&&stream==(H)0x99&&p==ep&&e==ee&&check();errno=EDOM;return ok?17:91;
}
extern "C" int hipExtModuleLaunchKernel(H f,uint32_t gx,uint32_t gy,uint32_t gz,uint32_t bx,uint32_t by,uint32_t bz,size_t s,H stream,void**p,void**e,H start,H stop,uint32_t flags){
 ++calls[2];bool ok=f==(H)0x123400&&gx==20&&gy==30&&gz==40&&bx==50&&by==60&&bz==70&&s==((size_t(1)<<32)+8)&&stream==(H)0x99&&p==ep&&e==ee&&start==(H)0xcc&&stop==(H)0xdd&&flags==11&&check();errno=EDOM;return ok?23:92;
}
extern "C" int hipMemcpyAsync(void*d,const void*s,size_t n,int k,H stream){
 ++calls[3];bool ok=d==ed&&s==es&&n==en&&k==ek&&stream==(H)0x99&&check();errno=ERANGE;return ok?53:93;
}
extern "C" int count(int i){return calls[i];}
'''

MAIN = r'''
#include <cstdint>
#include <cstddef>
#include <cerrno>
#include <cstring>
#include <cassert>
using H=void*;
extern "C" int hipModuleGetFunction(H*,H,const char*);
extern "C" int hipModuleLaunchKernel(H,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,H,void**,void**);
extern "C" int hipExtModuleLaunchKernel(H,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,size_t,H,void**,void**,H,H,uint32_t);
extern "C" int hipMemcpyAsync(void*,const void*,size_t,int,H);
extern "C" void expect_launch(void**,void**);
extern "C" void expect_copy(void*,const void*,size_t,int);
extern "C" void watch(unsigned char*,size_t,bool);
extern "C" int count(int);
static H f;
static void ext(void**p,void**e){expect_launch(p,e);assert(hipExtModuleLaunchKernel(f,20,30,40,50,60,70,(size_t(1)<<32)+8,(H)0x99,p,e,(H)0xcc,(H)0xdd,11)==23);assert(errno==EDOM);}
static void copy(void*d,const void*s,size_t n,int k){expect_copy(d,s,n,k);assert(hipMemcpyAsync(d,s,n,k,(H)0x99)==53);assert(errno==ERANGE);}
int main(){
 char name[]={'C','i','j','k','_',34,92,char(255),0};assert(hipModuleGetFunction(&f,(H)0x88,name)==0);name[0]='X';
 unsigned char data[216];for(unsigned i=0;i<sizeof(data);++i)data[i]=(i*7+3)&255;
 unsigned char original[216];std::memcpy(original,data,sizeof(data));size_t n=sizeof(data);
 void* normal[]={(H)1,data,(H)2,&n,(H)3};watch(data,n,false);ext(nullptr,normal);assert(std::memcmp(data,original,n)==0);
 void* reversed[]={(H)2,&n,(H)1,data,(H)3};expect_launch(nullptr,reversed);watch(data,n,true);
 assert(hipModuleLaunchKernel(f,2,3,4,5,6,7,8,(H)0x99,nullptr,reversed)==17);assert(errno==EDOM);assert(data[0]==0xee);data[0]=original[0];
 size_t huge=4097;void* oversized[]={(H)1,(H)1,(H)2,&huge,(H)3};ext(nullptr,oversized);
 void* unknown[]={(H)0x55,(H)1,(H)3};ext(nullptr,unknown);
 void* duplicate[]={(H)1,data,(H)1,(H)1,(H)3};ext(nullptr,duplicate);
 void* incomplete[]={(H)3};ext(nullptr,incomplete);
 void* bad_size[]={(H)1,data,(H)2,(H)1,(H)3};ext(nullptr,bad_size);
 void* bad_buffer[]={(H)1,(H)1,(H)2,&n,(H)3};ext(nullptr,bad_buffer);
 void* params[]={(H)1};ext(params,(void**)1);
 ext(nullptr,(void**)1);
 size_t zero=0;void* empty[]={(H)1,nullptr,(H)2,&zero,(H)3};ext(nullptr,empty);
 ext(nullptr,nullptr);
 void* too_long[]={(H)1,data,(H)2,&n,(H)1};ext(nullptr,too_long);
 unsigned char small[8]={0,1,2,3,4,5,6,255};watch(small,sizeof(small),true);copy((H)0xdead,small,sizeof(small),1);assert(small[0]==0xee);
 copy((H)0xdead,(H)1,8,2);copy((H)0xdead,(H)1,4097,1);copy((H)0xdead,(H)1,8,4);
 copy((H)0xdead,(H)1,8,1);copy((H)0xdead,nullptr,0,1);
 assert(count(0)==1&&count(1)==1&&count(2)==12&&count(3)==6);
 return 0;
}
'''


class InterposerTest(unittest.TestCase):
    def test_cpu_forwarding_and_capture_protocol(self):
        source = Path(__file__).with_name('trace_projection_hip_launches_v2.cpp')
        keep = os.environ.get('PROJECTION_INTERPOSER_TEST_DIR')
        temporary = None if keep else tempfile.TemporaryDirectory()
        root = Path(keep or temporary.name)
        root.mkdir(parents=True, exist_ok=True)
        for name, text in [('mock.cpp', MOCK), ('main.cpp', MAIN)]:
            (root / name).write_text(text)
        commands = [
            ['g++', '-shared', '-fPIC', '-O2', '-std=c++17', '-Wall', '-Wextra', '-Werror', str(source), '-ldl', '-pthread', '-o', str(root / 'interposer.so')],
            ['g++', '-shared', '-fPIC', '-O2', '-std=c++17', str(root / 'mock.cpp'), '-o', str(root / 'mock.so')],
            ['g++', '-O2', '-std=c++17', str(root / 'main.cpp'), str(root / 'mock.so'), '-o', str(root / 'test')],
        ]
        for command in commands:
            subprocess.run(command, check=True, capture_output=True, text=True)
        env = os.environ.copy()
        for key in ['HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES']:
            env[key] = ''
        # Baseline and preload execute identical mocks, argument validation,
        # explicit post-original mutations, errno checks, and call counters.
        baseline = env.copy()
        baseline.pop('LD_PRELOAD', None)
        baseline.pop('PROJECTION_HIP_LAUNCH_TRACE', None)
        subprocess.run([str(root / 'test')], env=baseline, check=True)
        trace = root / 'trace.jsonl'
        trace.unlink(missing_ok=True)
        env.update(LD_PRELOAD=str(root / 'interposer.so'), PROJECTION_HIP_LAUNCH_TRACE=str(trace))
        subprocess.run([str(root / 'test')], env=env, check=True)
        records = [json.loads(line) for line in trace.read_text().splitlines()]
        launches = [r for r in records if r['event'] == 'launch_begin']
        returns = [r for r in records if r['event'] == 'launch_return']
        self.assertEqual(len(launches), 13)
        self.assertEqual(len(returns), 13)
        self.assertEqual([r['launch_id'] for r in launches], list(range(1, 14)))
        self.assertEqual([r['launch_id'] for r in returns], list(range(1, 14)))
        expected = bytes((i * 7 + 3) & 255 for i in range(216)).hex()
        for index in (0, 1):
            self.assertEqual(launches[index]['argument_capture']['status'], 'captured')
            self.assertEqual(launches[index]['argument_capture']['buffer_hex'], expected)
        reasons = [r['argument_capture'].get('reason') for r in launches]
        self.assertEqual(reasons, [None, None, 'buffer_exceeds_4096', 'unknown_extra_tag', 'duplicate_extra_tag', 'incomplete_extra_format', 'buffer_size_unreadable', 'buffer_unreadable', 'kernel_params_sizes_unknown', 'extra_tag_unreadable', None, 'no_extra', 'extra_format_too_long'])
        self.assertEqual(launches[10]['argument_capture']['buffer_hex'], '')
        self.assertTrue(all(r['name'] == 'Cijk_"\\\xff' for r in launches))
        self.assertEqual(launches[0]['shared_bytes'], (1 << 32) + 8)
        copies = [r for r in records if r['event'] == 'h2d_copy_begin']
        self.assertEqual(len(copies), 3)
        self.assertEqual(copies[0]['source_capture'], {'status': 'captured', 'source_hex': '00010203040506ff'})
        self.assertEqual(copies[1]['source_capture'], {'status': 'unreadable'})
        self.assertEqual(copies[2]['source_capture'], {'status': 'captured', 'source_hex': ''})
        self.assertEqual([r['hip_error'] for r in records if r['event'] == 'h2d_copy_return'], [53, 53, 53])
        report = {
            'format': 'hip_launch_interposer_v2_cpu_test_v1', 'outcome': 'PASS',
            'gpu_initialized': False, 'linked_api': 'CPU mock only',
            'launch_cases': 13, 'h2d_or_bypassed_copy_cases': 6,
            'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'test_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'interposer_sha256': hashlib.sha256((root / 'interposer.so').read_bytes()).hexdigest(),
            'trace_sha256': hashlib.sha256(trace.read_bytes()).hexdigest(),
            'checks': ['unmodified argument identity and values', 'exact original call counts and results', 'errno retained after original', 'source unchanged except deliberate original API mutation', 'captured before deliberate original API mutation', 'both extra pair orders', 'strict tag and size bounds', 'invalid host pointers skipped without fault', 'D2H/default/oversize copies bypass capture', 'name byte escaping and lifetime', '64-bit shared-memory metadata'],
        }
        (root / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        if temporary:
            temporary.cleanup()


if __name__ == '__main__':
    unittest.main()
