M5 (split-T NT weight gradients) GPU check kit -- NOT RUN. Built 2026-10-11 on CPU only; revised after
review (2026-10-11). Summary and full plan: /home/chcai/m5_work/M5_REPORT.md (section 8).
Supersedes /home/chcai/m5_work/gpu_kit.
a37-1 rules: one GEMM / one shape per process, idle GPUs (no other KFD process -- the MES repro loop
must be stopped first), preflight, live dmesg watch; stop everything on exit code 3 (fault).

Lib dirs (host / container), each with m5_splits.json (per-shape sol for every s in 1..256):
  /home/chcai/training/arbor_libs/rocm-10.2_nnonly_v3/s103 = /armlibs/rocm-10.2_nnonly_v3/s103  all six on sol 103
  /home/chcai/training/arbor_libs/rocm-10.2_nnonly_v3/s102 = /armlibs/rocm-10.2_nnonly_v3/s102  all six on sol 102
  /home/chcai/training/arbor_libs/rocm-10.2_nnonly_v3/mix  = /armlibs/rocm-10.2_nnonly_v3/mix   G20/G19 103, rest 102
Overlays (repo gin/mi450_hipblaslt/): m5_split_t.gin (s103), m5_split_t_s102.gin, m5_split_t_mix.gin.

== Stage 1: single GEMMs (hipblaslt-bench via gemm_watch.sh), per variant, per T ==
Files: m5_bench_T2097152.yaml (40 lines), m5_bench_T2533005.yaml (63 lines); the .index.txt next to each
maps line -> shape, s, K, expected kernel per variant. Do NOT set T= (gemm_watch would rewrite the literal
2097152, which also occurs inside a stride: T2097152 line 40, G27 s=256 stride_b). Generator: gen_bench.py.
A 1-row tail (T % s == 1) is an elementwise fp32 addr_ in the helper -> no bench line (listed in the index).
  cd /home/chcai/a37-1_env
  ./gemm_watch.sh /home/chcai/training/arbor_libs/rocm-10.2_nnonly_v3/s102 0 /home/chcai/m5_work/gpu_check/runs/bench_s102_T2097152 /home/chcai/m5_work/gpu_check/m5_bench_T2097152.yaml
  ./gemm_watch.sh /home/chcai/training/arbor_libs/rocm-10.2_nnonly_v3/s102 0 /home/chcai/m5_work/gpu_check/runs/bench_s102_T2533005 /home/chcai/m5_work/gpu_check/m5_bench_T2533005.yaml
  ./gemm_watch.sh /home/chcai/training/arbor_libs/rocm-10.2_nnonly_v3/s103 0 /home/chcai/m5_work/gpu_check/runs/bench_s103_T2097152 /home/chcai/m5_work/gpu_check/m5_bench_T2097152.yaml
  ./gemm_watch.sh /home/chcai/training/arbor_libs/rocm-10.2_nnonly_v3/s103 0 /home/chcai/m5_work/gpu_check/runs/bench_s103_T2533005 /home/chcai/m5_work/gpu_check/m5_bench_T2533005.yaml
(s102 first: it never runs sol 103. mix = s103 lines of G20/G19 + s102 lines of the rest: no own run.)
Check in summary.txt: every bf16 line MT32x16x32 (s102) / MT256x128x64 (s103), fp32 tail lines MT32x32x16
(SS catalog sol 2821); any other kernel = stop and re-check the catalog. The s=1 line of the s103 file is
the T8 baseline; the s=1 line of the s102 file is sol 102 (NOT T8).
Per shape and variant pick s = argmin(bmm us + tail us + reduce); reduce ~= s*D*O*2 B read + D*O*(4+4+2) B
at ~3 TB/s plus ~10 us per extra launch (3 for r=0, 4 for r=1, 6 for r>=2): <= ~0.05 ms. On 103 prefer
fewer WGs when within 5 %. s must be pinned by the lib dir (m5_splits.json); triton_addmm refuses others.

== Stage 2: numerics through the actual M5 function (splitt_loop.py via run_splitt_check.sh) ==
  run_splitt_check.sh <dev> <s103|s102|mix> <shape> <s> <T> <zero|rand> [calls=1000] [iid|drift|halves|users]
For each variant to be A/B'd and each CHOSEN (shape, s):
  1. zero-DY 1000 calls at T=2097152 (batched kernels; partials checked every call = 2x the training
     launches of that kernel),
  2. rand iid 200 calls at T=2533005 (tail path; determinism; ratio <= 1.6 vs batch-1 on this lib),
  3. rand drift 50 calls at T=2533005 (cancelling chunks: error must follow the sqrt(1+q^2) model and the
     elementwise bound -- shows the WORST-case numerics of bf16 partials on the real kernels).
  cd /home/chcai/m5_work/gpu_check
  ./run_splitt_check.sh 0 s102 G25 4   2097152 zero 1000
  ./run_splitt_check.sh 0 s102 G25 4   2533005 rand 200
  ./run_splitt_check.sh 0 s102 G25 4   2533005 rand 50 drift
  ./run_splitt_check.sh 0 s103 G20 8   2097152 zero 1000
  ./run_splitt_check.sh 0 s103 G20 8   2533005 rand 200
  ./run_splitt_check.sh 0 s103 G20 8   2533005 rand 50 drift
  ... (s103 overlay: G19 10, G25 64, G29 32, G31 64, G27 128; s102 overlay: G20 4, G29 2, G31 4, G27 64;
  replace with the Stage-1 choices). Pass = "PASS", exit 0. The check fails (exit 2) unless call 1 shows
  exactly one hipBLASLt bf16 NT launch with the manifest's tile, the fp32 MT32x32x16 tail iff r >= 2, no
  rocBLAS / other GEMM, and the process-default BLAS backend is hipBLASLt (no forcing, as in training).

== Stage 3: E2E ABBA, 200 steps each (run_ab.sh; extra args go to docker run) ==
A = M1+T8 (v2 lib, NT wgrads batch-1 on sol 103), B = M1+T8+M5 (v3/s103 lib = T8 pins + split-T).
  cd /home/chcai/a37-1_env
  ./run_ab.sh M1T8_A1   /t8/m1_t8_lib.gin -v /home/chcai/t8_work/gpu_check:/t8:ro
  ./run_ab.sh M1T8M5_B1 m5_split_t.gin
  ./run_ab.sh M1T8M5_B2 m5_split_t.gin
  ./run_ab.sh M1T8_A2   /t8/m1_t8_lib.gin -v /home/chcai/t8_work/gpu_check:/t8:ro
Stop at the first exit 3. Per-shape s chosen in Stage 1 WITHOUT editing the repo (a pre-set env var wins
over gin): append e.g.
  -e HSTU_SPLITT_WGRAD_SPLITS=2048x512:8,512x1536:10,256x512:64,1024x256:32,512x256:64,24x256:128
to the B lines. Keys = hipBLASLt MxN: G20 2048x512, G19 512x1536, G25 256x512, G29 1024x256, G31 512x256,
G27 24x256. An s the lib dir does not pin makes the ranks fail at import ("not pinned in the hipBLASLt
catalog") -- by design. Setting -e HSTU_SPLITT_WGRAD=... alone with a non-M5 overlay is a no-op (logged
"nothing is split").
Other B arms: m5_split_t_mix.gin (6 sol-103 launches/step instead of 12), m5_split_t_s102.gin (no sol 103
in the NT wgrads at all).
After each B arm check train.log (all ranks):
  - "M5: split factors ... verified against catalog <variant>" once per rank;
  - six "M5 split-T wgrad m x n = ..." lines per rank, each "expected sol 103|102" and "preferred BLAS
    _BlasBackend.Cublaslt"; NO "configured but not applied" line;
  - NO "bgemm_internal_cublaslt error" / "falling back" line (torch's silent hipBLASLt -> rocBLAS fallback):
      grep -n "bgemm_internal_cublaslt\|cublaslt error\|configured but not applied" <run>/train.log
Kernel trace (one extra B run, not timed): OUTPUT_TRACE=1 ./run_ab.sh M1T8M5_TR m5_split_t.gin, then
  python3 /home/chcai/a37-1_env/trace_kernels.py <trace.json> 40
must show Cijk_Ailk_Bjlk_BBS_..._MT256x128x64 at 12 calls/step with grid ~240-256 WGs (s103 overlay), no
batch-1 MT256x128x64 with 2..32 WGs, and no rocBLAS kernel.
Numerics in training (one extra B run, not timed): -e HSTU_SPLITT_AUDIT_EVERY=50 logs per shape every 50th
call "M5 audit ... |split-orig|/|orig|, chunk cancellation q" (q ~1: 1.41x today's rounding error; q >> 1:
real gradients cancel across T-chunks -> prefer a small s or the f32-out variant, M5_REPORT section 6).
Compare perf.txt (step time), loss curve and grad-norm / finiteness vs the A arms.
