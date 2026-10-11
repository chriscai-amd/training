ARM_LIB_DIR for roadmap L1 stage 1 + T8 (NN only, NT wgrads pinned by the catalog): identical to rocm-10.2_nnonly_v1
(ROCm 10.2 (/opt/rocm-10.2) libhipblaslt 1.5 + tensilelite-host + origami + rocroller + libomp + sysdeps_z, gfx1250
kernel folder symlinked) EXCEPT the bf16 NT (Ailk_Bjlk BB_BB_HA_Bias_SAV) catalog .dat, which is v1's filtered copy
(Origami Prediction row removed, solutions pruned to the GridBased stub's sol 102 MT32x16x32 / 103 MT256x128x64, SK0)
PLUS 19 new GridBased table entries (35 -> 54). No solution, kernel, predicate, row or property changed; .co untouched.
Why: the stub sends every DLRMv4 NT weight gradient (K = T = jagged token count, 1.59-2.53 M) to sol 102 (MT32x16x32,
slow) except (256,512). The bench pin to 103 gave 684.7 -> 502.7 ms/step at K=2,097,152, but the
HIPBLASLT_TUNING_OVERRIDE_FILE route matches exact (m,n,k) and never fires with jagged K. GridBased matching
(rocm-libraries 5911365c PropertyMatching.hpp, default env) = M ceiling, then N ceiling in that M block, then batch,
then nearest K; each added (M,N,1) group has ONE entry, so the choice is K-independent (designed for K 1.0M-3.2M).
Added entries ([M, N, batch, K] -> sol); "guard" rows keep neighbouring shapes on their old result (102):
  [23, 1, 1, 1]          -> 102   (guard)
  [24, 255, 1, 1]        -> 102   (guard)
  [24, 256, 1, 1]        -> 103   (TARGET)
  [24, 1048576, 1, 1]    -> 102   (guard)
  [511, 1, 1, 1]         -> 102   (guard)
  [512, 255, 1, 1]       -> 102   (guard)
  [512, 256, 1, 1]       -> 103   (TARGET)
  [512, 1535, 1, 1]      -> 102   (guard)
  [512, 1536, 1, 1]      -> 103   (TARGET)
  [512, 1048576, 1, 1]   -> 102   (guard)
  [1023, 1, 1, 1]        -> 102   (guard)
  [1024, 255, 1, 1]      -> 102   (guard)
  [1024, 256, 1, 1]      -> 103   (TARGET)
  [1024, 1048576, 1, 1]  -> 102   (guard)
  [2047, 1, 1, 1]        -> 102   (guard)
  [2048, 511, 1, 1]      -> 102   (guard)
  [2048, 512, 1, 1]      -> 103   (TARGET)
  [2048, 1048576, 1, 1]  -> 102   (guard)
  [1048576, 1, 1, 1]     -> 102   (guard)
Targets switched to 103 in this build: (2048,512) G20 [32 WG on 103] (512,1536) G19 [24 WG on 103] (1024,256) G29 [8 WG on 103] (512,256) G31 [4 WG on 103] (24,256) G27 [2 WG on 103]
Effect (gridmatch.py simulation, cross-checked against the C++ matcher compiled from 5911365c; see
/home/chcai/t8_work/simulate_v1_v2.txt and T8_REPORT.md): the targets above go 102 -> 103 at every K; (256,512) G25
stays 103; G17 (516,512,k1024) and G23 (512,512,batch 8) stay 102. Only other changes (default batch flattening,
none in the workload): batched shapes whose flattened (m*b,n) if m>n else (m,n*b) is a target.
Keep TENSILE_GRIDBASED_KDTREE, TENSILE_GRIDBASED_BATCH_EXP, TENSILE_METRIC and TENSILE_SOLUTION_INDEX UNSET (with
KDTREE=1, v2 would send G25 to 102 and G17 to 103). Keep HIPBLASLT_TENSILE_LIBPATH set explicitly (there is no
gfx1250v0 folder here).
Kernel note: sol 103 here is the 10.2-built MT256x128x64 SK0 kernel (676 VGPRs, no split-K) from the symlinked 10.2 .co.
It is the same kernel binary the M2 bench pin measured (that bench ran on rocm-10.2_nnonly_v1 with the 10.2 kernel name
in tuning_override.csv); the 684.7 -> 502.7 ms/step figure is a 5-shape aggregate, not per shape.
RISK: sol 103 (MT256x128x64, 676 VGPRs) is the kernel of the A0 corrupted-weight-gradient (DW) failures
(docs/mi450_a0/mi450_a0.md L923-924: isolated zero-DY DW fails at call 196 on 103; sol 102 passes 2x1000 calls; the
mitigation was to redirect DW 103 -> 102). The faulty CWSR handler suspected there is still installed on a37-1. This
build raises sol-103 launches from 3 to 12 per training step. Use only for guarded <=200-step A/Bs with grad-finiteness
and loss checks, after the single-GEMM checks in /home/chcai/t8_work/T8_REPORT.md; no long runs until the handler is fixed.
Built by /home/chcai/t8_work/make_v2.py (sha256 of the .dat below). Do NOT rebuild in place; make a _v3 directory.
sha256 1053790f4c7ba8fe07b9f93b69baeaa02488035545b810168c828dc55630accc  hipblaslt/library/gfx1250/TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Ailk_Bjlk_Cijk_Dijk_gfx1250.dat.zlib
