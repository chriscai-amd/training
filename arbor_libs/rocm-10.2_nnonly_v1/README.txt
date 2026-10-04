ARM_LIB_DIR for roadmap L1 stage 1 (NN only): ROCm 10.2 (/opt/rocm-10.2) libhipblaslt 1.5 + tensilelite-host + origami + rocroller
+ libomp + sysdeps_z, gfx1250 kernel folder symlinked EXCEPT the bf16 NT (Ailk_Bjlk BB_BB_HA_Bias_SAV) catalog .dat, which is a
filtered copy: its Origami Prediction row (422 SK3 stream-K kernels) is removed and its solutions pruned to the GridBased stub's
(sol 102 MT32x16x32 / 103 MT256x128x64, SK0) -- so NT wgrads stay on the stock kernels (F7 hazard), NN gets the 10.2 Origami pool.
The .co is untouched (the pruned kernels are unreachable). Do NOT rebuild in place; make a _v2 directory.
