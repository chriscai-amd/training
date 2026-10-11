rocm-10.2_nnonly_v3/mix: hipBLASLt lib dir for M5 (split-T NT weight gradients), variant mix.
Identical to rocm-10.2_nnonly_v2 (same symlinks into /opt/rocm-10.2) EXCEPT the bf16 NT catalog
hipblaslt/library/gfx1250/TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Ailk_Bjlk_Cijk_Dijk_gfx1250.dat.zlib
whose GridBased table = v2 table + 67 added / 3 flipped entries:
  [24, 256, 1, 1]          flip -> 102 (was 103)
  [24, 4095, 1, 1]         guard -> 102
  [24, 4096, 1, 1]         target -> 102
  [24, 8191, 1, 1]         guard -> 102
  [24, 8192, 1, 1]         target -> 102
  [24, 16383, 1, 1]        guard -> 102
  [24, 16384, 1, 1]        target -> 102
  [24, 32767, 1, 1]        guard -> 102
  [24, 32768, 1, 1]        target -> 102
  [24, 65535, 1, 1]        guard -> 102
  [24, 65536, 1, 1]        target -> 102
  [256, 511, 1, 1]         guard -> 102
  [256, 1048576, 1, 1]     guard -> 102
  [256, 512, 1, 1]         target -> 102
  [256, 2047, 1, 1]        guard -> 102
  [256, 2048, 1, 1]        target -> 102
  [256, 4095, 1, 1]        guard -> 102
  [256, 4096, 1, 1]        target -> 102
  [256, 8191, 1, 1]        guard -> 102
  [256, 8192, 1, 1]        target -> 102
  [256, 16383, 1, 1]       guard -> 102
  [256, 16384, 1, 1]       target -> 102
  [256, 32767, 1, 1]       guard -> 102
  [256, 32768, 1, 1]       target -> 102
  [256, 65535, 1, 1]       guard -> 102
  [256, 65536, 1, 1]       target -> 102
  [512, 256, 1, 1]         flip -> 102 (was 103)
  [512, 3071, 1, 1]        guard -> 102
  [512, 3072, 1, 1]        target -> 103
  [512, 6143, 1, 1]        guard -> 102
  [512, 6144, 1, 1]        target -> 103
  [512, 12287, 1, 1]       guard -> 102
  [512, 12288, 1, 1]       target -> 103
  [512, 15359, 1, 1]       guard -> 102
  [512, 15360, 1, 1]       target -> 103
  [512, 24575, 1, 1]       guard -> 102
  [512, 24576, 1, 1]       target -> 103
  [512, 30719, 1, 1]       guard -> 102
  [512, 30720, 1, 1]       target -> 103
  [1024, 256, 1, 1]        flip -> 102 (was 103)
  [2048, 255, 1, 1]        guard -> 102
  [2048, 256, 1, 1]        target -> 102
  [4095, 1, 1, 1]          guard -> 102
  [4096, 1048576, 1, 1]    guard -> 102
  [4096, 255, 1, 1]        guard -> 102
  [4096, 256, 1, 1]        target -> 102
  [4096, 511, 1, 1]        guard -> 102
  [4096, 512, 1, 1]        target -> 103
  [8191, 1, 1, 1]          guard -> 102
  [8192, 1048576, 1, 1]    guard -> 102
  [8192, 255, 1, 1]        guard -> 102
  [8192, 256, 1, 1]        target -> 102
  [8192, 511, 1, 1]        guard -> 102
  [8192, 512, 1, 1]        target -> 103
  [16383, 1, 1, 1]         guard -> 102
  [16384, 1048576, 1, 1]   guard -> 102
  [16384, 255, 1, 1]       guard -> 102
  [16384, 256, 1, 1]       target -> 102
  [16384, 511, 1, 1]       guard -> 102
  [16384, 512, 1, 1]       target -> 103
  [32767, 1, 1, 1]         guard -> 102
  [32768, 1048576, 1, 1]   guard -> 102
  [32768, 255, 1, 1]       guard -> 102
  [32768, 256, 1, 1]       target -> 102
  [32768, 511, 1, 1]       guard -> 102
  [32768, 512, 1, 1]       target -> 103
  [65535, 1, 1, 1]         guard -> 102
  [65536, 1048576, 1, 1]   guard -> 102
  [65536, 255, 1, 1]       guard -> 102
  [65536, 256, 1, 1]       target -> 102
Solution map per wgrad shape: {'G20': 103, 'G19': 103, 'G25': 102, 'G29': 102, 'G31': 102, 'G27': 102}
Effect (gridmatch.py, see /home/chcai/m5_work/verify_mix.txt): the batch-1 wgrads and every
batched M5 problem (m, n, batch=s) with s in the candidate list select the mapped sol at any K; no
other (m, n) changes (exhaustive boundary grid + 300k random sweep). Candidate s per shape:
G20 [2, 4, 8, 16], G19 [2, 4, 8, 10, 16, 20], G25 [4, 8, 16, 32, 64, 128], G29 [2, 4, 8, 16, 32, 64], G31 [4, 8, 16, 32, 64, 128], G27 [16, 32, 64, 128, 256]
UNTESTED ON A GPU. Sol 103 (676 VGPRs) is the A0 DW-corruption kernel under the faulty CWSR handler.
Built by /home/chcai/m5_work/make_v3.py. sha256 32e7a28c427e6b0477915c6ef604afcea335639a70354c7c9368f6560ebb85cc
Keep TENSILE_GRIDBASED_KDTREE / _BATCH_EXP / TENSILE_METRIC / TENSILE_SOLUTION_INDEX unset.
