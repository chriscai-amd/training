ARM_LIB_DIR for roadmap L1 stage 0: ROCm 10.1 (/opt/rocm-2407757, hipBLASLt 1.4 git 6230cb1c) libhipblaslt + origami + rocroller + gfx1250 kernel folder.
No libamdhip64: resolves to the image's HIP 7.14 (all 38 versioned hip_* symbols the 10.1 libs need are exported by it, checked 2026-10-04).
Do NOT rebuild in place; make a new _vN directory.
