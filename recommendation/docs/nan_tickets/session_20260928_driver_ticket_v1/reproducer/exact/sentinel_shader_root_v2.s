.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.amdhsa_code_object_version 6
.text
.protected sentinel_00_0
.globl sentinel_00_0
.p2align 8
.type sentinel_00_0,@function
sentinel_00_0:
global_prefetch_b8 v0, s[0:1] scope:SCOPE_SE
v_nop
s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
s_load_b64 s[4:5], s[0:1], 0x0 nv
s_wait_kmcnt 0
s_mov_b32 s6, 0
s_setreg_b32 hwreg(HW_REG_WAVE_MODE, 12, 8), s6
s_nop 0
s_set_vgpr_msb 0x0000
s_nop 0
v_mov_b32 v1, 0x11000000
s_wait_idle
s_mov_b32 s7, 0x11000001
v_writelane_b32 v1, s7, 1
s_wait_idle
s_mov_b32 s7, 0x11000010
v_writelane_b32 v1, s7, 16
s_wait_idle
s_set_vgpr_msb 0x0041
s_nop 0
v_mov_b32 v1, 0x22000000
s_wait_idle
s_mov_b32 s7, 0x22000001
v_writelane_b32 v1, s7, 1
s_wait_idle
s_mov_b32 s7, 0x22000010
v_writelane_b32 v1, s7, 16
s_wait_idle
s_set_vgpr_msb 0x4182
s_nop 0
v_mov_b32 v1, 0x33000000
s_wait_idle
s_mov_b32 s7, 0x33000001
v_writelane_b32 v1, s7, 1
s_wait_idle
s_mov_b32 s7, 0x33000010
v_writelane_b32 v1, s7, 16
s_wait_idle
v_readlane_b32 s22, v1, 0
v_readlane_b32 s23, v1, 1
v_readlane_b32 s24, v1, 16
s_set_vgpr_msb 0x8241
s_nop 0
v_readlane_b32 s19, v1, 0
v_readlane_b32 s20, v1, 1
v_readlane_b32 s21, v1, 16
s_set_vgpr_msb 0x4100
s_nop 0
v_readlane_b32 s16, v1, 0
v_readlane_b32 s17, v1, 1
v_readlane_b32 s18, v1, 16
s_mov_b32 s8, 0x53545250
s_mov_b32 s9, 1
s_mov_b32 s10, 0x00
s_mov_b32 s11, 0
s_mov_b32 s40, 0x434f4d50
s_set_vgpr_msb 0x0000
s_nop 0
s_wait_idle
s_getreg_b32 s14, hwreg(HW_REG_WAVE_STATUS)
s_mov_b32 s38, exec_lo
s_getreg_b32 s12, hwreg(HW_REG_WAVE_MODE)
s_get_pc_i64 s[34:35]
.Lpcnext_sentinel_00_0:
s_add_co_u32 s34, s34, (.Lreturn_sentinel_00_0 - .Lpcnext_sentinel_00_0)
s_add_co_ci_u32 s35, s35, 0
s_nop 0
s_nop 0
.Lreturn_sentinel_00_0:
s_getreg_b32 s13, hwreg(HW_REG_WAVE_MODE)
s_mov_b32 s36, ttmp0
s_mov_b32 s37, ttmp1
s_mov_b32 s39, exec_lo
s_wait_idle
s_getreg_b32 s15, hwreg(HW_REG_WAVE_STATUS)
s_set_vgpr_msb 0x0000
s_nop 0
v_readlane_b32 s25, v1, 0
v_readlane_b32 s26, v1, 1
v_readlane_b32 s27, v1, 16
s_set_vgpr_msb 0x0041
s_nop 0
v_readlane_b32 s28, v1, 0
v_readlane_b32 s29, v1, 1
v_readlane_b32 s30, v1, 16
s_set_vgpr_msb 0x4182
s_nop 0
v_readlane_b32 s31, v1, 0
v_readlane_b32 s32, v1, 1
v_readlane_b32 s33, v1, 16
s_set_vgpr_msb 0x8200
s_nop 0
s_mov_b32 exec_lo, 1
v_mov_b32 v0, 0
v_mov_b32 v2, s8
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:0
s_wait_storecnt 0
v_mov_b32 v2, s9
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:4
s_wait_storecnt 0
v_mov_b32 v2, s10
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:8
s_wait_storecnt 0
v_mov_b32 v2, s11
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:12
s_wait_storecnt 0
v_mov_b32 v2, s12
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:16
s_wait_storecnt 0
v_mov_b32 v2, s13
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:20
s_wait_storecnt 0
v_mov_b32 v2, s14
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:24
s_wait_storecnt 0
v_mov_b32 v2, s15
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:28
s_wait_storecnt 0
v_mov_b32 v2, s16
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:32
s_wait_storecnt 0
v_mov_b32 v2, s17
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:36
s_wait_storecnt 0
v_mov_b32 v2, s18
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:40
s_wait_storecnt 0
v_mov_b32 v2, s19
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:44
s_wait_storecnt 0
v_mov_b32 v2, s20
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:48
s_wait_storecnt 0
v_mov_b32 v2, s21
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:52
s_wait_storecnt 0
v_mov_b32 v2, s22
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:56
s_wait_storecnt 0
v_mov_b32 v2, s23
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:60
s_wait_storecnt 0
v_mov_b32 v2, s24
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:64
s_wait_storecnt 0
v_mov_b32 v2, s25
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:68
s_wait_storecnt 0
v_mov_b32 v2, s26
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:72
s_wait_storecnt 0
v_mov_b32 v2, s27
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:76
s_wait_storecnt 0
v_mov_b32 v2, s28
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:80
s_wait_storecnt 0
v_mov_b32 v2, s29
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:84
s_wait_storecnt 0
v_mov_b32 v2, s30
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:88
s_wait_storecnt 0
v_mov_b32 v2, s31
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:92
s_wait_storecnt 0
v_mov_b32 v2, s32
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:96
s_wait_storecnt 0
v_mov_b32 v2, s33
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:100
s_wait_storecnt 0
v_mov_b32 v2, s34
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:104
s_wait_storecnt 0
v_mov_b32 v2, s35
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:108
s_wait_storecnt 0
v_mov_b32 v2, s36
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:112
s_wait_storecnt 0
v_mov_b32 v2, s37
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:116
s_wait_storecnt 0
v_mov_b32 v2, s38
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:120
s_wait_storecnt 0
v_mov_b32 v2, s39
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:124
s_wait_storecnt 0
v_mov_b32 v2, s40
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:128
s_wait_storecnt 0
s_wait_storecnt 0
s_endpgm
.Lend_sentinel_00_0:
.size sentinel_00_0, .Lend_sentinel_00_0-sentinel_00_0
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel sentinel_00_0
.amdhsa_group_segment_fixed_size 0
.amdhsa_private_segment_fixed_size 0
.amdhsa_kernarg_size 8
.amdhsa_user_sgpr_count 2
.amdhsa_user_sgpr_dispatch_ptr 0
.amdhsa_user_sgpr_queue_ptr 0
.amdhsa_user_sgpr_kernarg_segment_ptr 1
.amdhsa_user_sgpr_dispatch_id 0
.amdhsa_user_sgpr_kernarg_preload_length 0
.amdhsa_user_sgpr_kernarg_preload_offset 0
.amdhsa_user_sgpr_private_segment_size 0
.amdhsa_wavefront_size32 1
.amdhsa_uses_dynamic_stack 0
.amdhsa_enable_private_segment 0
.amdhsa_system_sgpr_workgroup_id_x 0
.amdhsa_system_sgpr_workgroup_id_y 0
.amdhsa_system_sgpr_workgroup_id_z 0
.amdhsa_system_sgpr_workgroup_info 0
.amdhsa_system_vgpr_workitem_id 0
.amdhsa_next_free_vgpr 516
.amdhsa_next_free_sgpr 42
.amdhsa_named_barrier_count 0
.amdhsa_reserve_vcc 0
.amdhsa_float_round_mode_32 0
.amdhsa_float_round_mode_16_64 0
.amdhsa_float_denorm_mode_32 3
.amdhsa_float_denorm_mode_16_64 3
.amdhsa_fp16_overflow 0
.amdhsa_memory_ordered 1
.amdhsa_forward_progress 1
.amdhsa_round_robin_scheduling 0
.amdhsa_exception_fp_ieee_invalid_op 0
.amdhsa_exception_fp_denorm_src 0
.amdhsa_exception_fp_ieee_div_zero 0
.amdhsa_exception_fp_ieee_overflow 0
.amdhsa_exception_fp_ieee_underflow 0
.amdhsa_exception_fp_ieee_inexact 0
.amdhsa_exception_int_div_zero 0
.end_amdhsa_kernel
.text
.protected sentinel_00_1
.globl sentinel_00_1
.p2align 8
.type sentinel_00_1,@function
sentinel_00_1:
global_prefetch_b8 v0, s[0:1] scope:SCOPE_SE
v_nop
s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
s_load_b64 s[4:5], s[0:1], 0x0 nv
s_wait_kmcnt 0
s_mov_b32 s6, 0
s_setreg_b32 hwreg(HW_REG_WAVE_MODE, 12, 8), s6
s_nop 0
s_set_vgpr_msb 0x0000
s_nop 0
v_mov_b32 v1, 0x11000000
s_wait_idle
s_mov_b32 s7, 0x11000001
v_writelane_b32 v1, s7, 1
s_wait_idle
s_mov_b32 s7, 0x11000010
v_writelane_b32 v1, s7, 16
s_wait_idle
s_set_vgpr_msb 0x0041
s_nop 0
v_mov_b32 v1, 0x22000000
s_wait_idle
s_mov_b32 s7, 0x22000001
v_writelane_b32 v1, s7, 1
s_wait_idle
s_mov_b32 s7, 0x22000010
v_writelane_b32 v1, s7, 16
s_wait_idle
s_set_vgpr_msb 0x4182
s_nop 0
v_mov_b32 v1, 0x33000000
s_wait_idle
s_mov_b32 s7, 0x33000001
v_writelane_b32 v1, s7, 1
s_wait_idle
s_mov_b32 s7, 0x33000010
v_writelane_b32 v1, s7, 16
s_wait_idle
v_readlane_b32 s22, v1, 0
v_readlane_b32 s23, v1, 1
v_readlane_b32 s24, v1, 16
s_set_vgpr_msb 0x8241
s_nop 0
v_readlane_b32 s19, v1, 0
v_readlane_b32 s20, v1, 1
v_readlane_b32 s21, v1, 16
s_set_vgpr_msb 0x4100
s_nop 0
v_readlane_b32 s16, v1, 0
v_readlane_b32 s17, v1, 1
v_readlane_b32 s18, v1, 16
s_mov_b32 s8, 0x53545250
s_mov_b32 s9, 1
s_mov_b32 s10, 0x00
s_mov_b32 s11, 1
s_mov_b32 s40, 0x434f4d50
s_set_vgpr_msb 0x0000
s_nop 0
s_wait_idle
s_getreg_b32 s14, hwreg(HW_REG_WAVE_STATUS)
s_mov_b32 s38, exec_lo
s_getreg_b32 s12, hwreg(HW_REG_WAVE_MODE)
s_get_pc_i64 s[34:35]
.Lpcnext_sentinel_00_1:
s_add_co_u32 s34, s34, (.Lreturn_sentinel_00_1 - .Lpcnext_sentinel_00_1)
s_add_co_ci_u32 s35, s35, 0
s_nop 0
s_trap 3
.Lreturn_sentinel_00_1:
s_getreg_b32 s13, hwreg(HW_REG_WAVE_MODE)
s_mov_b32 s36, ttmp0
s_mov_b32 s37, ttmp1
s_mov_b32 s39, exec_lo
s_wait_idle
s_getreg_b32 s15, hwreg(HW_REG_WAVE_STATUS)
s_set_vgpr_msb 0x0000
s_nop 0
v_readlane_b32 s25, v1, 0
v_readlane_b32 s26, v1, 1
v_readlane_b32 s27, v1, 16
s_set_vgpr_msb 0x0041
s_nop 0
v_readlane_b32 s28, v1, 0
v_readlane_b32 s29, v1, 1
v_readlane_b32 s30, v1, 16
s_set_vgpr_msb 0x4182
s_nop 0
v_readlane_b32 s31, v1, 0
v_readlane_b32 s32, v1, 1
v_readlane_b32 s33, v1, 16
s_set_vgpr_msb 0x8200
s_nop 0
s_mov_b32 exec_lo, 1
v_mov_b32 v0, 0
v_mov_b32 v2, s8
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:0
s_wait_storecnt 0
v_mov_b32 v2, s9
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:4
s_wait_storecnt 0
v_mov_b32 v2, s10
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:8
s_wait_storecnt 0
v_mov_b32 v2, s11
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:12
s_wait_storecnt 0
v_mov_b32 v2, s12
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:16
s_wait_storecnt 0
v_mov_b32 v2, s13
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:20
s_wait_storecnt 0
v_mov_b32 v2, s14
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:24
s_wait_storecnt 0
v_mov_b32 v2, s15
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:28
s_wait_storecnt 0
v_mov_b32 v2, s16
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:32
s_wait_storecnt 0
v_mov_b32 v2, s17
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:36
s_wait_storecnt 0
v_mov_b32 v2, s18
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:40
s_wait_storecnt 0
v_mov_b32 v2, s19
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:44
s_wait_storecnt 0
v_mov_b32 v2, s20
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:48
s_wait_storecnt 0
v_mov_b32 v2, s21
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:52
s_wait_storecnt 0
v_mov_b32 v2, s22
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:56
s_wait_storecnt 0
v_mov_b32 v2, s23
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:60
s_wait_storecnt 0
v_mov_b32 v2, s24
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:64
s_wait_storecnt 0
v_mov_b32 v2, s25
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:68
s_wait_storecnt 0
v_mov_b32 v2, s26
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:72
s_wait_storecnt 0
v_mov_b32 v2, s27
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:76
s_wait_storecnt 0
v_mov_b32 v2, s28
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:80
s_wait_storecnt 0
v_mov_b32 v2, s29
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:84
s_wait_storecnt 0
v_mov_b32 v2, s30
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:88
s_wait_storecnt 0
v_mov_b32 v2, s31
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:92
s_wait_storecnt 0
v_mov_b32 v2, s32
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:96
s_wait_storecnt 0
v_mov_b32 v2, s33
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:100
s_wait_storecnt 0
v_mov_b32 v2, s34
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:104
s_wait_storecnt 0
v_mov_b32 v2, s35
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:108
s_wait_storecnt 0
v_mov_b32 v2, s36
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:112
s_wait_storecnt 0
v_mov_b32 v2, s37
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:116
s_wait_storecnt 0
v_mov_b32 v2, s38
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:120
s_wait_storecnt 0
v_mov_b32 v2, s39
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:124
s_wait_storecnt 0
v_mov_b32 v2, s40
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:128
s_wait_storecnt 0
s_wait_storecnt 0
s_endpgm
.Lend_sentinel_00_1:
.size sentinel_00_1, .Lend_sentinel_00_1-sentinel_00_1
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel sentinel_00_1
.amdhsa_group_segment_fixed_size 0
.amdhsa_private_segment_fixed_size 0
.amdhsa_kernarg_size 8
.amdhsa_user_sgpr_count 2
.amdhsa_user_sgpr_dispatch_ptr 0
.amdhsa_user_sgpr_queue_ptr 0
.amdhsa_user_sgpr_kernarg_segment_ptr 1
.amdhsa_user_sgpr_dispatch_id 0
.amdhsa_user_sgpr_kernarg_preload_length 0
.amdhsa_user_sgpr_kernarg_preload_offset 0
.amdhsa_user_sgpr_private_segment_size 0
.amdhsa_wavefront_size32 1
.amdhsa_uses_dynamic_stack 0
.amdhsa_enable_private_segment 0
.amdhsa_system_sgpr_workgroup_id_x 0
.amdhsa_system_sgpr_workgroup_id_y 0
.amdhsa_system_sgpr_workgroup_id_z 0
.amdhsa_system_sgpr_workgroup_info 0
.amdhsa_system_vgpr_workitem_id 0
.amdhsa_next_free_vgpr 516
.amdhsa_next_free_sgpr 42
.amdhsa_named_barrier_count 0
.amdhsa_reserve_vcc 0
.amdhsa_float_round_mode_32 0
.amdhsa_float_round_mode_16_64 0
.amdhsa_float_denorm_mode_32 3
.amdhsa_float_denorm_mode_16_64 3
.amdhsa_fp16_overflow 0
.amdhsa_memory_ordered 1
.amdhsa_forward_progress 1
.amdhsa_round_robin_scheduling 0
.amdhsa_exception_fp_ieee_invalid_op 0
.amdhsa_exception_fp_denorm_src 0
.amdhsa_exception_fp_ieee_div_zero 0
.amdhsa_exception_fp_ieee_overflow 0
.amdhsa_exception_fp_ieee_underflow 0
.amdhsa_exception_fp_ieee_inexact 0
.amdhsa_exception_int_div_zero 0
.end_amdhsa_kernel
.text
.protected sentinel_41_0
.globl sentinel_41_0
.p2align 8
.type sentinel_41_0,@function
sentinel_41_0:
global_prefetch_b8 v0, s[0:1] scope:SCOPE_SE
v_nop
s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
s_load_b64 s[4:5], s[0:1], 0x0 nv
s_wait_kmcnt 0
s_mov_b32 s6, 0
s_setreg_b32 hwreg(HW_REG_WAVE_MODE, 12, 8), s6
s_nop 0
s_set_vgpr_msb 0x0000
s_nop 0
v_mov_b32 v1, 0x11000000
s_wait_idle
s_mov_b32 s7, 0x11000001
v_writelane_b32 v1, s7, 1
s_wait_idle
s_mov_b32 s7, 0x11000010
v_writelane_b32 v1, s7, 16
s_wait_idle
s_set_vgpr_msb 0x0041
s_nop 0
v_mov_b32 v1, 0x22000000
s_wait_idle
s_mov_b32 s7, 0x22000001
v_writelane_b32 v1, s7, 1
s_wait_idle
s_mov_b32 s7, 0x22000010
v_writelane_b32 v1, s7, 16
s_wait_idle
s_set_vgpr_msb 0x4182
s_nop 0
v_mov_b32 v1, 0x33000000
s_wait_idle
s_mov_b32 s7, 0x33000001
v_writelane_b32 v1, s7, 1
s_wait_idle
s_mov_b32 s7, 0x33000010
v_writelane_b32 v1, s7, 16
s_wait_idle
v_readlane_b32 s22, v1, 0
v_readlane_b32 s23, v1, 1
v_readlane_b32 s24, v1, 16
s_set_vgpr_msb 0x8241
s_nop 0
v_readlane_b32 s19, v1, 0
v_readlane_b32 s20, v1, 1
v_readlane_b32 s21, v1, 16
s_set_vgpr_msb 0x4100
s_nop 0
v_readlane_b32 s16, v1, 0
v_readlane_b32 s17, v1, 1
v_readlane_b32 s18, v1, 16
s_mov_b32 s8, 0x53545250
s_mov_b32 s9, 1
s_mov_b32 s10, 0x41
s_mov_b32 s11, 0
s_mov_b32 s40, 0x434f4d50
s_set_vgpr_msb 0x0041
s_nop 0
s_wait_idle
s_getreg_b32 s14, hwreg(HW_REG_WAVE_STATUS)
s_mov_b32 s38, exec_lo
s_getreg_b32 s12, hwreg(HW_REG_WAVE_MODE)
s_get_pc_i64 s[34:35]
.Lpcnext_sentinel_41_0:
s_add_co_u32 s34, s34, (.Lreturn_sentinel_41_0 - .Lpcnext_sentinel_41_0)
s_add_co_ci_u32 s35, s35, 0
s_nop 0
s_nop 0
.Lreturn_sentinel_41_0:
s_getreg_b32 s13, hwreg(HW_REG_WAVE_MODE)
s_mov_b32 s36, ttmp0
s_mov_b32 s37, ttmp1
s_mov_b32 s39, exec_lo
s_wait_idle
s_getreg_b32 s15, hwreg(HW_REG_WAVE_STATUS)
s_set_vgpr_msb 0x4100
s_nop 0
v_readlane_b32 s25, v1, 0
v_readlane_b32 s26, v1, 1
v_readlane_b32 s27, v1, 16
s_set_vgpr_msb 0x0041
s_nop 0
v_readlane_b32 s28, v1, 0
v_readlane_b32 s29, v1, 1
v_readlane_b32 s30, v1, 16
s_set_vgpr_msb 0x4182
s_nop 0
v_readlane_b32 s31, v1, 0
v_readlane_b32 s32, v1, 1
v_readlane_b32 s33, v1, 16
s_set_vgpr_msb 0x8200
s_nop 0
s_mov_b32 exec_lo, 1
v_mov_b32 v0, 0
v_mov_b32 v2, s8
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:0
s_wait_storecnt 0
v_mov_b32 v2, s9
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:4
s_wait_storecnt 0
v_mov_b32 v2, s10
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:8
s_wait_storecnt 0
v_mov_b32 v2, s11
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:12
s_wait_storecnt 0
v_mov_b32 v2, s12
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:16
s_wait_storecnt 0
v_mov_b32 v2, s13
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:20
s_wait_storecnt 0
v_mov_b32 v2, s14
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:24
s_wait_storecnt 0
v_mov_b32 v2, s15
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:28
s_wait_storecnt 0
v_mov_b32 v2, s16
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:32
s_wait_storecnt 0
v_mov_b32 v2, s17
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:36
s_wait_storecnt 0
v_mov_b32 v2, s18
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:40
s_wait_storecnt 0
v_mov_b32 v2, s19
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:44
s_wait_storecnt 0
v_mov_b32 v2, s20
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:48
s_wait_storecnt 0
v_mov_b32 v2, s21
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:52
s_wait_storecnt 0
v_mov_b32 v2, s22
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:56
s_wait_storecnt 0
v_mov_b32 v2, s23
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:60
s_wait_storecnt 0
v_mov_b32 v2, s24
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:64
s_wait_storecnt 0
v_mov_b32 v2, s25
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:68
s_wait_storecnt 0
v_mov_b32 v2, s26
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:72
s_wait_storecnt 0
v_mov_b32 v2, s27
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:76
s_wait_storecnt 0
v_mov_b32 v2, s28
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:80
s_wait_storecnt 0
v_mov_b32 v2, s29
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:84
s_wait_storecnt 0
v_mov_b32 v2, s30
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:88
s_wait_storecnt 0
v_mov_b32 v2, s31
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:92
s_wait_storecnt 0
v_mov_b32 v2, s32
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:96
s_wait_storecnt 0
v_mov_b32 v2, s33
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:100
s_wait_storecnt 0
v_mov_b32 v2, s34
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:104
s_wait_storecnt 0
v_mov_b32 v2, s35
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:108
s_wait_storecnt 0
v_mov_b32 v2, s36
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:112
s_wait_storecnt 0
v_mov_b32 v2, s37
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:116
s_wait_storecnt 0
v_mov_b32 v2, s38
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:120
s_wait_storecnt 0
v_mov_b32 v2, s39
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:124
s_wait_storecnt 0
v_mov_b32 v2, s40
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:128
s_wait_storecnt 0
s_wait_storecnt 0
s_endpgm
.Lend_sentinel_41_0:
.size sentinel_41_0, .Lend_sentinel_41_0-sentinel_41_0
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel sentinel_41_0
.amdhsa_group_segment_fixed_size 0
.amdhsa_private_segment_fixed_size 0
.amdhsa_kernarg_size 8
.amdhsa_user_sgpr_count 2
.amdhsa_user_sgpr_dispatch_ptr 0
.amdhsa_user_sgpr_queue_ptr 0
.amdhsa_user_sgpr_kernarg_segment_ptr 1
.amdhsa_user_sgpr_dispatch_id 0
.amdhsa_user_sgpr_kernarg_preload_length 0
.amdhsa_user_sgpr_kernarg_preload_offset 0
.amdhsa_user_sgpr_private_segment_size 0
.amdhsa_wavefront_size32 1
.amdhsa_uses_dynamic_stack 0
.amdhsa_enable_private_segment 0
.amdhsa_system_sgpr_workgroup_id_x 0
.amdhsa_system_sgpr_workgroup_id_y 0
.amdhsa_system_sgpr_workgroup_id_z 0
.amdhsa_system_sgpr_workgroup_info 0
.amdhsa_system_vgpr_workitem_id 0
.amdhsa_next_free_vgpr 516
.amdhsa_next_free_sgpr 42
.amdhsa_named_barrier_count 0
.amdhsa_reserve_vcc 0
.amdhsa_float_round_mode_32 0
.amdhsa_float_round_mode_16_64 0
.amdhsa_float_denorm_mode_32 3
.amdhsa_float_denorm_mode_16_64 3
.amdhsa_fp16_overflow 0
.amdhsa_memory_ordered 1
.amdhsa_forward_progress 1
.amdhsa_round_robin_scheduling 0
.amdhsa_exception_fp_ieee_invalid_op 0
.amdhsa_exception_fp_denorm_src 0
.amdhsa_exception_fp_ieee_div_zero 0
.amdhsa_exception_fp_ieee_overflow 0
.amdhsa_exception_fp_ieee_underflow 0
.amdhsa_exception_fp_ieee_inexact 0
.amdhsa_exception_int_div_zero 0
.end_amdhsa_kernel
.text
.protected sentinel_41_1
.globl sentinel_41_1
.p2align 8
.type sentinel_41_1,@function
sentinel_41_1:
global_prefetch_b8 v0, s[0:1] scope:SCOPE_SE
v_nop
s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
s_load_b64 s[4:5], s[0:1], 0x0 nv
s_wait_kmcnt 0
s_mov_b32 s6, 0
s_setreg_b32 hwreg(HW_REG_WAVE_MODE, 12, 8), s6
s_nop 0
s_set_vgpr_msb 0x0000
s_nop 0
v_mov_b32 v1, 0x11000000
s_wait_idle
s_mov_b32 s7, 0x11000001
v_writelane_b32 v1, s7, 1
s_wait_idle
s_mov_b32 s7, 0x11000010
v_writelane_b32 v1, s7, 16
s_wait_idle
s_set_vgpr_msb 0x0041
s_nop 0
v_mov_b32 v1, 0x22000000
s_wait_idle
s_mov_b32 s7, 0x22000001
v_writelane_b32 v1, s7, 1
s_wait_idle
s_mov_b32 s7, 0x22000010
v_writelane_b32 v1, s7, 16
s_wait_idle
s_set_vgpr_msb 0x4182
s_nop 0
v_mov_b32 v1, 0x33000000
s_wait_idle
s_mov_b32 s7, 0x33000001
v_writelane_b32 v1, s7, 1
s_wait_idle
s_mov_b32 s7, 0x33000010
v_writelane_b32 v1, s7, 16
s_wait_idle
v_readlane_b32 s22, v1, 0
v_readlane_b32 s23, v1, 1
v_readlane_b32 s24, v1, 16
s_set_vgpr_msb 0x8241
s_nop 0
v_readlane_b32 s19, v1, 0
v_readlane_b32 s20, v1, 1
v_readlane_b32 s21, v1, 16
s_set_vgpr_msb 0x4100
s_nop 0
v_readlane_b32 s16, v1, 0
v_readlane_b32 s17, v1, 1
v_readlane_b32 s18, v1, 16
s_mov_b32 s8, 0x53545250
s_mov_b32 s9, 1
s_mov_b32 s10, 0x41
s_mov_b32 s11, 1
s_mov_b32 s40, 0x434f4d50
s_set_vgpr_msb 0x0041
s_nop 0
s_wait_idle
s_getreg_b32 s14, hwreg(HW_REG_WAVE_STATUS)
s_mov_b32 s38, exec_lo
s_getreg_b32 s12, hwreg(HW_REG_WAVE_MODE)
s_get_pc_i64 s[34:35]
.Lpcnext_sentinel_41_1:
s_add_co_u32 s34, s34, (.Lreturn_sentinel_41_1 - .Lpcnext_sentinel_41_1)
s_add_co_ci_u32 s35, s35, 0
s_nop 0
s_trap 3
.Lreturn_sentinel_41_1:
s_getreg_b32 s13, hwreg(HW_REG_WAVE_MODE)
s_mov_b32 s36, ttmp0
s_mov_b32 s37, ttmp1
s_mov_b32 s39, exec_lo
s_wait_idle
s_getreg_b32 s15, hwreg(HW_REG_WAVE_STATUS)
s_set_vgpr_msb 0x4100
s_nop 0
v_readlane_b32 s25, v1, 0
v_readlane_b32 s26, v1, 1
v_readlane_b32 s27, v1, 16
s_set_vgpr_msb 0x0041
s_nop 0
v_readlane_b32 s28, v1, 0
v_readlane_b32 s29, v1, 1
v_readlane_b32 s30, v1, 16
s_set_vgpr_msb 0x4182
s_nop 0
v_readlane_b32 s31, v1, 0
v_readlane_b32 s32, v1, 1
v_readlane_b32 s33, v1, 16
s_set_vgpr_msb 0x8200
s_nop 0
s_mov_b32 exec_lo, 1
v_mov_b32 v0, 0
v_mov_b32 v2, s8
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:0
s_wait_storecnt 0
v_mov_b32 v2, s9
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:4
s_wait_storecnt 0
v_mov_b32 v2, s10
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:8
s_wait_storecnt 0
v_mov_b32 v2, s11
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:12
s_wait_storecnt 0
v_mov_b32 v2, s12
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:16
s_wait_storecnt 0
v_mov_b32 v2, s13
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:20
s_wait_storecnt 0
v_mov_b32 v2, s14
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:24
s_wait_storecnt 0
v_mov_b32 v2, s15
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:28
s_wait_storecnt 0
v_mov_b32 v2, s16
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:32
s_wait_storecnt 0
v_mov_b32 v2, s17
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:36
s_wait_storecnt 0
v_mov_b32 v2, s18
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:40
s_wait_storecnt 0
v_mov_b32 v2, s19
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:44
s_wait_storecnt 0
v_mov_b32 v2, s20
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:48
s_wait_storecnt 0
v_mov_b32 v2, s21
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:52
s_wait_storecnt 0
v_mov_b32 v2, s22
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:56
s_wait_storecnt 0
v_mov_b32 v2, s23
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:60
s_wait_storecnt 0
v_mov_b32 v2, s24
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:64
s_wait_storecnt 0
v_mov_b32 v2, s25
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:68
s_wait_storecnt 0
v_mov_b32 v2, s26
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:72
s_wait_storecnt 0
v_mov_b32 v2, s27
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:76
s_wait_storecnt 0
v_mov_b32 v2, s28
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:80
s_wait_storecnt 0
v_mov_b32 v2, s29
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:84
s_wait_storecnt 0
v_mov_b32 v2, s30
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:88
s_wait_storecnt 0
v_mov_b32 v2, s31
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:92
s_wait_storecnt 0
v_mov_b32 v2, s32
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:96
s_wait_storecnt 0
v_mov_b32 v2, s33
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:100
s_wait_storecnt 0
v_mov_b32 v2, s34
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:104
s_wait_storecnt 0
v_mov_b32 v2, s35
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:108
s_wait_storecnt 0
v_mov_b32 v2, s36
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:112
s_wait_storecnt 0
v_mov_b32 v2, s37
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:116
s_wait_storecnt 0
v_mov_b32 v2, s38
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:120
s_wait_storecnt 0
v_mov_b32 v2, s39
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:124
s_wait_storecnt 0
v_mov_b32 v2, s40
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:128
s_wait_storecnt 0
s_wait_storecnt 0
s_endpgm
.Lend_sentinel_41_1:
.size sentinel_41_1, .Lend_sentinel_41_1-sentinel_41_1
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel sentinel_41_1
.amdhsa_group_segment_fixed_size 0
.amdhsa_private_segment_fixed_size 0
.amdhsa_kernarg_size 8
.amdhsa_user_sgpr_count 2
.amdhsa_user_sgpr_dispatch_ptr 0
.amdhsa_user_sgpr_queue_ptr 0
.amdhsa_user_sgpr_kernarg_segment_ptr 1
.amdhsa_user_sgpr_dispatch_id 0
.amdhsa_user_sgpr_kernarg_preload_length 0
.amdhsa_user_sgpr_kernarg_preload_offset 0
.amdhsa_user_sgpr_private_segment_size 0
.amdhsa_wavefront_size32 1
.amdhsa_uses_dynamic_stack 0
.amdhsa_enable_private_segment 0
.amdhsa_system_sgpr_workgroup_id_x 0
.amdhsa_system_sgpr_workgroup_id_y 0
.amdhsa_system_sgpr_workgroup_id_z 0
.amdhsa_system_sgpr_workgroup_info 0
.amdhsa_system_vgpr_workitem_id 0
.amdhsa_next_free_vgpr 516
.amdhsa_next_free_sgpr 42
.amdhsa_named_barrier_count 0
.amdhsa_reserve_vcc 0
.amdhsa_float_round_mode_32 0
.amdhsa_float_round_mode_16_64 0
.amdhsa_float_denorm_mode_32 3
.amdhsa_float_denorm_mode_16_64 3
.amdhsa_fp16_overflow 0
.amdhsa_memory_ordered 1
.amdhsa_forward_progress 1
.amdhsa_round_robin_scheduling 0
.amdhsa_exception_fp_ieee_invalid_op 0
.amdhsa_exception_fp_denorm_src 0
.amdhsa_exception_fp_ieee_div_zero 0
.amdhsa_exception_fp_ieee_overflow 0
.amdhsa_exception_fp_ieee_underflow 0
.amdhsa_exception_fp_ieee_inexact 0
.amdhsa_exception_int_div_zero 0
.end_amdhsa_kernel
.text
.protected sentinel_45_0
.globl sentinel_45_0
.p2align 8
.type sentinel_45_0,@function
sentinel_45_0:
global_prefetch_b8 v0, s[0:1] scope:SCOPE_SE
v_nop
s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
s_load_b64 s[4:5], s[0:1], 0x0 nv
s_wait_kmcnt 0
s_mov_b32 s6, 0
s_setreg_b32 hwreg(HW_REG_WAVE_MODE, 12, 8), s6
s_nop 0
s_set_vgpr_msb 0x0000
s_nop 0
v_mov_b32 v1, 0x11000000
s_wait_idle
s_mov_b32 s7, 0x11000001
v_writelane_b32 v1, s7, 1
s_wait_idle
s_mov_b32 s7, 0x11000010
v_writelane_b32 v1, s7, 16
s_wait_idle
s_set_vgpr_msb 0x0041
s_nop 0
v_mov_b32 v1, 0x22000000
s_wait_idle
s_mov_b32 s7, 0x22000001
v_writelane_b32 v1, s7, 1
s_wait_idle
s_mov_b32 s7, 0x22000010
v_writelane_b32 v1, s7, 16
s_wait_idle
s_set_vgpr_msb 0x4182
s_nop 0
v_mov_b32 v1, 0x33000000
s_wait_idle
s_mov_b32 s7, 0x33000001
v_writelane_b32 v1, s7, 1
s_wait_idle
s_mov_b32 s7, 0x33000010
v_writelane_b32 v1, s7, 16
s_wait_idle
v_readlane_b32 s22, v1, 0
v_readlane_b32 s23, v1, 1
v_readlane_b32 s24, v1, 16
s_set_vgpr_msb 0x8241
s_nop 0
v_readlane_b32 s19, v1, 0
v_readlane_b32 s20, v1, 1
v_readlane_b32 s21, v1, 16
s_set_vgpr_msb 0x4100
s_nop 0
v_readlane_b32 s16, v1, 0
v_readlane_b32 s17, v1, 1
v_readlane_b32 s18, v1, 16
s_mov_b32 s8, 0x53545250
s_mov_b32 s9, 1
s_mov_b32 s10, 0x45
s_mov_b32 s11, 0
s_mov_b32 s40, 0x434f4d50
s_set_vgpr_msb 0x0045
s_nop 0
s_wait_idle
s_getreg_b32 s14, hwreg(HW_REG_WAVE_STATUS)
s_mov_b32 s38, exec_lo
s_getreg_b32 s12, hwreg(HW_REG_WAVE_MODE)
s_get_pc_i64 s[34:35]
.Lpcnext_sentinel_45_0:
s_add_co_u32 s34, s34, (.Lreturn_sentinel_45_0 - .Lpcnext_sentinel_45_0)
s_add_co_ci_u32 s35, s35, 0
s_nop 0
s_nop 0
.Lreturn_sentinel_45_0:
s_getreg_b32 s13, hwreg(HW_REG_WAVE_MODE)
s_mov_b32 s36, ttmp0
s_mov_b32 s37, ttmp1
s_mov_b32 s39, exec_lo
s_wait_idle
s_getreg_b32 s15, hwreg(HW_REG_WAVE_STATUS)
s_set_vgpr_msb 0x4500
s_nop 0
v_readlane_b32 s25, v1, 0
v_readlane_b32 s26, v1, 1
v_readlane_b32 s27, v1, 16
s_set_vgpr_msb 0x0041
s_nop 0
v_readlane_b32 s28, v1, 0
v_readlane_b32 s29, v1, 1
v_readlane_b32 s30, v1, 16
s_set_vgpr_msb 0x4182
s_nop 0
v_readlane_b32 s31, v1, 0
v_readlane_b32 s32, v1, 1
v_readlane_b32 s33, v1, 16
s_set_vgpr_msb 0x8200
s_nop 0
s_mov_b32 exec_lo, 1
v_mov_b32 v0, 0
v_mov_b32 v2, s8
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:0
s_wait_storecnt 0
v_mov_b32 v2, s9
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:4
s_wait_storecnt 0
v_mov_b32 v2, s10
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:8
s_wait_storecnt 0
v_mov_b32 v2, s11
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:12
s_wait_storecnt 0
v_mov_b32 v2, s12
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:16
s_wait_storecnt 0
v_mov_b32 v2, s13
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:20
s_wait_storecnt 0
v_mov_b32 v2, s14
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:24
s_wait_storecnt 0
v_mov_b32 v2, s15
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:28
s_wait_storecnt 0
v_mov_b32 v2, s16
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:32
s_wait_storecnt 0
v_mov_b32 v2, s17
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:36
s_wait_storecnt 0
v_mov_b32 v2, s18
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:40
s_wait_storecnt 0
v_mov_b32 v2, s19
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:44
s_wait_storecnt 0
v_mov_b32 v2, s20
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:48
s_wait_storecnt 0
v_mov_b32 v2, s21
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:52
s_wait_storecnt 0
v_mov_b32 v2, s22
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:56
s_wait_storecnt 0
v_mov_b32 v2, s23
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:60
s_wait_storecnt 0
v_mov_b32 v2, s24
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:64
s_wait_storecnt 0
v_mov_b32 v2, s25
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:68
s_wait_storecnt 0
v_mov_b32 v2, s26
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:72
s_wait_storecnt 0
v_mov_b32 v2, s27
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:76
s_wait_storecnt 0
v_mov_b32 v2, s28
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:80
s_wait_storecnt 0
v_mov_b32 v2, s29
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:84
s_wait_storecnt 0
v_mov_b32 v2, s30
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:88
s_wait_storecnt 0
v_mov_b32 v2, s31
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:92
s_wait_storecnt 0
v_mov_b32 v2, s32
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:96
s_wait_storecnt 0
v_mov_b32 v2, s33
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:100
s_wait_storecnt 0
v_mov_b32 v2, s34
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:104
s_wait_storecnt 0
v_mov_b32 v2, s35
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:108
s_wait_storecnt 0
v_mov_b32 v2, s36
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:112
s_wait_storecnt 0
v_mov_b32 v2, s37
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:116
s_wait_storecnt 0
v_mov_b32 v2, s38
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:120
s_wait_storecnt 0
v_mov_b32 v2, s39
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:124
s_wait_storecnt 0
v_mov_b32 v2, s40
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:128
s_wait_storecnt 0
s_wait_storecnt 0
s_endpgm
.Lend_sentinel_45_0:
.size sentinel_45_0, .Lend_sentinel_45_0-sentinel_45_0
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel sentinel_45_0
.amdhsa_group_segment_fixed_size 0
.amdhsa_private_segment_fixed_size 0
.amdhsa_kernarg_size 8
.amdhsa_user_sgpr_count 2
.amdhsa_user_sgpr_dispatch_ptr 0
.amdhsa_user_sgpr_queue_ptr 0
.amdhsa_user_sgpr_kernarg_segment_ptr 1
.amdhsa_user_sgpr_dispatch_id 0
.amdhsa_user_sgpr_kernarg_preload_length 0
.amdhsa_user_sgpr_kernarg_preload_offset 0
.amdhsa_user_sgpr_private_segment_size 0
.amdhsa_wavefront_size32 1
.amdhsa_uses_dynamic_stack 0
.amdhsa_enable_private_segment 0
.amdhsa_system_sgpr_workgroup_id_x 0
.amdhsa_system_sgpr_workgroup_id_y 0
.amdhsa_system_sgpr_workgroup_id_z 0
.amdhsa_system_sgpr_workgroup_info 0
.amdhsa_system_vgpr_workitem_id 0
.amdhsa_next_free_vgpr 516
.amdhsa_next_free_sgpr 42
.amdhsa_named_barrier_count 0
.amdhsa_reserve_vcc 0
.amdhsa_float_round_mode_32 0
.amdhsa_float_round_mode_16_64 0
.amdhsa_float_denorm_mode_32 3
.amdhsa_float_denorm_mode_16_64 3
.amdhsa_fp16_overflow 0
.amdhsa_memory_ordered 1
.amdhsa_forward_progress 1
.amdhsa_round_robin_scheduling 0
.amdhsa_exception_fp_ieee_invalid_op 0
.amdhsa_exception_fp_denorm_src 0
.amdhsa_exception_fp_ieee_div_zero 0
.amdhsa_exception_fp_ieee_overflow 0
.amdhsa_exception_fp_ieee_underflow 0
.amdhsa_exception_fp_ieee_inexact 0
.amdhsa_exception_int_div_zero 0
.end_amdhsa_kernel
.text
.protected sentinel_45_1
.globl sentinel_45_1
.p2align 8
.type sentinel_45_1,@function
sentinel_45_1:
global_prefetch_b8 v0, s[0:1] scope:SCOPE_SE
v_nop
s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
s_load_b64 s[4:5], s[0:1], 0x0 nv
s_wait_kmcnt 0
s_mov_b32 s6, 0
s_setreg_b32 hwreg(HW_REG_WAVE_MODE, 12, 8), s6
s_nop 0
s_set_vgpr_msb 0x0000
s_nop 0
v_mov_b32 v1, 0x11000000
s_wait_idle
s_mov_b32 s7, 0x11000001
v_writelane_b32 v1, s7, 1
s_wait_idle
s_mov_b32 s7, 0x11000010
v_writelane_b32 v1, s7, 16
s_wait_idle
s_set_vgpr_msb 0x0041
s_nop 0
v_mov_b32 v1, 0x22000000
s_wait_idle
s_mov_b32 s7, 0x22000001
v_writelane_b32 v1, s7, 1
s_wait_idle
s_mov_b32 s7, 0x22000010
v_writelane_b32 v1, s7, 16
s_wait_idle
s_set_vgpr_msb 0x4182
s_nop 0
v_mov_b32 v1, 0x33000000
s_wait_idle
s_mov_b32 s7, 0x33000001
v_writelane_b32 v1, s7, 1
s_wait_idle
s_mov_b32 s7, 0x33000010
v_writelane_b32 v1, s7, 16
s_wait_idle
v_readlane_b32 s22, v1, 0
v_readlane_b32 s23, v1, 1
v_readlane_b32 s24, v1, 16
s_set_vgpr_msb 0x8241
s_nop 0
v_readlane_b32 s19, v1, 0
v_readlane_b32 s20, v1, 1
v_readlane_b32 s21, v1, 16
s_set_vgpr_msb 0x4100
s_nop 0
v_readlane_b32 s16, v1, 0
v_readlane_b32 s17, v1, 1
v_readlane_b32 s18, v1, 16
s_mov_b32 s8, 0x53545250
s_mov_b32 s9, 1
s_mov_b32 s10, 0x45
s_mov_b32 s11, 1
s_mov_b32 s40, 0x434f4d50
s_set_vgpr_msb 0x0045
s_nop 0
s_wait_idle
s_getreg_b32 s14, hwreg(HW_REG_WAVE_STATUS)
s_mov_b32 s38, exec_lo
s_getreg_b32 s12, hwreg(HW_REG_WAVE_MODE)
s_get_pc_i64 s[34:35]
.Lpcnext_sentinel_45_1:
s_add_co_u32 s34, s34, (.Lreturn_sentinel_45_1 - .Lpcnext_sentinel_45_1)
s_add_co_ci_u32 s35, s35, 0
s_nop 0
s_trap 3
.Lreturn_sentinel_45_1:
s_getreg_b32 s13, hwreg(HW_REG_WAVE_MODE)
s_mov_b32 s36, ttmp0
s_mov_b32 s37, ttmp1
s_mov_b32 s39, exec_lo
s_wait_idle
s_getreg_b32 s15, hwreg(HW_REG_WAVE_STATUS)
s_set_vgpr_msb 0x4500
s_nop 0
v_readlane_b32 s25, v1, 0
v_readlane_b32 s26, v1, 1
v_readlane_b32 s27, v1, 16
s_set_vgpr_msb 0x0041
s_nop 0
v_readlane_b32 s28, v1, 0
v_readlane_b32 s29, v1, 1
v_readlane_b32 s30, v1, 16
s_set_vgpr_msb 0x4182
s_nop 0
v_readlane_b32 s31, v1, 0
v_readlane_b32 s32, v1, 1
v_readlane_b32 s33, v1, 16
s_set_vgpr_msb 0x8200
s_nop 0
s_mov_b32 exec_lo, 1
v_mov_b32 v0, 0
v_mov_b32 v2, s8
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:0
s_wait_storecnt 0
v_mov_b32 v2, s9
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:4
s_wait_storecnt 0
v_mov_b32 v2, s10
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:8
s_wait_storecnt 0
v_mov_b32 v2, s11
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:12
s_wait_storecnt 0
v_mov_b32 v2, s12
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:16
s_wait_storecnt 0
v_mov_b32 v2, s13
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:20
s_wait_storecnt 0
v_mov_b32 v2, s14
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:24
s_wait_storecnt 0
v_mov_b32 v2, s15
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:28
s_wait_storecnt 0
v_mov_b32 v2, s16
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:32
s_wait_storecnt 0
v_mov_b32 v2, s17
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:36
s_wait_storecnt 0
v_mov_b32 v2, s18
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:40
s_wait_storecnt 0
v_mov_b32 v2, s19
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:44
s_wait_storecnt 0
v_mov_b32 v2, s20
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:48
s_wait_storecnt 0
v_mov_b32 v2, s21
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:52
s_wait_storecnt 0
v_mov_b32 v2, s22
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:56
s_wait_storecnt 0
v_mov_b32 v2, s23
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:60
s_wait_storecnt 0
v_mov_b32 v2, s24
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:64
s_wait_storecnt 0
v_mov_b32 v2, s25
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:68
s_wait_storecnt 0
v_mov_b32 v2, s26
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:72
s_wait_storecnt 0
v_mov_b32 v2, s27
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:76
s_wait_storecnt 0
v_mov_b32 v2, s28
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:80
s_wait_storecnt 0
v_mov_b32 v2, s29
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:84
s_wait_storecnt 0
v_mov_b32 v2, s30
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:88
s_wait_storecnt 0
v_mov_b32 v2, s31
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:92
s_wait_storecnt 0
v_mov_b32 v2, s32
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:96
s_wait_storecnt 0
v_mov_b32 v2, s33
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:100
s_wait_storecnt 0
v_mov_b32 v2, s34
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:104
s_wait_storecnt 0
v_mov_b32 v2, s35
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:108
s_wait_storecnt 0
v_mov_b32 v2, s36
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:112
s_wait_storecnt 0
v_mov_b32 v2, s37
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:116
s_wait_storecnt 0
v_mov_b32 v2, s38
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:120
s_wait_storecnt 0
v_mov_b32 v2, s39
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:124
s_wait_storecnt 0
v_mov_b32 v2, s40
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:128
s_wait_storecnt 0
s_wait_storecnt 0
s_endpgm
.Lend_sentinel_45_1:
.size sentinel_45_1, .Lend_sentinel_45_1-sentinel_45_1
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel sentinel_45_1
.amdhsa_group_segment_fixed_size 0
.amdhsa_private_segment_fixed_size 0
.amdhsa_kernarg_size 8
.amdhsa_user_sgpr_count 2
.amdhsa_user_sgpr_dispatch_ptr 0
.amdhsa_user_sgpr_queue_ptr 0
.amdhsa_user_sgpr_kernarg_segment_ptr 1
.amdhsa_user_sgpr_dispatch_id 0
.amdhsa_user_sgpr_kernarg_preload_length 0
.amdhsa_user_sgpr_kernarg_preload_offset 0
.amdhsa_user_sgpr_private_segment_size 0
.amdhsa_wavefront_size32 1
.amdhsa_uses_dynamic_stack 0
.amdhsa_enable_private_segment 0
.amdhsa_system_sgpr_workgroup_id_x 0
.amdhsa_system_sgpr_workgroup_id_y 0
.amdhsa_system_sgpr_workgroup_id_z 0
.amdhsa_system_sgpr_workgroup_info 0
.amdhsa_system_vgpr_workitem_id 0
.amdhsa_next_free_vgpr 516
.amdhsa_next_free_sgpr 42
.amdhsa_named_barrier_count 0
.amdhsa_reserve_vcc 0
.amdhsa_float_round_mode_32 0
.amdhsa_float_round_mode_16_64 0
.amdhsa_float_denorm_mode_32 3
.amdhsa_float_denorm_mode_16_64 3
.amdhsa_fp16_overflow 0
.amdhsa_memory_ordered 1
.amdhsa_forward_progress 1
.amdhsa_round_robin_scheduling 0
.amdhsa_exception_fp_ieee_invalid_op 0
.amdhsa_exception_fp_denorm_src 0
.amdhsa_exception_fp_ieee_div_zero 0
.amdhsa_exception_fp_ieee_overflow 0
.amdhsa_exception_fp_ieee_underflow 0
.amdhsa_exception_fp_ieee_inexact 0
.amdhsa_exception_int_div_zero 0
.end_amdhsa_kernel
.text
.protected sentinel_82_0
.globl sentinel_82_0
.p2align 8
.type sentinel_82_0,@function
sentinel_82_0:
global_prefetch_b8 v0, s[0:1] scope:SCOPE_SE
v_nop
s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
s_load_b64 s[4:5], s[0:1], 0x0 nv
s_wait_kmcnt 0
s_mov_b32 s6, 0
s_setreg_b32 hwreg(HW_REG_WAVE_MODE, 12, 8), s6
s_nop 0
s_set_vgpr_msb 0x0000
s_nop 0
v_mov_b32 v1, 0x11000000
s_wait_idle
s_mov_b32 s7, 0x11000001
v_writelane_b32 v1, s7, 1
s_wait_idle
s_mov_b32 s7, 0x11000010
v_writelane_b32 v1, s7, 16
s_wait_idle
s_set_vgpr_msb 0x0041
s_nop 0
v_mov_b32 v1, 0x22000000
s_wait_idle
s_mov_b32 s7, 0x22000001
v_writelane_b32 v1, s7, 1
s_wait_idle
s_mov_b32 s7, 0x22000010
v_writelane_b32 v1, s7, 16
s_wait_idle
s_set_vgpr_msb 0x4182
s_nop 0
v_mov_b32 v1, 0x33000000
s_wait_idle
s_mov_b32 s7, 0x33000001
v_writelane_b32 v1, s7, 1
s_wait_idle
s_mov_b32 s7, 0x33000010
v_writelane_b32 v1, s7, 16
s_wait_idle
v_readlane_b32 s22, v1, 0
v_readlane_b32 s23, v1, 1
v_readlane_b32 s24, v1, 16
s_set_vgpr_msb 0x8241
s_nop 0
v_readlane_b32 s19, v1, 0
v_readlane_b32 s20, v1, 1
v_readlane_b32 s21, v1, 16
s_set_vgpr_msb 0x4100
s_nop 0
v_readlane_b32 s16, v1, 0
v_readlane_b32 s17, v1, 1
v_readlane_b32 s18, v1, 16
s_mov_b32 s8, 0x53545250
s_mov_b32 s9, 1
s_mov_b32 s10, 0x82
s_mov_b32 s11, 0
s_mov_b32 s40, 0x434f4d50
s_set_vgpr_msb 0x0082
s_nop 0
s_wait_idle
s_getreg_b32 s14, hwreg(HW_REG_WAVE_STATUS)
s_mov_b32 s38, exec_lo
s_getreg_b32 s12, hwreg(HW_REG_WAVE_MODE)
s_get_pc_i64 s[34:35]
.Lpcnext_sentinel_82_0:
s_add_co_u32 s34, s34, (.Lreturn_sentinel_82_0 - .Lpcnext_sentinel_82_0)
s_add_co_ci_u32 s35, s35, 0
s_nop 0
s_nop 0
.Lreturn_sentinel_82_0:
s_getreg_b32 s13, hwreg(HW_REG_WAVE_MODE)
s_mov_b32 s36, ttmp0
s_mov_b32 s37, ttmp1
s_mov_b32 s39, exec_lo
s_wait_idle
s_getreg_b32 s15, hwreg(HW_REG_WAVE_STATUS)
s_set_vgpr_msb 0x8200
s_nop 0
v_readlane_b32 s25, v1, 0
v_readlane_b32 s26, v1, 1
v_readlane_b32 s27, v1, 16
s_set_vgpr_msb 0x0041
s_nop 0
v_readlane_b32 s28, v1, 0
v_readlane_b32 s29, v1, 1
v_readlane_b32 s30, v1, 16
s_set_vgpr_msb 0x4182
s_nop 0
v_readlane_b32 s31, v1, 0
v_readlane_b32 s32, v1, 1
v_readlane_b32 s33, v1, 16
s_set_vgpr_msb 0x8200
s_nop 0
s_mov_b32 exec_lo, 1
v_mov_b32 v0, 0
v_mov_b32 v2, s8
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:0
s_wait_storecnt 0
v_mov_b32 v2, s9
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:4
s_wait_storecnt 0
v_mov_b32 v2, s10
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:8
s_wait_storecnt 0
v_mov_b32 v2, s11
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:12
s_wait_storecnt 0
v_mov_b32 v2, s12
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:16
s_wait_storecnt 0
v_mov_b32 v2, s13
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:20
s_wait_storecnt 0
v_mov_b32 v2, s14
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:24
s_wait_storecnt 0
v_mov_b32 v2, s15
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:28
s_wait_storecnt 0
v_mov_b32 v2, s16
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:32
s_wait_storecnt 0
v_mov_b32 v2, s17
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:36
s_wait_storecnt 0
v_mov_b32 v2, s18
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:40
s_wait_storecnt 0
v_mov_b32 v2, s19
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:44
s_wait_storecnt 0
v_mov_b32 v2, s20
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:48
s_wait_storecnt 0
v_mov_b32 v2, s21
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:52
s_wait_storecnt 0
v_mov_b32 v2, s22
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:56
s_wait_storecnt 0
v_mov_b32 v2, s23
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:60
s_wait_storecnt 0
v_mov_b32 v2, s24
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:64
s_wait_storecnt 0
v_mov_b32 v2, s25
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:68
s_wait_storecnt 0
v_mov_b32 v2, s26
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:72
s_wait_storecnt 0
v_mov_b32 v2, s27
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:76
s_wait_storecnt 0
v_mov_b32 v2, s28
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:80
s_wait_storecnt 0
v_mov_b32 v2, s29
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:84
s_wait_storecnt 0
v_mov_b32 v2, s30
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:88
s_wait_storecnt 0
v_mov_b32 v2, s31
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:92
s_wait_storecnt 0
v_mov_b32 v2, s32
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:96
s_wait_storecnt 0
v_mov_b32 v2, s33
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:100
s_wait_storecnt 0
v_mov_b32 v2, s34
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:104
s_wait_storecnt 0
v_mov_b32 v2, s35
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:108
s_wait_storecnt 0
v_mov_b32 v2, s36
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:112
s_wait_storecnt 0
v_mov_b32 v2, s37
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:116
s_wait_storecnt 0
v_mov_b32 v2, s38
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:120
s_wait_storecnt 0
v_mov_b32 v2, s39
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:124
s_wait_storecnt 0
v_mov_b32 v2, s40
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:128
s_wait_storecnt 0
s_wait_storecnt 0
s_endpgm
.Lend_sentinel_82_0:
.size sentinel_82_0, .Lend_sentinel_82_0-sentinel_82_0
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel sentinel_82_0
.amdhsa_group_segment_fixed_size 0
.amdhsa_private_segment_fixed_size 0
.amdhsa_kernarg_size 8
.amdhsa_user_sgpr_count 2
.amdhsa_user_sgpr_dispatch_ptr 0
.amdhsa_user_sgpr_queue_ptr 0
.amdhsa_user_sgpr_kernarg_segment_ptr 1
.amdhsa_user_sgpr_dispatch_id 0
.amdhsa_user_sgpr_kernarg_preload_length 0
.amdhsa_user_sgpr_kernarg_preload_offset 0
.amdhsa_user_sgpr_private_segment_size 0
.amdhsa_wavefront_size32 1
.amdhsa_uses_dynamic_stack 0
.amdhsa_enable_private_segment 0
.amdhsa_system_sgpr_workgroup_id_x 0
.amdhsa_system_sgpr_workgroup_id_y 0
.amdhsa_system_sgpr_workgroup_id_z 0
.amdhsa_system_sgpr_workgroup_info 0
.amdhsa_system_vgpr_workitem_id 0
.amdhsa_next_free_vgpr 516
.amdhsa_next_free_sgpr 42
.amdhsa_named_barrier_count 0
.amdhsa_reserve_vcc 0
.amdhsa_float_round_mode_32 0
.amdhsa_float_round_mode_16_64 0
.amdhsa_float_denorm_mode_32 3
.amdhsa_float_denorm_mode_16_64 3
.amdhsa_fp16_overflow 0
.amdhsa_memory_ordered 1
.amdhsa_forward_progress 1
.amdhsa_round_robin_scheduling 0
.amdhsa_exception_fp_ieee_invalid_op 0
.amdhsa_exception_fp_denorm_src 0
.amdhsa_exception_fp_ieee_div_zero 0
.amdhsa_exception_fp_ieee_overflow 0
.amdhsa_exception_fp_ieee_underflow 0
.amdhsa_exception_fp_ieee_inexact 0
.amdhsa_exception_int_div_zero 0
.end_amdhsa_kernel
.text
.protected sentinel_82_1
.globl sentinel_82_1
.p2align 8
.type sentinel_82_1,@function
sentinel_82_1:
global_prefetch_b8 v0, s[0:1] scope:SCOPE_SE
v_nop
s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
s_load_b64 s[4:5], s[0:1], 0x0 nv
s_wait_kmcnt 0
s_mov_b32 s6, 0
s_setreg_b32 hwreg(HW_REG_WAVE_MODE, 12, 8), s6
s_nop 0
s_set_vgpr_msb 0x0000
s_nop 0
v_mov_b32 v1, 0x11000000
s_wait_idle
s_mov_b32 s7, 0x11000001
v_writelane_b32 v1, s7, 1
s_wait_idle
s_mov_b32 s7, 0x11000010
v_writelane_b32 v1, s7, 16
s_wait_idle
s_set_vgpr_msb 0x0041
s_nop 0
v_mov_b32 v1, 0x22000000
s_wait_idle
s_mov_b32 s7, 0x22000001
v_writelane_b32 v1, s7, 1
s_wait_idle
s_mov_b32 s7, 0x22000010
v_writelane_b32 v1, s7, 16
s_wait_idle
s_set_vgpr_msb 0x4182
s_nop 0
v_mov_b32 v1, 0x33000000
s_wait_idle
s_mov_b32 s7, 0x33000001
v_writelane_b32 v1, s7, 1
s_wait_idle
s_mov_b32 s7, 0x33000010
v_writelane_b32 v1, s7, 16
s_wait_idle
v_readlane_b32 s22, v1, 0
v_readlane_b32 s23, v1, 1
v_readlane_b32 s24, v1, 16
s_set_vgpr_msb 0x8241
s_nop 0
v_readlane_b32 s19, v1, 0
v_readlane_b32 s20, v1, 1
v_readlane_b32 s21, v1, 16
s_set_vgpr_msb 0x4100
s_nop 0
v_readlane_b32 s16, v1, 0
v_readlane_b32 s17, v1, 1
v_readlane_b32 s18, v1, 16
s_mov_b32 s8, 0x53545250
s_mov_b32 s9, 1
s_mov_b32 s10, 0x82
s_mov_b32 s11, 1
s_mov_b32 s40, 0x434f4d50
s_set_vgpr_msb 0x0082
s_nop 0
s_wait_idle
s_getreg_b32 s14, hwreg(HW_REG_WAVE_STATUS)
s_mov_b32 s38, exec_lo
s_getreg_b32 s12, hwreg(HW_REG_WAVE_MODE)
s_get_pc_i64 s[34:35]
.Lpcnext_sentinel_82_1:
s_add_co_u32 s34, s34, (.Lreturn_sentinel_82_1 - .Lpcnext_sentinel_82_1)
s_add_co_ci_u32 s35, s35, 0
s_nop 0
s_trap 3
.Lreturn_sentinel_82_1:
s_getreg_b32 s13, hwreg(HW_REG_WAVE_MODE)
s_mov_b32 s36, ttmp0
s_mov_b32 s37, ttmp1
s_mov_b32 s39, exec_lo
s_wait_idle
s_getreg_b32 s15, hwreg(HW_REG_WAVE_STATUS)
s_set_vgpr_msb 0x8200
s_nop 0
v_readlane_b32 s25, v1, 0
v_readlane_b32 s26, v1, 1
v_readlane_b32 s27, v1, 16
s_set_vgpr_msb 0x0041
s_nop 0
v_readlane_b32 s28, v1, 0
v_readlane_b32 s29, v1, 1
v_readlane_b32 s30, v1, 16
s_set_vgpr_msb 0x4182
s_nop 0
v_readlane_b32 s31, v1, 0
v_readlane_b32 s32, v1, 1
v_readlane_b32 s33, v1, 16
s_set_vgpr_msb 0x8200
s_nop 0
s_mov_b32 exec_lo, 1
v_mov_b32 v0, 0
v_mov_b32 v2, s8
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:0
s_wait_storecnt 0
v_mov_b32 v2, s9
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:4
s_wait_storecnt 0
v_mov_b32 v2, s10
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:8
s_wait_storecnt 0
v_mov_b32 v2, s11
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:12
s_wait_storecnt 0
v_mov_b32 v2, s12
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:16
s_wait_storecnt 0
v_mov_b32 v2, s13
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:20
s_wait_storecnt 0
v_mov_b32 v2, s14
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:24
s_wait_storecnt 0
v_mov_b32 v2, s15
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:28
s_wait_storecnt 0
v_mov_b32 v2, s16
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:32
s_wait_storecnt 0
v_mov_b32 v2, s17
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:36
s_wait_storecnt 0
v_mov_b32 v2, s18
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:40
s_wait_storecnt 0
v_mov_b32 v2, s19
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:44
s_wait_storecnt 0
v_mov_b32 v2, s20
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:48
s_wait_storecnt 0
v_mov_b32 v2, s21
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:52
s_wait_storecnt 0
v_mov_b32 v2, s22
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:56
s_wait_storecnt 0
v_mov_b32 v2, s23
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:60
s_wait_storecnt 0
v_mov_b32 v2, s24
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:64
s_wait_storecnt 0
v_mov_b32 v2, s25
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:68
s_wait_storecnt 0
v_mov_b32 v2, s26
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:72
s_wait_storecnt 0
v_mov_b32 v2, s27
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:76
s_wait_storecnt 0
v_mov_b32 v2, s28
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:80
s_wait_storecnt 0
v_mov_b32 v2, s29
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:84
s_wait_storecnt 0
v_mov_b32 v2, s30
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:88
s_wait_storecnt 0
v_mov_b32 v2, s31
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:92
s_wait_storecnt 0
v_mov_b32 v2, s32
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:96
s_wait_storecnt 0
v_mov_b32 v2, s33
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:100
s_wait_storecnt 0
v_mov_b32 v2, s34
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:104
s_wait_storecnt 0
v_mov_b32 v2, s35
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:108
s_wait_storecnt 0
v_mov_b32 v2, s36
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:112
s_wait_storecnt 0
v_mov_b32 v2, s37
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:116
s_wait_storecnt 0
v_mov_b32 v2, s38
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:120
s_wait_storecnt 0
v_mov_b32 v2, s39
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:124
s_wait_storecnt 0
v_mov_b32 v2, s40
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:128
s_wait_storecnt 0
s_wait_storecnt 0
s_endpgm
.Lend_sentinel_82_1:
.size sentinel_82_1, .Lend_sentinel_82_1-sentinel_82_1
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel sentinel_82_1
.amdhsa_group_segment_fixed_size 0
.amdhsa_private_segment_fixed_size 0
.amdhsa_kernarg_size 8
.amdhsa_user_sgpr_count 2
.amdhsa_user_sgpr_dispatch_ptr 0
.amdhsa_user_sgpr_queue_ptr 0
.amdhsa_user_sgpr_kernarg_segment_ptr 1
.amdhsa_user_sgpr_dispatch_id 0
.amdhsa_user_sgpr_kernarg_preload_length 0
.amdhsa_user_sgpr_kernarg_preload_offset 0
.amdhsa_user_sgpr_private_segment_size 0
.amdhsa_wavefront_size32 1
.amdhsa_uses_dynamic_stack 0
.amdhsa_enable_private_segment 0
.amdhsa_system_sgpr_workgroup_id_x 0
.amdhsa_system_sgpr_workgroup_id_y 0
.amdhsa_system_sgpr_workgroup_id_z 0
.amdhsa_system_sgpr_workgroup_info 0
.amdhsa_system_vgpr_workitem_id 0
.amdhsa_next_free_vgpr 516
.amdhsa_next_free_sgpr 42
.amdhsa_named_barrier_count 0
.amdhsa_reserve_vcc 0
.amdhsa_float_round_mode_32 0
.amdhsa_float_round_mode_16_64 0
.amdhsa_float_denorm_mode_32 3
.amdhsa_float_denorm_mode_16_64 3
.amdhsa_fp16_overflow 0
.amdhsa_memory_ordered 1
.amdhsa_forward_progress 1
.amdhsa_round_robin_scheduling 0
.amdhsa_exception_fp_ieee_invalid_op 0
.amdhsa_exception_fp_denorm_src 0
.amdhsa_exception_fp_ieee_div_zero 0
.amdhsa_exception_fp_ieee_overflow 0
.amdhsa_exception_fp_ieee_underflow 0
.amdhsa_exception_fp_ieee_inexact 0
.amdhsa_exception_int_div_zero 0
.end_amdhsa_kernel
.text
.protected sentinel_05_0
.globl sentinel_05_0
.p2align 8
.type sentinel_05_0,@function
sentinel_05_0:
global_prefetch_b8 v0, s[0:1] scope:SCOPE_SE
v_nop
s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
s_load_b64 s[4:5], s[0:1], 0x0 nv
s_wait_kmcnt 0
s_mov_b32 s6, 0
s_setreg_b32 hwreg(HW_REG_WAVE_MODE, 12, 8), s6
s_nop 0
s_set_vgpr_msb 0x0000
s_nop 0
v_mov_b32 v1, 0x11000000
s_wait_idle
s_mov_b32 s7, 0x11000001
v_writelane_b32 v1, s7, 1
s_wait_idle
s_mov_b32 s7, 0x11000010
v_writelane_b32 v1, s7, 16
s_wait_idle
s_set_vgpr_msb 0x0041
s_nop 0
v_mov_b32 v1, 0x22000000
s_wait_idle
s_mov_b32 s7, 0x22000001
v_writelane_b32 v1, s7, 1
s_wait_idle
s_mov_b32 s7, 0x22000010
v_writelane_b32 v1, s7, 16
s_wait_idle
s_set_vgpr_msb 0x4182
s_nop 0
v_mov_b32 v1, 0x33000000
s_wait_idle
s_mov_b32 s7, 0x33000001
v_writelane_b32 v1, s7, 1
s_wait_idle
s_mov_b32 s7, 0x33000010
v_writelane_b32 v1, s7, 16
s_wait_idle
v_readlane_b32 s22, v1, 0
v_readlane_b32 s23, v1, 1
v_readlane_b32 s24, v1, 16
s_set_vgpr_msb 0x8241
s_nop 0
v_readlane_b32 s19, v1, 0
v_readlane_b32 s20, v1, 1
v_readlane_b32 s21, v1, 16
s_set_vgpr_msb 0x4100
s_nop 0
v_readlane_b32 s16, v1, 0
v_readlane_b32 s17, v1, 1
v_readlane_b32 s18, v1, 16
s_mov_b32 s8, 0x53545250
s_mov_b32 s9, 1
s_mov_b32 s10, 0x05
s_mov_b32 s11, 0
s_mov_b32 s40, 0x434f4d50
s_set_vgpr_msb 0x0005
s_nop 0
s_wait_idle
s_getreg_b32 s14, hwreg(HW_REG_WAVE_STATUS)
s_mov_b32 s38, exec_lo
s_getreg_b32 s12, hwreg(HW_REG_WAVE_MODE)
s_get_pc_i64 s[34:35]
.Lpcnext_sentinel_05_0:
s_add_co_u32 s34, s34, (.Lreturn_sentinel_05_0 - .Lpcnext_sentinel_05_0)
s_add_co_ci_u32 s35, s35, 0
s_nop 0
s_nop 0
.Lreturn_sentinel_05_0:
s_getreg_b32 s13, hwreg(HW_REG_WAVE_MODE)
s_mov_b32 s36, ttmp0
s_mov_b32 s37, ttmp1
s_mov_b32 s39, exec_lo
s_wait_idle
s_getreg_b32 s15, hwreg(HW_REG_WAVE_STATUS)
s_set_vgpr_msb 0x0500
s_nop 0
v_readlane_b32 s25, v1, 0
v_readlane_b32 s26, v1, 1
v_readlane_b32 s27, v1, 16
s_set_vgpr_msb 0x0041
s_nop 0
v_readlane_b32 s28, v1, 0
v_readlane_b32 s29, v1, 1
v_readlane_b32 s30, v1, 16
s_set_vgpr_msb 0x4182
s_nop 0
v_readlane_b32 s31, v1, 0
v_readlane_b32 s32, v1, 1
v_readlane_b32 s33, v1, 16
s_set_vgpr_msb 0x8200
s_nop 0
s_mov_b32 exec_lo, 1
v_mov_b32 v0, 0
v_mov_b32 v2, s8
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:0
s_wait_storecnt 0
v_mov_b32 v2, s9
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:4
s_wait_storecnt 0
v_mov_b32 v2, s10
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:8
s_wait_storecnt 0
v_mov_b32 v2, s11
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:12
s_wait_storecnt 0
v_mov_b32 v2, s12
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:16
s_wait_storecnt 0
v_mov_b32 v2, s13
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:20
s_wait_storecnt 0
v_mov_b32 v2, s14
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:24
s_wait_storecnt 0
v_mov_b32 v2, s15
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:28
s_wait_storecnt 0
v_mov_b32 v2, s16
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:32
s_wait_storecnt 0
v_mov_b32 v2, s17
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:36
s_wait_storecnt 0
v_mov_b32 v2, s18
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:40
s_wait_storecnt 0
v_mov_b32 v2, s19
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:44
s_wait_storecnt 0
v_mov_b32 v2, s20
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:48
s_wait_storecnt 0
v_mov_b32 v2, s21
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:52
s_wait_storecnt 0
v_mov_b32 v2, s22
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:56
s_wait_storecnt 0
v_mov_b32 v2, s23
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:60
s_wait_storecnt 0
v_mov_b32 v2, s24
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:64
s_wait_storecnt 0
v_mov_b32 v2, s25
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:68
s_wait_storecnt 0
v_mov_b32 v2, s26
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:72
s_wait_storecnt 0
v_mov_b32 v2, s27
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:76
s_wait_storecnt 0
v_mov_b32 v2, s28
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:80
s_wait_storecnt 0
v_mov_b32 v2, s29
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:84
s_wait_storecnt 0
v_mov_b32 v2, s30
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:88
s_wait_storecnt 0
v_mov_b32 v2, s31
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:92
s_wait_storecnt 0
v_mov_b32 v2, s32
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:96
s_wait_storecnt 0
v_mov_b32 v2, s33
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:100
s_wait_storecnt 0
v_mov_b32 v2, s34
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:104
s_wait_storecnt 0
v_mov_b32 v2, s35
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:108
s_wait_storecnt 0
v_mov_b32 v2, s36
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:112
s_wait_storecnt 0
v_mov_b32 v2, s37
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:116
s_wait_storecnt 0
v_mov_b32 v2, s38
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:120
s_wait_storecnt 0
v_mov_b32 v2, s39
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:124
s_wait_storecnt 0
v_mov_b32 v2, s40
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:128
s_wait_storecnt 0
s_wait_storecnt 0
s_endpgm
.Lend_sentinel_05_0:
.size sentinel_05_0, .Lend_sentinel_05_0-sentinel_05_0
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel sentinel_05_0
.amdhsa_group_segment_fixed_size 0
.amdhsa_private_segment_fixed_size 0
.amdhsa_kernarg_size 8
.amdhsa_user_sgpr_count 2
.amdhsa_user_sgpr_dispatch_ptr 0
.amdhsa_user_sgpr_queue_ptr 0
.amdhsa_user_sgpr_kernarg_segment_ptr 1
.amdhsa_user_sgpr_dispatch_id 0
.amdhsa_user_sgpr_kernarg_preload_length 0
.amdhsa_user_sgpr_kernarg_preload_offset 0
.amdhsa_user_sgpr_private_segment_size 0
.amdhsa_wavefront_size32 1
.amdhsa_uses_dynamic_stack 0
.amdhsa_enable_private_segment 0
.amdhsa_system_sgpr_workgroup_id_x 0
.amdhsa_system_sgpr_workgroup_id_y 0
.amdhsa_system_sgpr_workgroup_id_z 0
.amdhsa_system_sgpr_workgroup_info 0
.amdhsa_system_vgpr_workitem_id 0
.amdhsa_next_free_vgpr 516
.amdhsa_next_free_sgpr 42
.amdhsa_named_barrier_count 0
.amdhsa_reserve_vcc 0
.amdhsa_float_round_mode_32 0
.amdhsa_float_round_mode_16_64 0
.amdhsa_float_denorm_mode_32 3
.amdhsa_float_denorm_mode_16_64 3
.amdhsa_fp16_overflow 0
.amdhsa_memory_ordered 1
.amdhsa_forward_progress 1
.amdhsa_round_robin_scheduling 0
.amdhsa_exception_fp_ieee_invalid_op 0
.amdhsa_exception_fp_denorm_src 0
.amdhsa_exception_fp_ieee_div_zero 0
.amdhsa_exception_fp_ieee_overflow 0
.amdhsa_exception_fp_ieee_underflow 0
.amdhsa_exception_fp_ieee_inexact 0
.amdhsa_exception_int_div_zero 0
.end_amdhsa_kernel
.text
.protected sentinel_05_1
.globl sentinel_05_1
.p2align 8
.type sentinel_05_1,@function
sentinel_05_1:
global_prefetch_b8 v0, s[0:1] scope:SCOPE_SE
v_nop
s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
s_load_b64 s[4:5], s[0:1], 0x0 nv
s_wait_kmcnt 0
s_mov_b32 s6, 0
s_setreg_b32 hwreg(HW_REG_WAVE_MODE, 12, 8), s6
s_nop 0
s_set_vgpr_msb 0x0000
s_nop 0
v_mov_b32 v1, 0x11000000
s_wait_idle
s_mov_b32 s7, 0x11000001
v_writelane_b32 v1, s7, 1
s_wait_idle
s_mov_b32 s7, 0x11000010
v_writelane_b32 v1, s7, 16
s_wait_idle
s_set_vgpr_msb 0x0041
s_nop 0
v_mov_b32 v1, 0x22000000
s_wait_idle
s_mov_b32 s7, 0x22000001
v_writelane_b32 v1, s7, 1
s_wait_idle
s_mov_b32 s7, 0x22000010
v_writelane_b32 v1, s7, 16
s_wait_idle
s_set_vgpr_msb 0x4182
s_nop 0
v_mov_b32 v1, 0x33000000
s_wait_idle
s_mov_b32 s7, 0x33000001
v_writelane_b32 v1, s7, 1
s_wait_idle
s_mov_b32 s7, 0x33000010
v_writelane_b32 v1, s7, 16
s_wait_idle
v_readlane_b32 s22, v1, 0
v_readlane_b32 s23, v1, 1
v_readlane_b32 s24, v1, 16
s_set_vgpr_msb 0x8241
s_nop 0
v_readlane_b32 s19, v1, 0
v_readlane_b32 s20, v1, 1
v_readlane_b32 s21, v1, 16
s_set_vgpr_msb 0x4100
s_nop 0
v_readlane_b32 s16, v1, 0
v_readlane_b32 s17, v1, 1
v_readlane_b32 s18, v1, 16
s_mov_b32 s8, 0x53545250
s_mov_b32 s9, 1
s_mov_b32 s10, 0x05
s_mov_b32 s11, 1
s_mov_b32 s40, 0x434f4d50
s_set_vgpr_msb 0x0005
s_nop 0
s_wait_idle
s_getreg_b32 s14, hwreg(HW_REG_WAVE_STATUS)
s_mov_b32 s38, exec_lo
s_getreg_b32 s12, hwreg(HW_REG_WAVE_MODE)
s_get_pc_i64 s[34:35]
.Lpcnext_sentinel_05_1:
s_add_co_u32 s34, s34, (.Lreturn_sentinel_05_1 - .Lpcnext_sentinel_05_1)
s_add_co_ci_u32 s35, s35, 0
s_nop 0
s_trap 3
.Lreturn_sentinel_05_1:
s_getreg_b32 s13, hwreg(HW_REG_WAVE_MODE)
s_mov_b32 s36, ttmp0
s_mov_b32 s37, ttmp1
s_mov_b32 s39, exec_lo
s_wait_idle
s_getreg_b32 s15, hwreg(HW_REG_WAVE_STATUS)
s_set_vgpr_msb 0x0500
s_nop 0
v_readlane_b32 s25, v1, 0
v_readlane_b32 s26, v1, 1
v_readlane_b32 s27, v1, 16
s_set_vgpr_msb 0x0041
s_nop 0
v_readlane_b32 s28, v1, 0
v_readlane_b32 s29, v1, 1
v_readlane_b32 s30, v1, 16
s_set_vgpr_msb 0x4182
s_nop 0
v_readlane_b32 s31, v1, 0
v_readlane_b32 s32, v1, 1
v_readlane_b32 s33, v1, 16
s_set_vgpr_msb 0x8200
s_nop 0
s_mov_b32 exec_lo, 1
v_mov_b32 v0, 0
v_mov_b32 v2, s8
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:0
s_wait_storecnt 0
v_mov_b32 v2, s9
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:4
s_wait_storecnt 0
v_mov_b32 v2, s10
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:8
s_wait_storecnt 0
v_mov_b32 v2, s11
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:12
s_wait_storecnt 0
v_mov_b32 v2, s12
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:16
s_wait_storecnt 0
v_mov_b32 v2, s13
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:20
s_wait_storecnt 0
v_mov_b32 v2, s14
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:24
s_wait_storecnt 0
v_mov_b32 v2, s15
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:28
s_wait_storecnt 0
v_mov_b32 v2, s16
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:32
s_wait_storecnt 0
v_mov_b32 v2, s17
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:36
s_wait_storecnt 0
v_mov_b32 v2, s18
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:40
s_wait_storecnt 0
v_mov_b32 v2, s19
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:44
s_wait_storecnt 0
v_mov_b32 v2, s20
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:48
s_wait_storecnt 0
v_mov_b32 v2, s21
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:52
s_wait_storecnt 0
v_mov_b32 v2, s22
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:56
s_wait_storecnt 0
v_mov_b32 v2, s23
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:60
s_wait_storecnt 0
v_mov_b32 v2, s24
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:64
s_wait_storecnt 0
v_mov_b32 v2, s25
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:68
s_wait_storecnt 0
v_mov_b32 v2, s26
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:72
s_wait_storecnt 0
v_mov_b32 v2, s27
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:76
s_wait_storecnt 0
v_mov_b32 v2, s28
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:80
s_wait_storecnt 0
v_mov_b32 v2, s29
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:84
s_wait_storecnt 0
v_mov_b32 v2, s30
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:88
s_wait_storecnt 0
v_mov_b32 v2, s31
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:92
s_wait_storecnt 0
v_mov_b32 v2, s32
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:96
s_wait_storecnt 0
v_mov_b32 v2, s33
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:100
s_wait_storecnt 0
v_mov_b32 v2, s34
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:104
s_wait_storecnt 0
v_mov_b32 v2, s35
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:108
s_wait_storecnt 0
v_mov_b32 v2, s36
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:112
s_wait_storecnt 0
v_mov_b32 v2, s37
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:116
s_wait_storecnt 0
v_mov_b32 v2, s38
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:120
s_wait_storecnt 0
v_mov_b32 v2, s39
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:124
s_wait_storecnt 0
v_mov_b32 v2, s40
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:128
s_wait_storecnt 0
s_wait_storecnt 0
s_endpgm
.Lend_sentinel_05_1:
.size sentinel_05_1, .Lend_sentinel_05_1-sentinel_05_1
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel sentinel_05_1
.amdhsa_group_segment_fixed_size 0
.amdhsa_private_segment_fixed_size 0
.amdhsa_kernarg_size 8
.amdhsa_user_sgpr_count 2
.amdhsa_user_sgpr_dispatch_ptr 0
.amdhsa_user_sgpr_queue_ptr 0
.amdhsa_user_sgpr_kernarg_segment_ptr 1
.amdhsa_user_sgpr_dispatch_id 0
.amdhsa_user_sgpr_kernarg_preload_length 0
.amdhsa_user_sgpr_kernarg_preload_offset 0
.amdhsa_user_sgpr_private_segment_size 0
.amdhsa_wavefront_size32 1
.amdhsa_uses_dynamic_stack 0
.amdhsa_enable_private_segment 0
.amdhsa_system_sgpr_workgroup_id_x 0
.amdhsa_system_sgpr_workgroup_id_y 0
.amdhsa_system_sgpr_workgroup_id_z 0
.amdhsa_system_sgpr_workgroup_info 0
.amdhsa_system_vgpr_workitem_id 0
.amdhsa_next_free_vgpr 516
.amdhsa_next_free_sgpr 42
.amdhsa_named_barrier_count 0
.amdhsa_reserve_vcc 0
.amdhsa_float_round_mode_32 0
.amdhsa_float_round_mode_16_64 0
.amdhsa_float_denorm_mode_32 3
.amdhsa_float_denorm_mode_16_64 3
.amdhsa_fp16_overflow 0
.amdhsa_memory_ordered 1
.amdhsa_forward_progress 1
.amdhsa_round_robin_scheduling 0
.amdhsa_exception_fp_ieee_invalid_op 0
.amdhsa_exception_fp_denorm_src 0
.amdhsa_exception_fp_ieee_div_zero 0
.amdhsa_exception_fp_ieee_overflow 0
.amdhsa_exception_fp_ieee_underflow 0
.amdhsa_exception_fp_ieee_inexact 0
.amdhsa_exception_int_div_zero 0
.end_amdhsa_kernel
.text
.protected sentinel_81_0
.globl sentinel_81_0
.p2align 8
.type sentinel_81_0,@function
sentinel_81_0:
global_prefetch_b8 v0, s[0:1] scope:SCOPE_SE
v_nop
s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
s_load_b64 s[4:5], s[0:1], 0x0 nv
s_wait_kmcnt 0
s_mov_b32 s6, 0
s_setreg_b32 hwreg(HW_REG_WAVE_MODE, 12, 8), s6
s_nop 0
s_set_vgpr_msb 0x0000
s_nop 0
v_mov_b32 v1, 0x11000000
s_wait_idle
s_mov_b32 s7, 0x11000001
v_writelane_b32 v1, s7, 1
s_wait_idle
s_mov_b32 s7, 0x11000010
v_writelane_b32 v1, s7, 16
s_wait_idle
s_set_vgpr_msb 0x0041
s_nop 0
v_mov_b32 v1, 0x22000000
s_wait_idle
s_mov_b32 s7, 0x22000001
v_writelane_b32 v1, s7, 1
s_wait_idle
s_mov_b32 s7, 0x22000010
v_writelane_b32 v1, s7, 16
s_wait_idle
s_set_vgpr_msb 0x4182
s_nop 0
v_mov_b32 v1, 0x33000000
s_wait_idle
s_mov_b32 s7, 0x33000001
v_writelane_b32 v1, s7, 1
s_wait_idle
s_mov_b32 s7, 0x33000010
v_writelane_b32 v1, s7, 16
s_wait_idle
v_readlane_b32 s22, v1, 0
v_readlane_b32 s23, v1, 1
v_readlane_b32 s24, v1, 16
s_set_vgpr_msb 0x8241
s_nop 0
v_readlane_b32 s19, v1, 0
v_readlane_b32 s20, v1, 1
v_readlane_b32 s21, v1, 16
s_set_vgpr_msb 0x4100
s_nop 0
v_readlane_b32 s16, v1, 0
v_readlane_b32 s17, v1, 1
v_readlane_b32 s18, v1, 16
s_mov_b32 s8, 0x53545250
s_mov_b32 s9, 1
s_mov_b32 s10, 0x81
s_mov_b32 s11, 0
s_mov_b32 s40, 0x434f4d50
s_set_vgpr_msb 0x0081
s_nop 0
s_wait_idle
s_getreg_b32 s14, hwreg(HW_REG_WAVE_STATUS)
s_mov_b32 s38, exec_lo
s_getreg_b32 s12, hwreg(HW_REG_WAVE_MODE)
s_get_pc_i64 s[34:35]
.Lpcnext_sentinel_81_0:
s_add_co_u32 s34, s34, (.Lreturn_sentinel_81_0 - .Lpcnext_sentinel_81_0)
s_add_co_ci_u32 s35, s35, 0
s_nop 0
s_nop 0
.Lreturn_sentinel_81_0:
s_getreg_b32 s13, hwreg(HW_REG_WAVE_MODE)
s_mov_b32 s36, ttmp0
s_mov_b32 s37, ttmp1
s_mov_b32 s39, exec_lo
s_wait_idle
s_getreg_b32 s15, hwreg(HW_REG_WAVE_STATUS)
s_set_vgpr_msb 0x8100
s_nop 0
v_readlane_b32 s25, v1, 0
v_readlane_b32 s26, v1, 1
v_readlane_b32 s27, v1, 16
s_set_vgpr_msb 0x0041
s_nop 0
v_readlane_b32 s28, v1, 0
v_readlane_b32 s29, v1, 1
v_readlane_b32 s30, v1, 16
s_set_vgpr_msb 0x4182
s_nop 0
v_readlane_b32 s31, v1, 0
v_readlane_b32 s32, v1, 1
v_readlane_b32 s33, v1, 16
s_set_vgpr_msb 0x8200
s_nop 0
s_mov_b32 exec_lo, 1
v_mov_b32 v0, 0
v_mov_b32 v2, s8
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:0
s_wait_storecnt 0
v_mov_b32 v2, s9
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:4
s_wait_storecnt 0
v_mov_b32 v2, s10
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:8
s_wait_storecnt 0
v_mov_b32 v2, s11
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:12
s_wait_storecnt 0
v_mov_b32 v2, s12
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:16
s_wait_storecnt 0
v_mov_b32 v2, s13
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:20
s_wait_storecnt 0
v_mov_b32 v2, s14
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:24
s_wait_storecnt 0
v_mov_b32 v2, s15
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:28
s_wait_storecnt 0
v_mov_b32 v2, s16
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:32
s_wait_storecnt 0
v_mov_b32 v2, s17
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:36
s_wait_storecnt 0
v_mov_b32 v2, s18
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:40
s_wait_storecnt 0
v_mov_b32 v2, s19
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:44
s_wait_storecnt 0
v_mov_b32 v2, s20
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:48
s_wait_storecnt 0
v_mov_b32 v2, s21
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:52
s_wait_storecnt 0
v_mov_b32 v2, s22
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:56
s_wait_storecnt 0
v_mov_b32 v2, s23
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:60
s_wait_storecnt 0
v_mov_b32 v2, s24
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:64
s_wait_storecnt 0
v_mov_b32 v2, s25
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:68
s_wait_storecnt 0
v_mov_b32 v2, s26
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:72
s_wait_storecnt 0
v_mov_b32 v2, s27
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:76
s_wait_storecnt 0
v_mov_b32 v2, s28
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:80
s_wait_storecnt 0
v_mov_b32 v2, s29
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:84
s_wait_storecnt 0
v_mov_b32 v2, s30
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:88
s_wait_storecnt 0
v_mov_b32 v2, s31
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:92
s_wait_storecnt 0
v_mov_b32 v2, s32
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:96
s_wait_storecnt 0
v_mov_b32 v2, s33
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:100
s_wait_storecnt 0
v_mov_b32 v2, s34
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:104
s_wait_storecnt 0
v_mov_b32 v2, s35
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:108
s_wait_storecnt 0
v_mov_b32 v2, s36
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:112
s_wait_storecnt 0
v_mov_b32 v2, s37
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:116
s_wait_storecnt 0
v_mov_b32 v2, s38
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:120
s_wait_storecnt 0
v_mov_b32 v2, s39
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:124
s_wait_storecnt 0
v_mov_b32 v2, s40
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:128
s_wait_storecnt 0
s_wait_storecnt 0
s_endpgm
.Lend_sentinel_81_0:
.size sentinel_81_0, .Lend_sentinel_81_0-sentinel_81_0
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel sentinel_81_0
.amdhsa_group_segment_fixed_size 0
.amdhsa_private_segment_fixed_size 0
.amdhsa_kernarg_size 8
.amdhsa_user_sgpr_count 2
.amdhsa_user_sgpr_dispatch_ptr 0
.amdhsa_user_sgpr_queue_ptr 0
.amdhsa_user_sgpr_kernarg_segment_ptr 1
.amdhsa_user_sgpr_dispatch_id 0
.amdhsa_user_sgpr_kernarg_preload_length 0
.amdhsa_user_sgpr_kernarg_preload_offset 0
.amdhsa_user_sgpr_private_segment_size 0
.amdhsa_wavefront_size32 1
.amdhsa_uses_dynamic_stack 0
.amdhsa_enable_private_segment 0
.amdhsa_system_sgpr_workgroup_id_x 0
.amdhsa_system_sgpr_workgroup_id_y 0
.amdhsa_system_sgpr_workgroup_id_z 0
.amdhsa_system_sgpr_workgroup_info 0
.amdhsa_system_vgpr_workitem_id 0
.amdhsa_next_free_vgpr 516
.amdhsa_next_free_sgpr 42
.amdhsa_named_barrier_count 0
.amdhsa_reserve_vcc 0
.amdhsa_float_round_mode_32 0
.amdhsa_float_round_mode_16_64 0
.amdhsa_float_denorm_mode_32 3
.amdhsa_float_denorm_mode_16_64 3
.amdhsa_fp16_overflow 0
.amdhsa_memory_ordered 1
.amdhsa_forward_progress 1
.amdhsa_round_robin_scheduling 0
.amdhsa_exception_fp_ieee_invalid_op 0
.amdhsa_exception_fp_denorm_src 0
.amdhsa_exception_fp_ieee_div_zero 0
.amdhsa_exception_fp_ieee_overflow 0
.amdhsa_exception_fp_ieee_underflow 0
.amdhsa_exception_fp_ieee_inexact 0
.amdhsa_exception_int_div_zero 0
.end_amdhsa_kernel
.text
.protected sentinel_81_1
.globl sentinel_81_1
.p2align 8
.type sentinel_81_1,@function
sentinel_81_1:
global_prefetch_b8 v0, s[0:1] scope:SCOPE_SE
v_nop
s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
s_load_b64 s[4:5], s[0:1], 0x0 nv
s_wait_kmcnt 0
s_mov_b32 s6, 0
s_setreg_b32 hwreg(HW_REG_WAVE_MODE, 12, 8), s6
s_nop 0
s_set_vgpr_msb 0x0000
s_nop 0
v_mov_b32 v1, 0x11000000
s_wait_idle
s_mov_b32 s7, 0x11000001
v_writelane_b32 v1, s7, 1
s_wait_idle
s_mov_b32 s7, 0x11000010
v_writelane_b32 v1, s7, 16
s_wait_idle
s_set_vgpr_msb 0x0041
s_nop 0
v_mov_b32 v1, 0x22000000
s_wait_idle
s_mov_b32 s7, 0x22000001
v_writelane_b32 v1, s7, 1
s_wait_idle
s_mov_b32 s7, 0x22000010
v_writelane_b32 v1, s7, 16
s_wait_idle
s_set_vgpr_msb 0x4182
s_nop 0
v_mov_b32 v1, 0x33000000
s_wait_idle
s_mov_b32 s7, 0x33000001
v_writelane_b32 v1, s7, 1
s_wait_idle
s_mov_b32 s7, 0x33000010
v_writelane_b32 v1, s7, 16
s_wait_idle
v_readlane_b32 s22, v1, 0
v_readlane_b32 s23, v1, 1
v_readlane_b32 s24, v1, 16
s_set_vgpr_msb 0x8241
s_nop 0
v_readlane_b32 s19, v1, 0
v_readlane_b32 s20, v1, 1
v_readlane_b32 s21, v1, 16
s_set_vgpr_msb 0x4100
s_nop 0
v_readlane_b32 s16, v1, 0
v_readlane_b32 s17, v1, 1
v_readlane_b32 s18, v1, 16
s_mov_b32 s8, 0x53545250
s_mov_b32 s9, 1
s_mov_b32 s10, 0x81
s_mov_b32 s11, 1
s_mov_b32 s40, 0x434f4d50
s_set_vgpr_msb 0x0081
s_nop 0
s_wait_idle
s_getreg_b32 s14, hwreg(HW_REG_WAVE_STATUS)
s_mov_b32 s38, exec_lo
s_getreg_b32 s12, hwreg(HW_REG_WAVE_MODE)
s_get_pc_i64 s[34:35]
.Lpcnext_sentinel_81_1:
s_add_co_u32 s34, s34, (.Lreturn_sentinel_81_1 - .Lpcnext_sentinel_81_1)
s_add_co_ci_u32 s35, s35, 0
s_nop 0
s_trap 3
.Lreturn_sentinel_81_1:
s_getreg_b32 s13, hwreg(HW_REG_WAVE_MODE)
s_mov_b32 s36, ttmp0
s_mov_b32 s37, ttmp1
s_mov_b32 s39, exec_lo
s_wait_idle
s_getreg_b32 s15, hwreg(HW_REG_WAVE_STATUS)
s_set_vgpr_msb 0x8100
s_nop 0
v_readlane_b32 s25, v1, 0
v_readlane_b32 s26, v1, 1
v_readlane_b32 s27, v1, 16
s_set_vgpr_msb 0x0041
s_nop 0
v_readlane_b32 s28, v1, 0
v_readlane_b32 s29, v1, 1
v_readlane_b32 s30, v1, 16
s_set_vgpr_msb 0x4182
s_nop 0
v_readlane_b32 s31, v1, 0
v_readlane_b32 s32, v1, 1
v_readlane_b32 s33, v1, 16
s_set_vgpr_msb 0x8200
s_nop 0
s_mov_b32 exec_lo, 1
v_mov_b32 v0, 0
v_mov_b32 v2, s8
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:0
s_wait_storecnt 0
v_mov_b32 v2, s9
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:4
s_wait_storecnt 0
v_mov_b32 v2, s10
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:8
s_wait_storecnt 0
v_mov_b32 v2, s11
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:12
s_wait_storecnt 0
v_mov_b32 v2, s12
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:16
s_wait_storecnt 0
v_mov_b32 v2, s13
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:20
s_wait_storecnt 0
v_mov_b32 v2, s14
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:24
s_wait_storecnt 0
v_mov_b32 v2, s15
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:28
s_wait_storecnt 0
v_mov_b32 v2, s16
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:32
s_wait_storecnt 0
v_mov_b32 v2, s17
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:36
s_wait_storecnt 0
v_mov_b32 v2, s18
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:40
s_wait_storecnt 0
v_mov_b32 v2, s19
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:44
s_wait_storecnt 0
v_mov_b32 v2, s20
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:48
s_wait_storecnt 0
v_mov_b32 v2, s21
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:52
s_wait_storecnt 0
v_mov_b32 v2, s22
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:56
s_wait_storecnt 0
v_mov_b32 v2, s23
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:60
s_wait_storecnt 0
v_mov_b32 v2, s24
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:64
s_wait_storecnt 0
v_mov_b32 v2, s25
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:68
s_wait_storecnt 0
v_mov_b32 v2, s26
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:72
s_wait_storecnt 0
v_mov_b32 v2, s27
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:76
s_wait_storecnt 0
v_mov_b32 v2, s28
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:80
s_wait_storecnt 0
v_mov_b32 v2, s29
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:84
s_wait_storecnt 0
v_mov_b32 v2, s30
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:88
s_wait_storecnt 0
v_mov_b32 v2, s31
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:92
s_wait_storecnt 0
v_mov_b32 v2, s32
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:96
s_wait_storecnt 0
v_mov_b32 v2, s33
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:100
s_wait_storecnt 0
v_mov_b32 v2, s34
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:104
s_wait_storecnt 0
v_mov_b32 v2, s35
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:108
s_wait_storecnt 0
v_mov_b32 v2, s36
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:112
s_wait_storecnt 0
v_mov_b32 v2, s37
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:116
s_wait_storecnt 0
v_mov_b32 v2, s38
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:120
s_wait_storecnt 0
v_mov_b32 v2, s39
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:124
s_wait_storecnt 0
v_mov_b32 v2, s40
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:128
s_wait_storecnt 0
s_wait_storecnt 0
s_endpgm
.Lend_sentinel_81_1:
.size sentinel_81_1, .Lend_sentinel_81_1-sentinel_81_1
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel sentinel_81_1
.amdhsa_group_segment_fixed_size 0
.amdhsa_private_segment_fixed_size 0
.amdhsa_kernarg_size 8
.amdhsa_user_sgpr_count 2
.amdhsa_user_sgpr_dispatch_ptr 0
.amdhsa_user_sgpr_queue_ptr 0
.amdhsa_user_sgpr_kernarg_segment_ptr 1
.amdhsa_user_sgpr_dispatch_id 0
.amdhsa_user_sgpr_kernarg_preload_length 0
.amdhsa_user_sgpr_kernarg_preload_offset 0
.amdhsa_user_sgpr_private_segment_size 0
.amdhsa_wavefront_size32 1
.amdhsa_uses_dynamic_stack 0
.amdhsa_enable_private_segment 0
.amdhsa_system_sgpr_workgroup_id_x 0
.amdhsa_system_sgpr_workgroup_id_y 0
.amdhsa_system_sgpr_workgroup_id_z 0
.amdhsa_system_sgpr_workgroup_info 0
.amdhsa_system_vgpr_workitem_id 0
.amdhsa_next_free_vgpr 516
.amdhsa_next_free_sgpr 42
.amdhsa_named_barrier_count 0
.amdhsa_reserve_vcc 0
.amdhsa_float_round_mode_32 0
.amdhsa_float_round_mode_16_64 0
.amdhsa_float_denorm_mode_32 3
.amdhsa_float_denorm_mode_16_64 3
.amdhsa_fp16_overflow 0
.amdhsa_memory_ordered 1
.amdhsa_forward_progress 1
.amdhsa_round_robin_scheduling 0
.amdhsa_exception_fp_ieee_invalid_op 0
.amdhsa_exception_fp_denorm_src 0
.amdhsa_exception_fp_ieee_div_zero 0
.amdhsa_exception_fp_ieee_overflow 0
.amdhsa_exception_fp_ieee_underflow 0
.amdhsa_exception_fp_ieee_inexact 0
.amdhsa_exception_int_div_zero 0
.end_amdhsa_kernel
.text
.protected sentinel_85_0
.globl sentinel_85_0
.p2align 8
.type sentinel_85_0,@function
sentinel_85_0:
global_prefetch_b8 v0, s[0:1] scope:SCOPE_SE
v_nop
s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
s_load_b64 s[4:5], s[0:1], 0x0 nv
s_wait_kmcnt 0
s_mov_b32 s6, 0
s_setreg_b32 hwreg(HW_REG_WAVE_MODE, 12, 8), s6
s_nop 0
s_set_vgpr_msb 0x0000
s_nop 0
v_mov_b32 v1, 0x11000000
s_wait_idle
s_mov_b32 s7, 0x11000001
v_writelane_b32 v1, s7, 1
s_wait_idle
s_mov_b32 s7, 0x11000010
v_writelane_b32 v1, s7, 16
s_wait_idle
s_set_vgpr_msb 0x0041
s_nop 0
v_mov_b32 v1, 0x22000000
s_wait_idle
s_mov_b32 s7, 0x22000001
v_writelane_b32 v1, s7, 1
s_wait_idle
s_mov_b32 s7, 0x22000010
v_writelane_b32 v1, s7, 16
s_wait_idle
s_set_vgpr_msb 0x4182
s_nop 0
v_mov_b32 v1, 0x33000000
s_wait_idle
s_mov_b32 s7, 0x33000001
v_writelane_b32 v1, s7, 1
s_wait_idle
s_mov_b32 s7, 0x33000010
v_writelane_b32 v1, s7, 16
s_wait_idle
v_readlane_b32 s22, v1, 0
v_readlane_b32 s23, v1, 1
v_readlane_b32 s24, v1, 16
s_set_vgpr_msb 0x8241
s_nop 0
v_readlane_b32 s19, v1, 0
v_readlane_b32 s20, v1, 1
v_readlane_b32 s21, v1, 16
s_set_vgpr_msb 0x4100
s_nop 0
v_readlane_b32 s16, v1, 0
v_readlane_b32 s17, v1, 1
v_readlane_b32 s18, v1, 16
s_mov_b32 s8, 0x53545250
s_mov_b32 s9, 1
s_mov_b32 s10, 0x85
s_mov_b32 s11, 0
s_mov_b32 s40, 0x434f4d50
s_set_vgpr_msb 0x0085
s_nop 0
s_wait_idle
s_getreg_b32 s14, hwreg(HW_REG_WAVE_STATUS)
s_mov_b32 s38, exec_lo
s_getreg_b32 s12, hwreg(HW_REG_WAVE_MODE)
s_get_pc_i64 s[34:35]
.Lpcnext_sentinel_85_0:
s_add_co_u32 s34, s34, (.Lreturn_sentinel_85_0 - .Lpcnext_sentinel_85_0)
s_add_co_ci_u32 s35, s35, 0
s_nop 0
s_nop 0
.Lreturn_sentinel_85_0:
s_getreg_b32 s13, hwreg(HW_REG_WAVE_MODE)
s_mov_b32 s36, ttmp0
s_mov_b32 s37, ttmp1
s_mov_b32 s39, exec_lo
s_wait_idle
s_getreg_b32 s15, hwreg(HW_REG_WAVE_STATUS)
s_set_vgpr_msb 0x8500
s_nop 0
v_readlane_b32 s25, v1, 0
v_readlane_b32 s26, v1, 1
v_readlane_b32 s27, v1, 16
s_set_vgpr_msb 0x0041
s_nop 0
v_readlane_b32 s28, v1, 0
v_readlane_b32 s29, v1, 1
v_readlane_b32 s30, v1, 16
s_set_vgpr_msb 0x4182
s_nop 0
v_readlane_b32 s31, v1, 0
v_readlane_b32 s32, v1, 1
v_readlane_b32 s33, v1, 16
s_set_vgpr_msb 0x8200
s_nop 0
s_mov_b32 exec_lo, 1
v_mov_b32 v0, 0
v_mov_b32 v2, s8
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:0
s_wait_storecnt 0
v_mov_b32 v2, s9
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:4
s_wait_storecnt 0
v_mov_b32 v2, s10
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:8
s_wait_storecnt 0
v_mov_b32 v2, s11
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:12
s_wait_storecnt 0
v_mov_b32 v2, s12
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:16
s_wait_storecnt 0
v_mov_b32 v2, s13
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:20
s_wait_storecnt 0
v_mov_b32 v2, s14
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:24
s_wait_storecnt 0
v_mov_b32 v2, s15
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:28
s_wait_storecnt 0
v_mov_b32 v2, s16
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:32
s_wait_storecnt 0
v_mov_b32 v2, s17
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:36
s_wait_storecnt 0
v_mov_b32 v2, s18
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:40
s_wait_storecnt 0
v_mov_b32 v2, s19
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:44
s_wait_storecnt 0
v_mov_b32 v2, s20
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:48
s_wait_storecnt 0
v_mov_b32 v2, s21
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:52
s_wait_storecnt 0
v_mov_b32 v2, s22
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:56
s_wait_storecnt 0
v_mov_b32 v2, s23
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:60
s_wait_storecnt 0
v_mov_b32 v2, s24
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:64
s_wait_storecnt 0
v_mov_b32 v2, s25
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:68
s_wait_storecnt 0
v_mov_b32 v2, s26
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:72
s_wait_storecnt 0
v_mov_b32 v2, s27
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:76
s_wait_storecnt 0
v_mov_b32 v2, s28
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:80
s_wait_storecnt 0
v_mov_b32 v2, s29
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:84
s_wait_storecnt 0
v_mov_b32 v2, s30
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:88
s_wait_storecnt 0
v_mov_b32 v2, s31
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:92
s_wait_storecnt 0
v_mov_b32 v2, s32
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:96
s_wait_storecnt 0
v_mov_b32 v2, s33
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:100
s_wait_storecnt 0
v_mov_b32 v2, s34
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:104
s_wait_storecnt 0
v_mov_b32 v2, s35
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:108
s_wait_storecnt 0
v_mov_b32 v2, s36
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:112
s_wait_storecnt 0
v_mov_b32 v2, s37
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:116
s_wait_storecnt 0
v_mov_b32 v2, s38
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:120
s_wait_storecnt 0
v_mov_b32 v2, s39
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:124
s_wait_storecnt 0
v_mov_b32 v2, s40
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:128
s_wait_storecnt 0
s_wait_storecnt 0
s_endpgm
.Lend_sentinel_85_0:
.size sentinel_85_0, .Lend_sentinel_85_0-sentinel_85_0
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel sentinel_85_0
.amdhsa_group_segment_fixed_size 0
.amdhsa_private_segment_fixed_size 0
.amdhsa_kernarg_size 8
.amdhsa_user_sgpr_count 2
.amdhsa_user_sgpr_dispatch_ptr 0
.amdhsa_user_sgpr_queue_ptr 0
.amdhsa_user_sgpr_kernarg_segment_ptr 1
.amdhsa_user_sgpr_dispatch_id 0
.amdhsa_user_sgpr_kernarg_preload_length 0
.amdhsa_user_sgpr_kernarg_preload_offset 0
.amdhsa_user_sgpr_private_segment_size 0
.amdhsa_wavefront_size32 1
.amdhsa_uses_dynamic_stack 0
.amdhsa_enable_private_segment 0
.amdhsa_system_sgpr_workgroup_id_x 0
.amdhsa_system_sgpr_workgroup_id_y 0
.amdhsa_system_sgpr_workgroup_id_z 0
.amdhsa_system_sgpr_workgroup_info 0
.amdhsa_system_vgpr_workitem_id 0
.amdhsa_next_free_vgpr 516
.amdhsa_next_free_sgpr 42
.amdhsa_named_barrier_count 0
.amdhsa_reserve_vcc 0
.amdhsa_float_round_mode_32 0
.amdhsa_float_round_mode_16_64 0
.amdhsa_float_denorm_mode_32 3
.amdhsa_float_denorm_mode_16_64 3
.amdhsa_fp16_overflow 0
.amdhsa_memory_ordered 1
.amdhsa_forward_progress 1
.amdhsa_round_robin_scheduling 0
.amdhsa_exception_fp_ieee_invalid_op 0
.amdhsa_exception_fp_denorm_src 0
.amdhsa_exception_fp_ieee_div_zero 0
.amdhsa_exception_fp_ieee_overflow 0
.amdhsa_exception_fp_ieee_underflow 0
.amdhsa_exception_fp_ieee_inexact 0
.amdhsa_exception_int_div_zero 0
.end_amdhsa_kernel
.text
.protected sentinel_85_1
.globl sentinel_85_1
.p2align 8
.type sentinel_85_1,@function
sentinel_85_1:
global_prefetch_b8 v0, s[0:1] scope:SCOPE_SE
v_nop
s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
s_load_b64 s[4:5], s[0:1], 0x0 nv
s_wait_kmcnt 0
s_mov_b32 s6, 0
s_setreg_b32 hwreg(HW_REG_WAVE_MODE, 12, 8), s6
s_nop 0
s_set_vgpr_msb 0x0000
s_nop 0
v_mov_b32 v1, 0x11000000
s_wait_idle
s_mov_b32 s7, 0x11000001
v_writelane_b32 v1, s7, 1
s_wait_idle
s_mov_b32 s7, 0x11000010
v_writelane_b32 v1, s7, 16
s_wait_idle
s_set_vgpr_msb 0x0041
s_nop 0
v_mov_b32 v1, 0x22000000
s_wait_idle
s_mov_b32 s7, 0x22000001
v_writelane_b32 v1, s7, 1
s_wait_idle
s_mov_b32 s7, 0x22000010
v_writelane_b32 v1, s7, 16
s_wait_idle
s_set_vgpr_msb 0x4182
s_nop 0
v_mov_b32 v1, 0x33000000
s_wait_idle
s_mov_b32 s7, 0x33000001
v_writelane_b32 v1, s7, 1
s_wait_idle
s_mov_b32 s7, 0x33000010
v_writelane_b32 v1, s7, 16
s_wait_idle
v_readlane_b32 s22, v1, 0
v_readlane_b32 s23, v1, 1
v_readlane_b32 s24, v1, 16
s_set_vgpr_msb 0x8241
s_nop 0
v_readlane_b32 s19, v1, 0
v_readlane_b32 s20, v1, 1
v_readlane_b32 s21, v1, 16
s_set_vgpr_msb 0x4100
s_nop 0
v_readlane_b32 s16, v1, 0
v_readlane_b32 s17, v1, 1
v_readlane_b32 s18, v1, 16
s_mov_b32 s8, 0x53545250
s_mov_b32 s9, 1
s_mov_b32 s10, 0x85
s_mov_b32 s11, 1
s_mov_b32 s40, 0x434f4d50
s_set_vgpr_msb 0x0085
s_nop 0
s_wait_idle
s_getreg_b32 s14, hwreg(HW_REG_WAVE_STATUS)
s_mov_b32 s38, exec_lo
s_getreg_b32 s12, hwreg(HW_REG_WAVE_MODE)
s_get_pc_i64 s[34:35]
.Lpcnext_sentinel_85_1:
s_add_co_u32 s34, s34, (.Lreturn_sentinel_85_1 - .Lpcnext_sentinel_85_1)
s_add_co_ci_u32 s35, s35, 0
s_nop 0
s_trap 3
.Lreturn_sentinel_85_1:
s_getreg_b32 s13, hwreg(HW_REG_WAVE_MODE)
s_mov_b32 s36, ttmp0
s_mov_b32 s37, ttmp1
s_mov_b32 s39, exec_lo
s_wait_idle
s_getreg_b32 s15, hwreg(HW_REG_WAVE_STATUS)
s_set_vgpr_msb 0x8500
s_nop 0
v_readlane_b32 s25, v1, 0
v_readlane_b32 s26, v1, 1
v_readlane_b32 s27, v1, 16
s_set_vgpr_msb 0x0041
s_nop 0
v_readlane_b32 s28, v1, 0
v_readlane_b32 s29, v1, 1
v_readlane_b32 s30, v1, 16
s_set_vgpr_msb 0x4182
s_nop 0
v_readlane_b32 s31, v1, 0
v_readlane_b32 s32, v1, 1
v_readlane_b32 s33, v1, 16
s_set_vgpr_msb 0x8200
s_nop 0
s_mov_b32 exec_lo, 1
v_mov_b32 v0, 0
v_mov_b32 v2, s8
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:0
s_wait_storecnt 0
v_mov_b32 v2, s9
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:4
s_wait_storecnt 0
v_mov_b32 v2, s10
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:8
s_wait_storecnt 0
v_mov_b32 v2, s11
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:12
s_wait_storecnt 0
v_mov_b32 v2, s12
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:16
s_wait_storecnt 0
v_mov_b32 v2, s13
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:20
s_wait_storecnt 0
v_mov_b32 v2, s14
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:24
s_wait_storecnt 0
v_mov_b32 v2, s15
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:28
s_wait_storecnt 0
v_mov_b32 v2, s16
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:32
s_wait_storecnt 0
v_mov_b32 v2, s17
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:36
s_wait_storecnt 0
v_mov_b32 v2, s18
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:40
s_wait_storecnt 0
v_mov_b32 v2, s19
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:44
s_wait_storecnt 0
v_mov_b32 v2, s20
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:48
s_wait_storecnt 0
v_mov_b32 v2, s21
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:52
s_wait_storecnt 0
v_mov_b32 v2, s22
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:56
s_wait_storecnt 0
v_mov_b32 v2, s23
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:60
s_wait_storecnt 0
v_mov_b32 v2, s24
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:64
s_wait_storecnt 0
v_mov_b32 v2, s25
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:68
s_wait_storecnt 0
v_mov_b32 v2, s26
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:72
s_wait_storecnt 0
v_mov_b32 v2, s27
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:76
s_wait_storecnt 0
v_mov_b32 v2, s28
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:80
s_wait_storecnt 0
v_mov_b32 v2, s29
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:84
s_wait_storecnt 0
v_mov_b32 v2, s30
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:88
s_wait_storecnt 0
v_mov_b32 v2, s31
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:92
s_wait_storecnt 0
v_mov_b32 v2, s32
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:96
s_wait_storecnt 0
v_mov_b32 v2, s33
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:100
s_wait_storecnt 0
v_mov_b32 v2, s34
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:104
s_wait_storecnt 0
v_mov_b32 v2, s35
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:108
s_wait_storecnt 0
v_mov_b32 v2, s36
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:112
s_wait_storecnt 0
v_mov_b32 v2, s37
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:116
s_wait_storecnt 0
v_mov_b32 v2, s38
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:120
s_wait_storecnt 0
v_mov_b32 v2, s39
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:124
s_wait_storecnt 0
v_mov_b32 v2, s40
s_wait_idle
global_store_b32 v0, v2, s[4:5] offset:128
s_wait_storecnt 0
s_wait_storecnt 0
s_endpgm
.Lend_sentinel_85_1:
.size sentinel_85_1, .Lend_sentinel_85_1-sentinel_85_1
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel sentinel_85_1
.amdhsa_group_segment_fixed_size 0
.amdhsa_private_segment_fixed_size 0
.amdhsa_kernarg_size 8
.amdhsa_user_sgpr_count 2
.amdhsa_user_sgpr_dispatch_ptr 0
.amdhsa_user_sgpr_queue_ptr 0
.amdhsa_user_sgpr_kernarg_segment_ptr 1
.amdhsa_user_sgpr_dispatch_id 0
.amdhsa_user_sgpr_kernarg_preload_length 0
.amdhsa_user_sgpr_kernarg_preload_offset 0
.amdhsa_user_sgpr_private_segment_size 0
.amdhsa_wavefront_size32 1
.amdhsa_uses_dynamic_stack 0
.amdhsa_enable_private_segment 0
.amdhsa_system_sgpr_workgroup_id_x 0
.amdhsa_system_sgpr_workgroup_id_y 0
.amdhsa_system_sgpr_workgroup_id_z 0
.amdhsa_system_sgpr_workgroup_info 0
.amdhsa_system_vgpr_workitem_id 0
.amdhsa_next_free_vgpr 516
.amdhsa_next_free_sgpr 42
.amdhsa_named_barrier_count 0
.amdhsa_reserve_vcc 0
.amdhsa_float_round_mode_32 0
.amdhsa_float_round_mode_16_64 0
.amdhsa_float_denorm_mode_32 3
.amdhsa_float_denorm_mode_16_64 3
.amdhsa_fp16_overflow 0
.amdhsa_memory_ordered 1
.amdhsa_forward_progress 1
.amdhsa_round_robin_scheduling 0
.amdhsa_exception_fp_ieee_invalid_op 0
.amdhsa_exception_fp_denorm_src 0
.amdhsa_exception_fp_ieee_div_zero 0
.amdhsa_exception_fp_ieee_overflow 0
.amdhsa_exception_fp_ieee_underflow 0
.amdhsa_exception_fp_ieee_inexact 0
.amdhsa_exception_int_div_zero 0
.end_amdhsa_kernel
.text
.section .note.GNU-stack,"",@progbits
.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.target: amdgcn-amd-amdhsa--gfx1250
amdhsa.kernels:
  - .name: sentinel_00_0
    .symbol: sentinel_00_0.kd
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 32
    .wavefront_size: 32
    .sgpr_count: 42
    .vgpr_count: 516
    .sgpr_spill_count: 0
    .vgpr_spill_count: 0
    .uses_dynamic_stack: false
    .args:
      - .offset: 0
        .size: 8
        .value_kind: global_buffer
        .address_space: global
        .type_name: "uint*"
  - .name: sentinel_00_1
    .symbol: sentinel_00_1.kd
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 32
    .wavefront_size: 32
    .sgpr_count: 42
    .vgpr_count: 516
    .sgpr_spill_count: 0
    .vgpr_spill_count: 0
    .uses_dynamic_stack: false
    .args:
      - .offset: 0
        .size: 8
        .value_kind: global_buffer
        .address_space: global
        .type_name: "uint*"
  - .name: sentinel_41_0
    .symbol: sentinel_41_0.kd
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 32
    .wavefront_size: 32
    .sgpr_count: 42
    .vgpr_count: 516
    .sgpr_spill_count: 0
    .vgpr_spill_count: 0
    .uses_dynamic_stack: false
    .args:
      - .offset: 0
        .size: 8
        .value_kind: global_buffer
        .address_space: global
        .type_name: "uint*"
  - .name: sentinel_41_1
    .symbol: sentinel_41_1.kd
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 32
    .wavefront_size: 32
    .sgpr_count: 42
    .vgpr_count: 516
    .sgpr_spill_count: 0
    .vgpr_spill_count: 0
    .uses_dynamic_stack: false
    .args:
      - .offset: 0
        .size: 8
        .value_kind: global_buffer
        .address_space: global
        .type_name: "uint*"
  - .name: sentinel_45_0
    .symbol: sentinel_45_0.kd
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 32
    .wavefront_size: 32
    .sgpr_count: 42
    .vgpr_count: 516
    .sgpr_spill_count: 0
    .vgpr_spill_count: 0
    .uses_dynamic_stack: false
    .args:
      - .offset: 0
        .size: 8
        .value_kind: global_buffer
        .address_space: global
        .type_name: "uint*"
  - .name: sentinel_45_1
    .symbol: sentinel_45_1.kd
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 32
    .wavefront_size: 32
    .sgpr_count: 42
    .vgpr_count: 516
    .sgpr_spill_count: 0
    .vgpr_spill_count: 0
    .uses_dynamic_stack: false
    .args:
      - .offset: 0
        .size: 8
        .value_kind: global_buffer
        .address_space: global
        .type_name: "uint*"
  - .name: sentinel_82_0
    .symbol: sentinel_82_0.kd
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 32
    .wavefront_size: 32
    .sgpr_count: 42
    .vgpr_count: 516
    .sgpr_spill_count: 0
    .vgpr_spill_count: 0
    .uses_dynamic_stack: false
    .args:
      - .offset: 0
        .size: 8
        .value_kind: global_buffer
        .address_space: global
        .type_name: "uint*"
  - .name: sentinel_82_1
    .symbol: sentinel_82_1.kd
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 32
    .wavefront_size: 32
    .sgpr_count: 42
    .vgpr_count: 516
    .sgpr_spill_count: 0
    .vgpr_spill_count: 0
    .uses_dynamic_stack: false
    .args:
      - .offset: 0
        .size: 8
        .value_kind: global_buffer
        .address_space: global
        .type_name: "uint*"
  - .name: sentinel_05_0
    .symbol: sentinel_05_0.kd
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 32
    .wavefront_size: 32
    .sgpr_count: 42
    .vgpr_count: 516
    .sgpr_spill_count: 0
    .vgpr_spill_count: 0
    .uses_dynamic_stack: false
    .args:
      - .offset: 0
        .size: 8
        .value_kind: global_buffer
        .address_space: global
        .type_name: "uint*"
  - .name: sentinel_05_1
    .symbol: sentinel_05_1.kd
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 32
    .wavefront_size: 32
    .sgpr_count: 42
    .vgpr_count: 516
    .sgpr_spill_count: 0
    .vgpr_spill_count: 0
    .uses_dynamic_stack: false
    .args:
      - .offset: 0
        .size: 8
        .value_kind: global_buffer
        .address_space: global
        .type_name: "uint*"
  - .name: sentinel_81_0
    .symbol: sentinel_81_0.kd
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 32
    .wavefront_size: 32
    .sgpr_count: 42
    .vgpr_count: 516
    .sgpr_spill_count: 0
    .vgpr_spill_count: 0
    .uses_dynamic_stack: false
    .args:
      - .offset: 0
        .size: 8
        .value_kind: global_buffer
        .address_space: global
        .type_name: "uint*"
  - .name: sentinel_81_1
    .symbol: sentinel_81_1.kd
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 32
    .wavefront_size: 32
    .sgpr_count: 42
    .vgpr_count: 516
    .sgpr_spill_count: 0
    .vgpr_spill_count: 0
    .uses_dynamic_stack: false
    .args:
      - .offset: 0
        .size: 8
        .value_kind: global_buffer
        .address_space: global
        .type_name: "uint*"
  - .name: sentinel_85_0
    .symbol: sentinel_85_0.kd
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 32
    .wavefront_size: 32
    .sgpr_count: 42
    .vgpr_count: 516
    .sgpr_spill_count: 0
    .vgpr_spill_count: 0
    .uses_dynamic_stack: false
    .args:
      - .offset: 0
        .size: 8
        .value_kind: global_buffer
        .address_space: global
        .type_name: "uint*"
  - .name: sentinel_85_1
    .symbol: sentinel_85_1.kd
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 32
    .wavefront_size: 32
    .sgpr_count: 42
    .vgpr_count: 516
    .sgpr_spill_count: 0
    .vgpr_spill_count: 0
    .uses_dynamic_stack: false
    .args:
      - .offset: 0
        .size: 8
        .value_kind: global_buffer
        .address_space: global
        .type_name: "uint*"
...
.end_amdgpu_metadata
