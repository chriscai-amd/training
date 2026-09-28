.text
L_0000:
s_branch L_0008
L_0004:
s_branch L_0fcc
L_0008:
s_version UC_VERSION_GFX12|UC_VERSION_W32_BIT
L_000c:
s_getreg_b32 ttmp12, hwreg(HW_REG_WAVE_STATE_PRIV)
L_0010:
s_getreg_b32 ttmp14, hwreg(HW_REG_WAVE_EXCP_FLAG_PRIV)
L_0014:
s_bitcmp1_b32 ttmp14, 8
L_0018:
s_cbranch_scc0 L_003c
L_001c:
s_setreg_imm32_b32 hwreg(HW_REG_WAVE_EXCP_FLAG_PRIV, 8, 1), 0
L_0024:
s_mov_b64 ttmp[2:3], 0
L_0028:
v_mov_b32_e32 v1, 0
L_002c:
global_prefetch_b8 v1, ttmp[2:3] scope:SCOPE_SE
L_0038:
s_rfe_i64 ttmp[0:1]
L_003c:
s_getreg_b32 ttmp2, hwreg(HW_REG_WAVE_MODE, 12, 6)
s_mov_b32 ttmp3, 0
s_setreg_b32 hwreg(HW_REG_WAVE_MODE, 12, 6), ttmp3
s_mov_b32 ttmp14, exec_lo
s_mov_b32 exec_lo, 1
.long 0xd760007b, 0x00010101 // preserved: v_readlane_b32 ttmp15, v1, 0
v_sub_nc_u32_e64 v1, 63, ttmp2
global_prefetch_b8 v1, ttmp[2:3] offset:-63 scope:SCOPE_SE th:TH_LOAD_RT
.long 0xd7610001, 0x0001007b // preserved: v_writelane_b32 v1, ttmp15, 0
s_setreg_b32 hwreg(HW_REG_WAVE_MODE, 12, 6), ttmp2
s_mov_b32 exec_lo, ttmp14
L_006c:
s_and_not1_b32 ttmp12, ttmp12, 0x8c00
L_0074:
s_getreg_b32 ttmp15, hwreg(HW_REG_WAVE_EXCP_FLAG_PRIV)
L_0078:
s_bitcmp0_b32 ttmp15, 7
L_007c:
s_cbranch_scc1 L_0088
L_0080:
s_and_not1_b32 ttmp1, ttmp1, 0xf0000000
L_0088:
s_bitcmp1_b32 ttmp15, 5
L_008c:
s_cbranch_scc0 L_00a8
L_0090:
s_and_not1_b32 ttmp15, ttmp15, 0x480
L_0098:
s_setreg_imm32_b32 hwreg(HW_REG_WAVE_EXCP_FLAG_PRIV, 7, 1), 0
L_00a0:
s_setreg_imm32_b32 hwreg(HW_REG_WAVE_EXCP_FLAG_PRIV, 10, 1), 0
L_00a8:
s_and_b32 ttmp2, ttmp12, 0x4000
L_00b0:
s_cbranch_scc0 L_00d4
L_00b4:
s_and_b32 ttmp2, ttmp15, 0x480
L_00bc:
s_cbranch_scc1 L_0120
L_00c0:
s_and_b32 ttmp2, ttmp15, 32
L_00c4:
s_cbranch_scc1 L_0470
L_00c8:
s_sleep 16
L_00cc:
s_getreg_b32 ttmp15, hwreg(HW_REG_WAVE_EXCP_FLAG_PRIV)
L_00d0:
s_branch L_00c0
L_00d4:
s_and_b32 ttmp2, ttmp15, 0xbd0
L_00dc:
s_cbranch_scc1 L_0120
L_00e0:
s_getreg_b32 ttmp2, hwreg(HW_REG_WAVE_EXCP_FLAG_USER)
L_00e4:
s_and_b32 ttmp3, ttmp15, 15
L_00e8:
s_cbranch_scc0 L_00f4
L_00ec:
s_or_b32 ttmp2, ttmp2, 0x80
L_00f4:
s_getreg_b32 ttmp3, hwreg(HW_REG_WAVE_TRAP_CTRL)
L_00f8:
s_and_b32 ttmp2, ttmp3, ttmp2
L_00fc:
s_cbranch_scc1 L_0120
L_0100:
s_and_b32 ttmp2, ttmp1, 0xf0000000
L_0108:
s_cbranch_scc1 L_0120
L_010c:
s_and_b32 ttmp3, ttmp3, 0x200
L_0114:
s_cbranch_scc1 L_0120
L_0118:
s_and_b32 ttmp2, ttmp15, 32
L_011c:
s_cbranch_scc1 L_0470
L_0120:
s_and_not1_b32 ttmp11, ttmp11, 0x7fc000
L_0128:
s_getreg_b32 ttmp14, hwreg(HW_REG_XNACK_STATE_PRIV, 18, 1)
L_012c:
s_lshl_b32 ttmp14, ttmp14, 22
L_0130:
s_or_b32 ttmp11, ttmp11, ttmp14
L_0134:
s_getreg_b32 ttmp14, hwreg(HW_REG_XNACK_STATE_PRIV, 16, 1)
L_0138:
s_lshl_b32 ttmp14, ttmp14, 21
L_013c:
s_or_b32 ttmp11, ttmp11, ttmp14
L_0140:
s_getreg_b32 ttmp14, hwreg(HW_REG_XNACK_STATE_PRIV, 0, 7)
L_0144:
s_lshl_b32 ttmp14, ttmp14, 14
L_0148:
s_or_b32 ttmp11, ttmp11, ttmp14
L_014c:
s_setreg_imm32_b32 hwreg(HW_REG_XNACK_STATE_PRIV), 0
L_0154:
s_mov_b32 ttmp3, ttmp15
L_0158:
s_sendmsg_rtn_b64 ttmp[14:15], sendmsg(MSG_RTN_GET_TMA)
L_015c:
s_wait_idle
L_0160:
s_lshl_b64 ttmp[14:15], ttmp[14:15], 8
L_0164:
s_bitcmp1_b32 ttmp15, 24
L_0168:
s_cbranch_scc0 L_0174
L_016c:
s_or_b32 ttmp15, ttmp15, 0xfe000000
L_0174:
s_load_b32 ttmp2, ttmp[14:15], 0x10 scope:SCOPE_SYS
L_017c:
s_wait_idle
L_0180:
s_lshl_b32 ttmp2, ttmp2, 23
L_0184:
s_and_not1_b32 ttmp11, ttmp11, 0x3800000
L_018c:
s_or_b32 ttmp11, ttmp11, ttmp2
L_0190:
s_and_b32 ttmp2, ttmp3, 0x480
L_0198:
s_cbranch_scc0 L_01cc
L_019c:
s_bitcmp1_b32 ttmp12, 14
L_01a0:
s_cbranch_scc1 L_01ac
L_01a4:
s_bitcmp1_b32 ttmp11, 24
L_01a8:
s_cbranch_scc1 L_01b4
L_01ac:
s_setreg_imm32_b32 hwreg(HW_REG_WAVE_EXCP_FLAG_PRIV, 7, 1), 0
L_01b4:
s_bitcmp1_b32 ttmp12, 14
L_01b8:
s_cbranch_scc1 L_01c4
L_01bc:
s_bitcmp1_b32 ttmp11, 25
L_01c0:
s_cbranch_scc1 L_01cc
L_01c4:
s_setreg_imm32_b32 hwreg(HW_REG_WAVE_EXCP_FLAG_PRIV, 10, 1), 0
L_01cc:
s_load_b64 ttmp[2:3], ttmp[14:15], 0x0 scope:SCOPE_SYS
L_01d4:
s_wait_idle
L_01d8:
s_load_b64 ttmp[14:15], ttmp[14:15], 0x8 scope:SCOPE_SYS
L_01e0:
s_wait_idle
L_01e4:
s_and_b64 ttmp[2:3], ttmp[2:3], ttmp[2:3]
L_01e8:
s_cbranch_scc0 L_01f0
L_01ec:
s_set_pc_i64 ttmp[2:3]
L_01f0:
s_and_b32 ttmp2, ttmp1, 0xf0000000
L_01f8:
s_cbranch_scc1 L_0220
L_01fc:
s_getreg_b32 ttmp2, hwreg(HW_REG_WAVE_EXCP_FLAG_PRIV)
L_0200:
s_and_b32 ttmp2, ttmp2, 0x80
L_0208:
s_cbranch_scc1 L_0228
L_020c:
s_or_b32 ttmp12, ttmp12, 0x4000
L_0214:
s_sub_co_u32 ttmp0, ttmp0, 8
L_0218:
s_sub_co_ci_u32 ttmp1, ttmp1, 0
L_021c:
s_branch L_0228
L_0220:
s_add_co_u32 ttmp0, ttmp0, 4
L_0224:
s_add_co_ci_u32 ttmp1, ttmp1, 0
L_0228:
s_and_b32 ttmp1, ttmp1, 0x1ffffff
L_0230:
s_getreg_b32 ttmp15, hwreg(HW_REG_WAVE_EXCP_FLAG_PRIV)
L_0234:
s_bitcmp1_b32 ttmp15, 4
L_0238:
s_cbranch_scc1 L_0440
L_023c:
s_load_b64 ttmp[14:15], ttmp[0:1], 0x0
L_0244:
s_wait_kmcnt 0x0
L_0248:
s_load_b64 ttmp[2:3], ttmp[0:1], 0x8
L_0250:
s_and_b32 ttmp10, ttmp14, 0x80000000
L_0258:
s_cbranch_scc1 L_02f8
L_025c:
s_bfe_u32 ttmp10, ttmp14, 0x60019
L_0264:
s_sub_co_i32 ttmp13, ttmp10, 35
L_0268:
s_cmp_le_u32 ttmp13, 1
L_026c:
s_cbranch_scc1 L_0428
L_0270:
s_sub_co_i32 ttmp13, ttmp10, 44
L_0274:
s_cmp_le_u32 ttmp13, 1
L_0278:
s_cbranch_scc1 L_041c
L_027c:
s_sub_co_i32 ttmp13, ttmp10, 55
L_0280:
s_cmp_le_u32 ttmp13, 1
L_0284:
s_cbranch_scc1 L_041c
L_0288:
s_and_b32 ttmp10, ttmp14, 0x1ff
L_0290:
s_cmp_eq_u32 ttmp10, 0xfe
L_0298:
s_cbranch_scc1 L_0428
L_029c:
s_cmp_eq_u32 ttmp10, 0xff
L_02a4:
s_cbranch_scc1 L_041c
L_02a8:
s_cmp_eq_u32 ttmp10, 0xfa
L_02b0:
s_cbranch_scc1 L_041c
L_02b4:
s_sub_co_i32 ttmp13, ttmp10, 0xe9
L_02bc:
s_cmp_le_u32 ttmp13, 1
L_02c0:
s_cbranch_scc1 L_041c
L_02c4:
s_and_b32 ttmp10, ttmp15, 0xffff0000
L_02cc:
s_cmp_eq_u32 ttmp10, 0xbf860000
L_02d4:
s_cbranch_scc0 L_0440
L_02d8:
s_bfe_u32 ttmp10, ttmp15, 0x2000e
L_02e0:
s_and_b32 ttmp13, ttmp15, 0x3f00
L_02e8:
s_lshr_b32 ttmp13, ttmp13, 6
L_02ec:
s_or_b32 ttmp10, ttmp10, ttmp13
L_02f0:
s_setreg_b32 hwreg(HW_REG_WAVE_MODE, 12, 8), ttmp10
L_02f4:
s_branch L_0440
L_02f8:
s_and_b32 ttmp10, ttmp14, 0xfc000000
L_0300:
s_cmp_eq_u32 ttmp10, 0xd4000000
L_0308:
s_cbranch_scc1 L_0370
L_030c:
s_cmp_eq_u32 ttmp10, 0xc8000000
L_0314:
s_cbranch_scc1 L_03cc
L_0318:
s_and_b32 ttmp10, ttmp14, 0xff000000
L_0320:
s_cmp_eq_u32 ttmp10, 0xcf000000
L_0328:
s_cbranch_scc1 L_0428
L_032c:
s_and_b32 ttmp13, ttmp14, 0xffff0000
L_0334:
s_cmp_eq_u32 ttmp13, 0xcc330000
L_033c:
s_cbranch_scc1 L_0434
L_0340:
s_cmp_eq_u32 ttmp13, 0xcc880000
L_0348:
s_cbranch_scc1 L_0434
L_034c:
s_cmp_eq_u32 ttmp13, 0xcc350000
L_0354:
s_cbranch_scc1 L_0440
L_0358:
s_cmp_eq_u32 ttmp13, 0xcc3a0000
L_0360:
s_cbranch_scc1 L_0440
L_0364:
s_cmp_eq_u32 ttmp10, 0xcc000000
L_036c:
s_cbranch_scc0 L_0440
L_0370:
s_and_b32 ttmp10, ttmp15, 0x1ff
L_0378:
s_cmp_eq_u32 ttmp10, 0xff
L_0380:
s_cbranch_scc1 L_0428
L_0384:
s_cmp_eq_u32 ttmp10, 0xfa
L_038c:
s_cbranch_scc1 L_0428
L_0390:
s_sub_co_i32 ttmp10, ttmp10, 0xe9
L_0398:
s_cmp_le_u32 ttmp10, 1
L_039c:
s_cbranch_scc1 L_0428
L_03a0:
s_and_b32 ttmp10, ttmp15, 0x3fe00
L_03a8:
s_cmp_eq_u32 ttmp10, 0x1fe00
L_03b0:
s_cbranch_scc1 L_0428
L_03b4:
s_and_b32 ttmp10, ttmp15, 0x7fc0000
L_03bc:
s_cmp_eq_u32 ttmp10, 0x3fc0000
L_03c4:
s_cbranch_scc1 L_0428
L_03c8:
s_branch L_041c
L_03cc:
s_bfe_u32 ttmp10, ttmp14, 0x40016
L_03d4:
s_sub_co_i32 ttmp10, ttmp10, 1
L_03d8:
s_cmp_le_u32 ttmp10, 1
L_03dc:
s_cbranch_scc1 L_0428
L_03e0:
s_bfe_u32 ttmp10, ttmp14, 0x50011
L_03e8:
s_sub_co_i32 ttmp10, ttmp10, 1
L_03ec:
s_cmp_le_u32 ttmp10, 1
L_03f0:
s_cbranch_scc1 L_0428
L_03f4:
s_and_b32 ttmp10, ttmp14, 0x1ff
L_03fc:
s_cmp_eq_u32 ttmp10, 0xff
L_0404:
s_cbranch_scc1 L_0428
L_0408:
s_and_b32 ttmp10, ttmp15, 0x1ff
L_0410:
s_cmp_eq_u32 ttmp10, 0xff
L_0418:
s_cbranch_scc1 L_0428
L_041c:
s_wait_kmcnt 0x0
L_0420:
s_mov_b32 ttmp15, ttmp2
L_0424:
s_branch L_02c4
L_0428:
s_wait_kmcnt 0x0
L_042c:
s_mov_b32 ttmp15, ttmp3
L_0430:
s_branch L_02c4
L_0434:
s_sub_co_u32 ttmp0, ttmp0, 8
L_0438:
s_sub_co_ci_u32 ttmp1, ttmp1, 0
L_043c:
s_branch L_041c
L_0440:
s_wait_kmcnt 0x0
L_0444:
s_lshr_b32 ttmp14, ttmp11, 22
L_0448:
s_setreg_b32 hwreg(HW_REG_XNACK_STATE_PRIV, 18, 1), ttmp14
L_044c:
s_lshr_b32 ttmp14, ttmp11, 21
L_0450:
s_setreg_b32 hwreg(HW_REG_XNACK_STATE_PRIV, 16, 1), ttmp14
L_0454:
s_lshr_b32 ttmp14, ttmp11, 14
L_0458:
s_setreg_b32 hwreg(HW_REG_XNACK_STATE_PRIV, 0, 7), ttmp14
L_045c:
s_and_b64 exec, exec, exec
L_0460:
s_and_b64 vcc, vcc, vcc
L_0464:
s_lshr_b32 ttmp12, ttmp12, 9
L_0468:
s_setreg_b32 hwreg(HW_REG_WAVE_STATE_PRIV, 9, 7), ttmp12
L_046c:
s_rfe_i64 ttmp[0:1]
L_0470:
s_getreg_b32 ttmp14, hwreg(HW_REG_WAVE_STATUS)
L_0474:
s_bitcmp1_b32 ttmp14, 24
L_0478:
s_cbranch_scc0 L_0480
L_047c:
s_endpgm
L_0480:
s_and_b32 ttmp1, ttmp1, 0x1ffffff
L_0488:
s_mov_b32 ttmp14, 0
L_048c:
s_setreg_b32 hwreg(HW_REG_WAVE_EXCP_FLAG_PRIV, 5, 1), ttmp14
L_0490:
s_and_not1_b32 ttmp11, ttmp11, 0x7fc000
L_0498:
s_getreg_b32 ttmp14, hwreg(HW_REG_XNACK_STATE_PRIV, 18, 1)
L_049c:
s_lshl_b32 ttmp14, ttmp14, 22
L_04a0:
s_or_b32 ttmp11, ttmp11, ttmp14
L_04a4:
s_getreg_b32 ttmp14, hwreg(HW_REG_XNACK_STATE_PRIV, 16, 1)
L_04a8:
s_lshl_b32 ttmp14, ttmp14, 21
L_04ac:
s_or_b32 ttmp11, ttmp11, ttmp14
L_04b0:
s_getreg_b32 ttmp14, hwreg(HW_REG_XNACK_STATE_PRIV, 0, 7)
L_04b4:
s_lshl_b32 ttmp14, ttmp14, 14
L_04b8:
s_or_b32 ttmp11, ttmp11, ttmp14
L_04bc:
s_setreg_imm32_b32 hwreg(HW_REG_XNACK_STATE_PRIV), 0
L_04c4:
s_bitcmp1_b32 ttmp15, 4
L_04c8:
s_cbranch_scc1 L_06d0
L_04cc:
s_load_b64 ttmp[14:15], ttmp[0:1], 0x0
L_04d4:
s_wait_kmcnt 0x0
L_04d8:
s_load_b64 ttmp[2:3], ttmp[0:1], 0x8
L_04e0:
s_and_b32 ttmp10, ttmp14, 0x80000000
L_04e8:
s_cbranch_scc1 L_0588
L_04ec:
s_bfe_u32 ttmp10, ttmp14, 0x60019
L_04f4:
s_sub_co_i32 ttmp13, ttmp10, 35
L_04f8:
s_cmp_le_u32 ttmp13, 1
L_04fc:
s_cbranch_scc1 L_06b8
L_0500:
s_sub_co_i32 ttmp13, ttmp10, 44
L_0504:
s_cmp_le_u32 ttmp13, 1
L_0508:
s_cbranch_scc1 L_06ac
L_050c:
s_sub_co_i32 ttmp13, ttmp10, 55
L_0510:
s_cmp_le_u32 ttmp13, 1
L_0514:
s_cbranch_scc1 L_06ac
L_0518:
s_and_b32 ttmp10, ttmp14, 0x1ff
L_0520:
s_cmp_eq_u32 ttmp10, 0xfe
L_0528:
s_cbranch_scc1 L_06b8
L_052c:
s_cmp_eq_u32 ttmp10, 0xff
L_0534:
s_cbranch_scc1 L_06ac
L_0538:
s_cmp_eq_u32 ttmp10, 0xfa
L_0540:
s_cbranch_scc1 L_06ac
L_0544:
s_sub_co_i32 ttmp13, ttmp10, 0xe9
L_054c:
s_cmp_le_u32 ttmp13, 1
L_0550:
s_cbranch_scc1 L_06ac
L_0554:
s_and_b32 ttmp10, ttmp15, 0xffff0000
L_055c:
s_cmp_eq_u32 ttmp10, 0xbf860000
L_0564:
s_cbranch_scc0 L_06d0
L_0568:
s_bfe_u32 ttmp10, ttmp15, 0x2000e
L_0570:
s_and_b32 ttmp13, ttmp15, 0x3f00
L_0578:
s_lshr_b32 ttmp13, ttmp13, 6
L_057c:
s_or_b32 ttmp10, ttmp10, ttmp13
L_0580:
s_setreg_b32 hwreg(HW_REG_WAVE_MODE, 12, 8), ttmp10
L_0584:
s_branch L_06d0
L_0588:
s_and_b32 ttmp10, ttmp14, 0xfc000000
L_0590:
s_cmp_eq_u32 ttmp10, 0xd4000000
L_0598:
s_cbranch_scc1 L_0600
L_059c:
s_cmp_eq_u32 ttmp10, 0xc8000000
L_05a4:
s_cbranch_scc1 L_065c
L_05a8:
s_and_b32 ttmp10, ttmp14, 0xff000000
L_05b0:
s_cmp_eq_u32 ttmp10, 0xcf000000
L_05b8:
s_cbranch_scc1 L_06b8
L_05bc:
s_and_b32 ttmp13, ttmp14, 0xffff0000
L_05c4:
s_cmp_eq_u32 ttmp13, 0xcc330000
L_05cc:
s_cbranch_scc1 L_06c4
L_05d0:
s_cmp_eq_u32 ttmp13, 0xcc880000
L_05d8:
s_cbranch_scc1 L_06c4
L_05dc:
s_cmp_eq_u32 ttmp13, 0xcc350000
L_05e4:
s_cbranch_scc1 L_06d0
L_05e8:
s_cmp_eq_u32 ttmp13, 0xcc3a0000
L_05f0:
s_cbranch_scc1 L_06d0
L_05f4:
s_cmp_eq_u32 ttmp10, 0xcc000000
L_05fc:
s_cbranch_scc0 L_06d0
L_0600:
s_and_b32 ttmp10, ttmp15, 0x1ff
L_0608:
s_cmp_eq_u32 ttmp10, 0xff
L_0610:
s_cbranch_scc1 L_06b8
L_0614:
s_cmp_eq_u32 ttmp10, 0xfa
L_061c:
s_cbranch_scc1 L_06b8
L_0620:
s_sub_co_i32 ttmp10, ttmp10, 0xe9
L_0628:
s_cmp_le_u32 ttmp10, 1
L_062c:
s_cbranch_scc1 L_06b8
L_0630:
s_and_b32 ttmp10, ttmp15, 0x3fe00
L_0638:
s_cmp_eq_u32 ttmp10, 0x1fe00
L_0640:
s_cbranch_scc1 L_06b8
L_0644:
s_and_b32 ttmp10, ttmp15, 0x7fc0000
L_064c:
s_cmp_eq_u32 ttmp10, 0x3fc0000
L_0654:
s_cbranch_scc1 L_06b8
L_0658:
s_branch L_06ac
L_065c:
s_bfe_u32 ttmp10, ttmp14, 0x40016
L_0664:
s_sub_co_i32 ttmp10, ttmp10, 1
L_0668:
s_cmp_le_u32 ttmp10, 1
L_066c:
s_cbranch_scc1 L_06b8
L_0670:
s_bfe_u32 ttmp10, ttmp14, 0x50011
L_0678:
s_sub_co_i32 ttmp10, ttmp10, 1
L_067c:
s_cmp_le_u32 ttmp10, 1
L_0680:
s_cbranch_scc1 L_06b8
L_0684:
s_and_b32 ttmp10, ttmp14, 0x1ff
L_068c:
s_cmp_eq_u32 ttmp10, 0xff
L_0694:
s_cbranch_scc1 L_06b8
L_0698:
s_and_b32 ttmp10, ttmp15, 0x1ff
L_06a0:
s_cmp_eq_u32 ttmp10, 0xff
L_06a8:
s_cbranch_scc1 L_06b8
L_06ac:
s_wait_kmcnt 0x0
L_06b0:
s_mov_b32 ttmp15, ttmp2
L_06b4:
s_branch L_0554
L_06b8:
s_wait_kmcnt 0x0
L_06bc:
s_mov_b32 ttmp15, ttmp3
L_06c0:
s_branch L_0554
L_06c4:
s_sub_co_u32 ttmp0, ttmp0, 8
L_06c8:
s_sub_co_ci_u32 ttmp1, ttmp1, 0
L_06cc:
s_branch L_06ac
L_06d0:
s_wait_kmcnt 0x0
L_06d4:
s_mov_b32 ttmp2, exec_lo
L_06d8:
s_mov_b32 ttmp3, exec_hi
L_06dc:
s_mov_b64 exec, 0
L_06e0:
s_sendmsg_rtn_b64 exec, sendmsg(MSG_RTN_SAVE_WAVE)
L_06e4:
s_wait_idle
L_06e8:
s_and_b32 ttmp14, exec_hi, 0x4000000
L_06f0:
s_lshl_b32 ttmp14, ttmp14, 5
L_06f4:
s_or_b32 ttmp1, ttmp1, ttmp14
L_06f8:
s_getreg_b32 ttmp3, hwreg(HW_REG_XNACK_MASK)
L_06fc:
s_setreg_imm32_b32 hwreg(HW_REG_XNACK_MASK), 0
L_0704:
s_getreg_b32 ttmp14, hwreg(HW_REG_WAVE_MODE, 12, 6)
L_0708:
s_lshl_b32 ttmp14, ttmp14, 25
L_070c:
s_or_b32 ttmp1, ttmp1, ttmp14
L_0710:
s_mov_b32 ttmp14, 0
L_0714:
s_setreg_b32 hwreg(HW_REG_WAVE_MODE, 12, 6), ttmp14
L_0718:
s_mov_b32 ttmp14, exec_lo
L_071c:
s_and_b32 ttmp15, exec_hi, 0x1ffffff
L_0724:
s_mov_b32 exec_lo, -1
L_0728:
s_mov_b32 exec_hi, -1
L_072c:
global_store_addtid_b32 v0, ttmp[14:15] scope:SCOPE_SYS
L_0738:
v_mov_b32_e32 v0, 0
L_073c:
s_mov_b32 exec_lo, ttmp14
L_0740:
s_mov_b32 exec_hi, ttmp15
L_0744:
s_getreg_b32 ttmp15, hwreg(HW_REG_WAVE_STATUS, 29, 1)
L_0748:
s_lshl_b32 ttmp15, ttmp15, 25
L_074c:
s_getreg_b32 ttmp14, hwreg(HW_REG_WAVE_GPR_ALLOC, 12, 8)
L_0750:
s_add_co_u32 ttmp14, ttmp14, 1
L_0754:
s_bitcmp1_b32 ttmp15, 25
L_0758:
s_cbranch_scc1 L_0764
L_075c:
s_lshl_b32 ttmp14, ttmp14, 9
L_0760:
s_branch L_0768
L_0764:
s_lshl_b32 ttmp14, ttmp14, 10
L_0768:
s_and_b32 ttmp15, exec_hi, 0x1ffffff
L_0770:
s_add_co_u32 ttmp14, ttmp14, 0x1c0
L_0778:
s_add_co_u32 ttmp14, ttmp14, exec_lo
L_077c:
s_add_co_ci_u32 ttmp15, ttmp15, 0
L_0780:
.long 0xd7610000, 0x00010870 // preserved: v_writelane_b32 v0, ttmp4, 4
L_0788:
.long 0xd7610000, 0x00010a71 // preserved: v_writelane_b32 v0, ttmp5, 5
L_0790:
.long 0xd7610000, 0x00010c72 // preserved: v_writelane_b32 v0, ttmp6, 6
L_0798:
.long 0xd7610000, 0x00010e73 // preserved: v_writelane_b32 v0, ttmp7, 7
L_07a0:
.long 0xd7610000, 0x00011074 // preserved: v_writelane_b32 v0, ttmp8, 8
L_07a8:
.long 0xd7610000, 0x00011275 // preserved: v_writelane_b32 v0, ttmp9, 9
L_07b0:
.long 0xd7610000, 0x00011476 // preserved: v_writelane_b32 v0, ttmp10, 10
L_07b8:
.long 0xd7610000, 0x00011677 // preserved: v_writelane_b32 v0, ttmp11, 11
L_07c0:
.long 0xd7610000, 0x00011a79 // preserved: v_writelane_b32 v0, ttmp13, 13
L_07c8:
.long 0xd7610000, 0x00011c7e // preserved: v_writelane_b32 v0, exec_lo, 14
L_07d0:
.long 0xd7610000, 0x00011e7f // preserved: v_writelane_b32 v0, exec_hi, 15
L_07d8:
s_mov_b32 exec_lo, 0x3fff
L_07e0:
s_mov_b32 exec_hi, 0
L_07e4:
global_store_addtid_b32 v0, ttmp[14:15] scope:SCOPE_SYS
L_07f0:
.long 0xd760007a, 0x00011d00 // preserved: v_readlane_b32 ttmp14, v0, 14
L_07f8:
.long 0xd760007b, 0x00011f00 // preserved: v_readlane_b32 ttmp15, v0, 15
L_0800:
s_mov_b32 exec_lo, ttmp14
L_0804:
s_mov_b32 exec_hi, ttmp15
L_0808:
s_mov_b32 ttmp8, exec_lo
L_080c:
s_and_b32 ttmp9, exec_hi, 0x1ffffff
L_0814:
s_mov_b32 ttmp5, m0
L_0818:
s_getreg_b32 ttmp7, hwreg(HW_REG_WAVE_STATUS, 29, 1)
L_081c:
s_lshl_b32 ttmp7, ttmp7, 25
L_0820:
s_mov_b32 exec_lo, -1
L_0824:
s_lshr_b32 m0, ttmp7, 25
L_0828:
s_and_b32 m0, m0, 1
L_082c:
s_cmp_eq_u32 m0, 1
L_0830:
s_cbranch_scc1 L_083c
L_0834:
s_mov_b32 exec_hi, 0
L_0838:
s_branch L_0844
L_083c:
s_mov_b32 exec_hi, -1
L_0840:
s_branch L_086c
L_0844:
global_store_addtid_b32 v1, ttmp[8:9] offset:128 scope:SCOPE_SYS
L_0850:
global_store_addtid_b32 v2, ttmp[8:9] offset:256 scope:SCOPE_SYS
L_085c:
global_store_addtid_b32 v3, ttmp[8:9] offset:384 scope:SCOPE_SYS
L_0868:
s_branch L_0890
L_086c:
global_store_addtid_b32 v1, ttmp[8:9] offset:256 scope:SCOPE_SYS
L_0878:
global_store_addtid_b32 v2, ttmp[8:9] offset:512 scope:SCOPE_SYS
L_0884:
global_store_addtid_b32 v3, ttmp[8:9] offset:768 scope:SCOPE_SYS
L_0890:
s_getreg_b32 ttmp4, hwreg(HW_REG_WAVE_GPR_ALLOC, 12, 8)
L_0894:
s_add_co_u32 ttmp4, ttmp4, 1
L_0898:
s_bitcmp1_b32 ttmp7, 25
L_089c:
s_cbranch_scc1 L_08a8
L_08a0:
s_lshl_b32 ttmp4, ttmp4, 9
L_08a4:
s_branch L_08ac
L_08a8:
s_lshl_b32 ttmp4, ttmp4, 10
L_08ac:
s_add_co_u32 ttmp4, ttmp4, 0x200
L_08b4:
v_mov_b32_e32 v0, 0
L_08b8:
v_mov_b32_e32 v1, 0
L_08bc:
v_mov_b32_e32 v2, 0
L_08c0:
s_mov_b32 m0, 0
L_08c4:
.long 0xd7610002, 0x0000fa71 // preserved: v_writelane_b32 v2, ttmp5, m0
L_08cc:
s_add_co_u32 m0, m0, 1
L_08d0:
s_getreg_b32 ttmp14, hwreg(HW_REG_WAVE_STATUS)
L_08d4:
s_bitcmp0_b32 ttmp14, 11
L_08d8:
s_cbranch_scc1 L_08e8
L_08dc:
s_barrier_signal_isfirst -2
L_08e0:
s_barrier_wait 0xfffe
L_08e4:
s_cbranch_scc0 L_08ec
L_08e8:
s_barrier_signal -4
L_08ec:
s_barrier_wait 0xfffc
L_08f0:
s_sendmsg_rtn_b32 ttmp14, sendmsg(MSG_RTN_GET_CLUSTER_BARRIER_STATE)
L_08f4:
s_wait_kmcnt 0x0
L_08f8:
s_bitcmp0_b32 ttmp14, 0
L_08fc:
s_cbranch_scc1 L_0918
L_0900:
s_bfe_u32 ttmp5, ttmp14, 0x70004
L_0908:
s_bfe_u32 ttmp14, ttmp14, 0x70010
L_0910:
s_cmp_eq_u32 ttmp14, ttmp5
L_0914:
s_cbranch_scc1 L_08f0
L_0918:
s_getreg_b32 ttmp14, hwreg(HW_REG_WAVE_STATE_PRIV)
L_091c:
s_and_b32 ttmp14, ttmp14, 0x1000c
L_0924:
s_and_not1_b32 ttmp12, ttmp12, 0x1000c
L_092c:
s_or_b32 ttmp12, ttmp12, ttmp14
L_0930:
.long 0xd7610002, 0x0000fa6c // preserved: v_writelane_b32 v2, ttmp0, m0
L_0938:
s_add_co_u32 m0, m0, 1
L_093c:
s_and_b32 ttmp14, ttmp1, 0x1ffffff
L_0944:
.long 0xd7610002, 0x0000fa7a // preserved: v_writelane_b32 v2, ttmp14, m0
L_094c:
s_add_co_u32 m0, m0, 1
L_0950:
.long 0xd7610002, 0x0000fa6e // preserved: v_writelane_b32 v2, ttmp2, m0
L_0958:
s_add_co_u32 m0, m0, 1
L_095c:
s_mov_b32 ttmp14, 0
L_0960:
.long 0xd7610002, 0x0000fa7a // preserved: v_writelane_b32 v2, ttmp14, m0
L_0968:
s_add_co_u32 m0, m0, 1
L_096c:
.long 0xd7610002, 0x0000fa78 // preserved: v_writelane_b32 v2, ttmp12, m0
L_0974:
s_add_co_u32 m0, m0, 1
L_0978:
s_getreg_b32 ttmp14, hwreg(HW_REG_WAVE_EXCP_FLAG_PRIV)
L_097c:
.long 0xd7610002, 0x0000fa7a // preserved: v_writelane_b32 v2, ttmp14, m0
L_0984:
s_add_co_u32 m0, m0, 1
L_0988:
.long 0xd7610002, 0x0000fa6f // preserved: v_writelane_b32 v2, ttmp3, m0
L_0990:
s_add_co_u32 m0, m0, 1
L_0994:
s_getreg_b32 ttmp5, hwreg(HW_REG_WAVE_MODE)
L_0998:
s_bfe_u32 ttmp14, ttmp1, 0x60019
L_09a0:
s_lshl_b32 ttmp14, ttmp14, 12
L_09a4:
s_or_b32 ttmp5, ttmp5, ttmp14
L_09a8:
.long 0xd7610002, 0x0000fa71 // preserved: v_writelane_b32 v2, ttmp5, m0
L_09b0:
s_add_co_u32 m0, m0, 1
L_09b4:
s_getreg_b32 ttmp5, hwreg(HW_REG_WAVE_SCRATCH_BASE_LO)
L_09b8:
.long 0xd7610002, 0x0000fa71 // preserved: v_writelane_b32 v2, ttmp5, m0
L_09c0:
s_add_co_u32 m0, m0, 1
L_09c4:
s_getreg_b32 ttmp5, hwreg(HW_REG_WAVE_SCRATCH_BASE_HI)
L_09c8:
.long 0xd7610002, 0x0000fa71 // preserved: v_writelane_b32 v2, ttmp5, m0
L_09d0:
s_add_co_u32 m0, m0, 1
L_09d4:
s_getreg_b32 ttmp5, hwreg(HW_REG_WAVE_EXCP_FLAG_USER)
L_09d8:
.long 0xd7610002, 0x0000fa71 // preserved: v_writelane_b32 v2, ttmp5, m0
L_09e0:
s_add_co_u32 m0, m0, 1
L_09e4:
s_getreg_b32 ttmp5, hwreg(HW_REG_WAVE_TRAP_CTRL)
L_09e8:
.long 0xd7610002, 0x0000fa71 // preserved: v_writelane_b32 v2, ttmp5, m0
L_09f0:
s_add_co_u32 m0, m0, 1
L_09f4:
s_getreg_b32 ttmp14, hwreg(HW_REG_WAVE_STATUS)
L_09f8:
.long 0xd7610002, 0x0000fa7a // preserved: v_writelane_b32 v2, ttmp14, m0
L_0a00:
s_add_co_u32 m0, m0, 1
L_0a04:
s_get_barrier_state ttmp14, -1
L_0a08:
s_wait_kmcnt 0x0
L_0a0c:
.long 0xd7610002, 0x0000fa7a // preserved: v_writelane_b32 v2, ttmp14, m0
L_0a14:
s_add_co_u32 m0, m0, 1
L_0a18:
s_sendmsg_rtn_b32 ttmp14, sendmsg(MSG_RTN_GET_CLUSTER_BARRIER_STATE)
L_0a1c:
s_wait_kmcnt 0x0
L_0a20:
.long 0xd7610002, 0x0000fa7a // preserved: v_writelane_b32 v2, ttmp14, m0
L_0a28:
s_add_co_u32 m0, m0, 1
L_0a2c:
s_getreg_b32 ttmp14, hwreg(HW_REG_WAVE_SCHED_MODE)
L_0a30:
.long 0xd7610002, 0x0000fa7a // preserved: v_writelane_b32 v2, ttmp14, m0
L_0a38:
s_add_co_u32 m0, m0, 1
L_0a3c:
s_mov_b32 exec_lo, -1
L_0a40:
s_mov_b32 exec_hi, 0
L_0a44:
s_add_co_u32 ttmp10, ttmp8, ttmp4
L_0a48:
s_add_co_ci_u32 ttmp11, ttmp9, 0
L_0a4c:
global_store_addtid_b32 v2, ttmp[10:11] scope:SCOPE_SYS
L_0a58:
s_mov_b32 exec_lo, -1
L_0a5c:
v_mov_b32_e32 v2, 0
L_0a60:
s_get_barrier_state ttmp14, 1
L_0a64:
s_wait_kmcnt 0x0
L_0a68:
.long 0xd7610002, 0x0001007a // preserved: v_writelane_b32 v2, ttmp14, 0
L_0a70:
s_get_barrier_state ttmp14, 2
L_0a74:
s_wait_kmcnt 0x0
L_0a78:
.long 0xd7610002, 0x0001027a // preserved: v_writelane_b32 v2, ttmp14, 1
L_0a80:
s_get_barrier_state ttmp14, 3
L_0a84:
s_wait_kmcnt 0x0
L_0a88:
.long 0xd7610002, 0x0001047a // preserved: v_writelane_b32 v2, ttmp14, 2
L_0a90:
s_get_barrier_state ttmp14, 4
L_0a94:
s_wait_kmcnt 0x0
L_0a98:
.long 0xd7610002, 0x0001067a // preserved: v_writelane_b32 v2, ttmp14, 3
L_0aa0:
s_get_barrier_state ttmp14, 5
L_0aa4:
s_wait_kmcnt 0x0
L_0aa8:
.long 0xd7610002, 0x0001087a // preserved: v_writelane_b32 v2, ttmp14, 4
L_0ab0:
s_get_barrier_state ttmp14, 6
L_0ab4:
s_wait_kmcnt 0x0
L_0ab8:
.long 0xd7610002, 0x00010a7a // preserved: v_writelane_b32 v2, ttmp14, 5
L_0ac0:
s_get_barrier_state ttmp14, 7
L_0ac4:
s_wait_kmcnt 0x0
L_0ac8:
.long 0xd7610002, 0x00010c7a // preserved: v_writelane_b32 v2, ttmp14, 6
L_0ad0:
s_get_barrier_state ttmp14, 8
L_0ad4:
s_wait_kmcnt 0x0
L_0ad8:
.long 0xd7610002, 0x00010e7a // preserved: v_writelane_b32 v2, ttmp14, 7
L_0ae0:
s_get_barrier_state ttmp14, 9
L_0ae4:
s_wait_kmcnt 0x0
L_0ae8:
.long 0xd7610002, 0x0001107a // preserved: v_writelane_b32 v2, ttmp14, 8
L_0af0:
s_get_barrier_state ttmp14, 10
L_0af4:
s_wait_kmcnt 0x0
L_0af8:
.long 0xd7610002, 0x0001127a // preserved: v_writelane_b32 v2, ttmp14, 9
L_0b00:
s_get_barrier_state ttmp14, 11
L_0b04:
s_wait_kmcnt 0x0
L_0b08:
.long 0xd7610002, 0x0001147a // preserved: v_writelane_b32 v2, ttmp14, 10
L_0b10:
s_get_barrier_state ttmp14, 12
L_0b14:
s_wait_kmcnt 0x0
L_0b18:
.long 0xd7610002, 0x0001167a // preserved: v_writelane_b32 v2, ttmp14, 11
L_0b20:
s_get_barrier_state ttmp14, 13
L_0b24:
s_wait_kmcnt 0x0
L_0b28:
.long 0xd7610002, 0x0001187a // preserved: v_writelane_b32 v2, ttmp14, 12
L_0b30:
s_get_barrier_state ttmp14, 14
L_0b34:
s_wait_kmcnt 0x0
L_0b38:
.long 0xd7610002, 0x00011a7a // preserved: v_writelane_b32 v2, ttmp14, 13
L_0b40:
s_get_barrier_state ttmp14, 15
L_0b44:
s_wait_kmcnt 0x0
L_0b48:
.long 0xd7610002, 0x00011c7a // preserved: v_writelane_b32 v2, ttmp14, 14
L_0b50:
s_get_barrier_state ttmp14, 16
L_0b54:
s_wait_kmcnt 0x0
L_0b58:
.long 0xd7610002, 0x00011e7a // preserved: v_writelane_b32 v2, ttmp14, 15
L_0b60:
global_store_addtid_b32 v2, ttmp[10:11] offset:128 scope:SCOPE_SYS
L_0b6c:
s_getreg_b32 ttmp4, hwreg(HW_REG_WAVE_GPR_ALLOC, 12, 8)
L_0b70:
s_add_co_u32 ttmp4, ttmp4, 1
L_0b74:
s_bitcmp1_b32 ttmp7, 25
L_0b78:
s_cbranch_scc1 L_0b84
L_0b7c:
s_lshl_b32 ttmp4, ttmp4, 9
L_0b80:
s_branch L_0b88
L_0b84:
s_lshl_b32 ttmp4, ttmp4, 10
L_0b88:
s_mov_b32 ttmp13, 0
L_0b8c:
s_mov_b32 m0, 0
L_0b90:
s_nop 0
L_0b94:
s_movrels_b64 s[0:1], s[0:1]
L_0b98:
s_movrels_b64 s[2:3], s[2:3]
L_0b9c:
s_movrels_b64 s[4:5], s[4:5]
L_0ba0:
s_movrels_b64 s[6:7], s[6:7]
L_0ba4:
s_movrels_b64 s[8:9], s[8:9]
L_0ba8:
s_movrels_b64 s[10:11], s[10:11]
L_0bac:
s_movrels_b64 s[12:13], s[12:13]
L_0bb0:
s_movrels_b64 s[14:15], s[14:15]
L_0bb4:
.long 0xd7610002, 0x0000f200 // preserved: v_writelane_b32 v2, s0, ttmp13
L_0bbc:
s_add_co_u32 ttmp13, ttmp13, 1
L_0bc0:
.long 0xd7610002, 0x0000f201 // preserved: v_writelane_b32 v2, s1, ttmp13
L_0bc8:
s_add_co_u32 ttmp13, ttmp13, 1
L_0bcc:
.long 0xd7610002, 0x0000f202 // preserved: v_writelane_b32 v2, s2, ttmp13
L_0bd4:
s_add_co_u32 ttmp13, ttmp13, 1
L_0bd8:
.long 0xd7610002, 0x0000f203 // preserved: v_writelane_b32 v2, s3, ttmp13
L_0be0:
s_add_co_u32 ttmp13, ttmp13, 1
L_0be4:
.long 0xd7610002, 0x0000f204 // preserved: v_writelane_b32 v2, s4, ttmp13
L_0bec:
s_add_co_u32 ttmp13, ttmp13, 1
L_0bf0:
.long 0xd7610002, 0x0000f205 // preserved: v_writelane_b32 v2, s5, ttmp13
L_0bf8:
s_add_co_u32 ttmp13, ttmp13, 1
L_0bfc:
.long 0xd7610002, 0x0000f206 // preserved: v_writelane_b32 v2, s6, ttmp13
L_0c04:
s_add_co_u32 ttmp13, ttmp13, 1
L_0c08:
.long 0xd7610002, 0x0000f207 // preserved: v_writelane_b32 v2, s7, ttmp13
L_0c10:
s_add_co_u32 ttmp13, ttmp13, 1
L_0c14:
.long 0xd7610002, 0x0000f208 // preserved: v_writelane_b32 v2, s8, ttmp13
L_0c1c:
s_add_co_u32 ttmp13, ttmp13, 1
L_0c20:
.long 0xd7610002, 0x0000f209 // preserved: v_writelane_b32 v2, s9, ttmp13
L_0c28:
s_add_co_u32 ttmp13, ttmp13, 1
L_0c2c:
.long 0xd7610002, 0x0000f20a // preserved: v_writelane_b32 v2, s10, ttmp13
L_0c34:
s_add_co_u32 ttmp13, ttmp13, 1
L_0c38:
.long 0xd7610002, 0x0000f20b // preserved: v_writelane_b32 v2, s11, ttmp13
L_0c40:
s_add_co_u32 ttmp13, ttmp13, 1
L_0c44:
.long 0xd7610002, 0x0000f20c // preserved: v_writelane_b32 v2, s12, ttmp13
L_0c4c:
s_add_co_u32 ttmp13, ttmp13, 1
L_0c50:
.long 0xd7610002, 0x0000f20d // preserved: v_writelane_b32 v2, s13, ttmp13
L_0c58:
s_add_co_u32 ttmp13, ttmp13, 1
L_0c5c:
.long 0xd7610002, 0x0000f20e // preserved: v_writelane_b32 v2, s14, ttmp13
L_0c64:
s_add_co_u32 ttmp13, ttmp13, 1
L_0c68:
.long 0xd7610002, 0x0000f20f // preserved: v_writelane_b32 v2, s15, ttmp13
L_0c70:
s_add_co_u32 ttmp13, ttmp13, 1
L_0c74:
s_cmp_eq_u32 ttmp13, 32
L_0c78:
s_cbranch_scc0 L_0ca0
L_0c7c:
s_add_co_u32 ttmp10, ttmp8, ttmp4
L_0c80:
s_add_co_ci_u32 ttmp11, ttmp9, 0
L_0c84:
global_store_addtid_b32 v2, ttmp[10:11] scope:SCOPE_SYS
L_0c90:
s_add_co_u32 ttmp4, ttmp4, 0x80
L_0c98:
s_mov_b32 ttmp13, 0
L_0c9c:
v_mov_b32_e32 v2, 0
L_0ca0:
s_add_co_u32 m0, m0, 16
L_0ca4:
s_cmp_lt_u32 m0, 0x60
L_0cac:
s_cbranch_scc1 L_0b94
L_0cb0:
s_movrels_b64 s[0:1], s[0:1]
L_0cb4:
s_movrels_b64 s[2:3], s[2:3]
L_0cb8:
s_movrels_b64 s[4:5], s[4:5]
L_0cbc:
s_movrels_b64 s[6:7], s[6:7]
L_0cc0:
s_movrels_b64 s[8:9], s[8:9]
L_0cc4:
s_movrels_b64 s[10:11], s[10:11]
L_0cc8:
.long 0xd7610002, 0x0000f200 // preserved: v_writelane_b32 v2, s0, ttmp13
L_0cd0:
s_add_co_u32 ttmp13, ttmp13, 1
L_0cd4:
.long 0xd7610002, 0x0000f201 // preserved: v_writelane_b32 v2, s1, ttmp13
L_0cdc:
s_add_co_u32 ttmp13, ttmp13, 1
L_0ce0:
.long 0xd7610002, 0x0000f202 // preserved: v_writelane_b32 v2, s2, ttmp13
L_0ce8:
s_add_co_u32 ttmp13, ttmp13, 1
L_0cec:
.long 0xd7610002, 0x0000f203 // preserved: v_writelane_b32 v2, s3, ttmp13
L_0cf4:
s_add_co_u32 ttmp13, ttmp13, 1
L_0cf8:
.long 0xd7610002, 0x0000f204 // preserved: v_writelane_b32 v2, s4, ttmp13
L_0d00:
s_add_co_u32 ttmp13, ttmp13, 1
L_0d04:
.long 0xd7610002, 0x0000f205 // preserved: v_writelane_b32 v2, s5, ttmp13
L_0d0c:
s_add_co_u32 ttmp13, ttmp13, 1
L_0d10:
.long 0xd7610002, 0x0000f206 // preserved: v_writelane_b32 v2, s6, ttmp13
L_0d18:
s_add_co_u32 ttmp13, ttmp13, 1
L_0d1c:
.long 0xd7610002, 0x0000f207 // preserved: v_writelane_b32 v2, s7, ttmp13
L_0d24:
s_add_co_u32 ttmp13, ttmp13, 1
L_0d28:
.long 0xd7610002, 0x0000f208 // preserved: v_writelane_b32 v2, s8, ttmp13
L_0d30:
s_add_co_u32 ttmp13, ttmp13, 1
L_0d34:
.long 0xd7610002, 0x0000f209 // preserved: v_writelane_b32 v2, s9, ttmp13
L_0d3c:
s_add_co_u32 ttmp13, ttmp13, 1
L_0d40:
.long 0xd7610002, 0x0000f20a // preserved: v_writelane_b32 v2, s10, ttmp13
L_0d48:
s_add_co_u32 ttmp13, ttmp13, 1
L_0d4c:
.long 0xd7610002, 0x0000f20b // preserved: v_writelane_b32 v2, s11, ttmp13
L_0d54:
s_add_co_u32 ttmp13, ttmp13, 1
L_0d58:
s_mov_b32 exec_lo, 0xffff
L_0d60:
s_add_co_u32 ttmp10, ttmp8, ttmp4
L_0d64:
s_add_co_ci_u32 ttmp11, ttmp9, 0
L_0d68:
global_store_addtid_b32 v2, ttmp[10:11] scope:SCOPE_SYS
L_0d74:
s_mov_b32 exec_lo, -1
L_0d78:
s_lshr_b32 m0, ttmp7, 25
L_0d7c:
s_and_b32 m0, m0, 1
L_0d80:
s_cmp_eq_u32 m0, 1
L_0d84:
s_cbranch_scc1 L_0d90
L_0d88:
s_mov_b32 exec_hi, 0
L_0d8c:
s_branch L_0d94
L_0d90:
s_mov_b32 exec_hi, -1
L_0d94:
s_getreg_b32 ttmp15, hwreg(HW_REG_WAVE_LDS_ALLOC, 12, 9)
L_0d98:
s_and_b32 ttmp15, ttmp15, -1
L_0d9c:
s_cbranch_scc0 L_0ea8
L_0da0:
s_and_b32 ttmp14, ttmp1, 0x80000000
L_0da8:
s_cbranch_scc0 L_0ea8
L_0dac:
s_lshl_b32 ttmp15, ttmp15, 10
L_0db0:
s_getreg_b32 ttmp4, hwreg(HW_REG_WAVE_GPR_ALLOC, 12, 8)
L_0db4:
s_add_co_u32 ttmp4, ttmp4, 1
L_0db8:
s_bitcmp1_b32 ttmp7, 25
L_0dbc:
s_cbranch_scc1 L_0dc8
L_0dc0:
s_lshl_b32 ttmp4, ttmp4, 9
L_0dc4:
s_branch L_0dcc
L_0dc8:
s_lshl_b32 ttmp4, ttmp4, 10
L_0dcc:
s_add_co_u32 ttmp4, ttmp4, 0x200
L_0dd4:
s_add_co_u32 ttmp4, ttmp4, 0x200
L_0ddc:
.long 0xd71f0000, 0x000100c1 // preserved: v_mbcnt_lo_u32_b32 v0, -1, 0
L_0de4:
.long 0xd7200000, 0x000200c1 // preserved: v_mbcnt_hi_u32_b32 v0, -1, v0
L_0dec:
v_mul_u32_u24_e32 v0, 4, v0
L_0df0:
s_lshr_b32 m0, ttmp7, 25
L_0df4:
s_and_b32 m0, m0, 1
L_0df8:
s_cmp_eq_u32 m0, 1
L_0dfc:
s_mov_b32 m0, 0
L_0e00:
s_cbranch_scc1 L_0e58
L_0e04:
s_mov_b32 s3, 0x80
L_0e0c:
s_nop 0
L_0e10:
s_nop 0
L_0e14:
s_nop 0
L_0e18:
ds_load_b32 v1, v0
L_0e20:
s_wait_idle
L_0e24:
s_add_co_u32 ttmp10, ttmp8, ttmp4
L_0e28:
s_add_co_ci_u32 ttmp11, ttmp9, 0
L_0e2c:
global_store_addtid_b32 v1, ttmp[10:11] scope:SCOPE_SYS
L_0e38:
s_add_co_u32 m0, m0, s3
L_0e3c:
s_add_co_u32 ttmp4, ttmp4, s3
L_0e40:
.long 0xd5250000, 0x0001ff00, 0x00000080 // preserved: v_add_nc_u32_e64 v0, v0, 0x80
L_0e4c:
s_cmp_lt_u32 m0, ttmp15
L_0e50:
s_cbranch_scc1 L_0e18
L_0e54:
s_branch L_0ea8
L_0e58:
s_mov_b32 s3, 0x100
L_0e60:
s_nop 0
L_0e64:
s_nop 0
L_0e68:
s_nop 0
L_0e6c:
ds_load_b32 v1, v0
L_0e74:
s_wait_idle
L_0e78:
s_add_co_u32 ttmp10, ttmp8, ttmp4
L_0e7c:
s_add_co_ci_u32 ttmp11, ttmp9, 0
L_0e80:
global_store_addtid_b32 v1, ttmp[10:11] scope:SCOPE_SYS
L_0e8c:
s_add_co_u32 m0, m0, s3
L_0e90:
s_add_co_u32 ttmp4, ttmp4, s3
L_0e94:
.long 0xd5250000, 0x0001ff00, 0x00000100 // preserved: v_add_nc_u32_e64 v0, v0, 0x100
L_0ea0:
s_cmp_lt_u32 m0, ttmp15
L_0ea4:
s_cbranch_scc1 L_0e6c
L_0ea8:
s_mov_b32 exec_lo, -1
L_0eac:
s_lshr_b32 m0, ttmp7, 25
L_0eb0:
s_and_b32 m0, m0, 1
L_0eb4:
s_cmp_eq_u32 m0, 1
L_0eb8:
s_cbranch_scc1 L_0ecc
L_0ebc:
s_mov_b32 ttmp4, 0x200
L_0ec4:
s_mov_b32 exec_hi, 0
L_0ec8:
s_branch L_0ed8
L_0ecc:
s_mov_b32 ttmp4, 0x400
L_0ed4:
s_mov_b32 exec_hi, -1
L_0ed8:
s_getreg_b32 ttmp15, hwreg(HW_REG_WAVE_GPR_ALLOC, 12, 8)
L_0edc:
s_add_co_u32 ttmp15, ttmp15, 1
L_0ee0:
s_lshl_b32 ttmp15, ttmp15, 2
L_0ee4:
s_lshr_b32 m0, ttmp7, 25
L_0ee8:
s_and_b32 m0, m0, 1
L_0eec:
s_cmp_eq_u32 m0, 1
L_0ef0:
s_cbranch_scc1 L_0f60
L_0ef4:
s_mov_b32 m0, 4
L_0ef8:
s_cmp_lt_u32 m0, ttmp15
L_0efc:
s_cbranch_scc0 L_0fc8
L_0f00:
v_movrels_b32_e32 v0, v0
L_0f04:
v_movrels_b32_e32 v1, v1
L_0f08:
v_movrels_b32_e32 v2, v2
L_0f0c:
v_movrels_b32_e32 v3, v3
L_0f10:
s_add_co_u32 ttmp10, ttmp8, ttmp4
L_0f14:
s_add_co_ci_u32 ttmp11, ttmp9, 0
L_0f18:
global_store_addtid_b32 v0, ttmp[10:11] scope:SCOPE_SYS
L_0f24:
global_store_addtid_b32 v1, ttmp[10:11] offset:128 scope:SCOPE_SYS
L_0f30:
global_store_addtid_b32 v2, ttmp[10:11] offset:256 scope:SCOPE_SYS
L_0f3c:
global_store_addtid_b32 v3, ttmp[10:11] offset:384 scope:SCOPE_SYS
L_0f48:
s_add_co_u32 m0, m0, 4
L_0f4c:
s_add_co_u32 ttmp4, ttmp4, 0x200
L_0f54:
s_cmp_lt_u32 m0, ttmp15
L_0f58:
s_cbranch_scc1 L_0f00
L_0f5c:
s_branch L_0fc8
L_0f60:
s_mov_b32 m0, 4
L_0f64:
s_cmp_lt_u32 m0, ttmp15
L_0f68:
s_cbranch_scc0 L_0fc8
L_0f6c:
v_movrels_b32_e32 v0, v0
L_0f70:
v_movrels_b32_e32 v1, v1
L_0f74:
v_movrels_b32_e32 v2, v2
L_0f78:
v_movrels_b32_e32 v3, v3
L_0f7c:
s_add_co_u32 ttmp10, ttmp8, ttmp4
L_0f80:
s_add_co_ci_u32 ttmp11, ttmp9, 0
L_0f84:
global_store_addtid_b32 v0, ttmp[10:11] scope:SCOPE_SYS
L_0f90:
global_store_addtid_b32 v1, ttmp[10:11] offset:256 scope:SCOPE_SYS
L_0f9c:
global_store_addtid_b32 v2, ttmp[10:11] offset:512 scope:SCOPE_SYS
L_0fa8:
global_store_addtid_b32 v3, ttmp[10:11] offset:768 scope:SCOPE_SYS
L_0fb4:
s_add_co_u32 m0, m0, 4
L_0fb8:
s_add_co_u32 ttmp4, ttmp4, 0x400
L_0fc0:
s_cmp_lt_u32 m0, ttmp15
L_0fc4:
s_cbranch_scc1 L_0f6c
L_0fc8:
s_branch L_15dc
L_0fcc:
s_mov_b32 ttmp8, exec_lo
L_0fd0:
s_and_b32 ttmp9, exec_hi, 0x1ffffff
L_0fd8:
s_mov_b32 ttmp5, exec_hi
L_0fdc:
s_getreg_b32 ttmp6, hwreg(HW_REG_WAVE_STATUS, 29, 1)
L_0fe0:
s_lshl_b32 ttmp6, ttmp6, 25
L_0fe4:
s_and_b32 ttmp2, exec_hi, 0x4000000
L_0fec:
s_cbranch_scc0 L_1100
L_0ff0:
s_mov_b32 exec_lo, -1
L_0ff4:
s_lshr_b32 m0, ttmp6, 25
L_0ff8:
s_and_b32 m0, m0, 1
L_0ffc:
s_cmp_eq_u32 m0, 1
L_1000:
s_cbranch_scc1 L_100c
L_1004:
s_mov_b32 exec_hi, 0
L_1008:
s_branch L_1010
L_100c:
s_mov_b32 exec_hi, -1
L_1010:
s_getreg_b32 ttmp3, hwreg(HW_REG_WAVE_LDS_ALLOC, 12, 9)
L_1014:
s_and_b32 ttmp3, ttmp3, -1
L_1018:
s_cbranch_scc0 L_1100
L_101c:
s_lshl_b32 ttmp3, ttmp3, 10
L_1020:
s_getreg_b32 ttmp12, hwreg(HW_REG_WAVE_GPR_ALLOC, 12, 8)
L_1024:
s_add_co_u32 ttmp12, ttmp12, 1
L_1028:
s_bitcmp1_b32 ttmp6, 25
L_102c:
s_cbranch_scc1 L_1038
L_1030:
s_lshl_b32 ttmp12, ttmp12, 9
L_1034:
s_branch L_103c
L_1038:
s_lshl_b32 ttmp12, ttmp12, 10
L_103c:
s_add_co_u32 ttmp12, ttmp12, 0x200
L_1044:
s_add_co_u32 ttmp12, ttmp12, 0x200
L_104c:
s_lshr_b32 m0, ttmp6, 25
L_1050:
s_and_b32 m0, m0, 1
L_1054:
s_cmp_eq_u32 m0, 1
L_1058:
s_mov_b32 m0, 0
L_105c:
.long 0xd71f0001, 0x000100c1 // preserved: v_mbcnt_lo_u32_b32 v1, -1, 0
L_1064:
.long 0xd7200001, 0x000202c1 // preserved: v_mbcnt_hi_u32_b32 v1, -1, v1
L_106c:
v_lshlrev_b32_e32 v1, 2, v1
L_1070:
s_cbranch_scc1 L_10bc
L_1074:
s_add_co_u32 ttmp10, ttmp8, ttmp12
L_1078:
s_add_co_ci_u32 ttmp11, ttmp9, 0
L_107c:
global_load_addtid_b32 v0, ttmp[10:11] scope:SCOPE_SYS
L_1088:
s_wait_idle
L_108c:
ds_store_b32 v1, v0
L_1094:
.long 0xd5250001, 0x0001ff01, 0x00000080 // preserved: v_add_nc_u32_e64 v1, v1, 0x80
L_10a0:
s_add_co_u32 m0, m0, 0x80
L_10a8:
s_add_co_u32 ttmp12, ttmp12, 0x80
L_10b0:
s_cmp_lt_u32 m0, ttmp3
L_10b4:
s_cbranch_scc1 L_1074
L_10b8:
s_branch L_1100
L_10bc:
s_add_co_u32 ttmp10, ttmp8, ttmp12
L_10c0:
s_add_co_ci_u32 ttmp11, ttmp9, 0
L_10c4:
global_load_addtid_b32 v0, ttmp[10:11] scope:SCOPE_SYS
L_10d0:
s_wait_idle
L_10d4:
ds_store_b32 v1, v0
L_10dc:
.long 0xd5250001, 0x0001ff01, 0x00000100 // preserved: v_add_nc_u32_e64 v1, v1, 0x100
L_10e8:
s_add_co_u32 m0, m0, 0x100
L_10f0:
s_add_co_u32 ttmp12, ttmp12, 0x100
L_10f8:
s_cmp_lt_u32 m0, ttmp3
L_10fc:
s_cbranch_scc1 L_10bc
L_1100:
s_mov_b32 ttmp12, 0
L_1104:
s_mov_b32 exec_lo, -1
L_1108:
s_lshr_b32 m0, ttmp6, 25
L_110c:
s_and_b32 m0, m0, 1
L_1110:
s_cmp_eq_u32 m0, 1
L_1114:
s_cbranch_scc1 L_1120
L_1118:
s_mov_b32 exec_hi, 0
L_111c:
s_branch L_1124
L_1120:
s_mov_b32 exec_hi, -1
L_1124:
s_getreg_b32 ttmp3, hwreg(HW_REG_WAVE_GPR_ALLOC, 12, 8)
L_1128:
s_add_co_u32 ttmp3, ttmp3, 1
L_112c:
s_lshl_b32 ttmp3, ttmp3, 2
L_1130:
s_lshr_b32 m0, ttmp6, 25
L_1134:
s_and_b32 m0, m0, 1
L_1138:
s_cmp_eq_u32 m0, 1
L_113c:
s_cbranch_scc1 L_11f0
L_1140:
s_mov_b32 ttmp2, ttmp12
L_1144:
s_add_co_u32 ttmp12, ttmp12, 0x200
L_114c:
s_mov_b32 m0, 4
L_1150:
s_add_co_u32 ttmp10, ttmp8, ttmp12
L_1154:
s_add_co_ci_u32 ttmp11, ttmp9, 0
L_1158:
global_load_addtid_b32 v0, ttmp[10:11] scope:SCOPE_SYS
L_1164:
global_load_addtid_b32 v1, ttmp[10:11] offset:128 scope:SCOPE_SYS
L_1170:
global_load_addtid_b32 v2, ttmp[10:11] offset:256 scope:SCOPE_SYS
L_117c:
global_load_addtid_b32 v3, ttmp[10:11] offset:384 scope:SCOPE_SYS
L_1188:
s_wait_idle
L_118c:
v_movreld_b32_e32 v0, v0
L_1190:
v_movreld_b32_e32 v1, v1
L_1194:
v_movreld_b32_e32 v2, v2
L_1198:
v_movreld_b32_e32 v3, v3
L_119c:
s_add_co_u32 m0, m0, 4
L_11a0:
s_add_co_u32 ttmp12, ttmp12, 0x200
L_11a8:
s_cmp_lt_u32 m0, ttmp3
L_11ac:
s_cbranch_scc1 L_1150
L_11b0:
s_add_co_u32 ttmp10, ttmp8, ttmp2
L_11b4:
s_add_co_ci_u32 ttmp11, ttmp9, 0
L_11b8:
global_load_addtid_b32 v0, ttmp[10:11] scope:SCOPE_SYS
L_11c4:
global_load_addtid_b32 v1, ttmp[10:11] offset:128 scope:SCOPE_SYS
L_11d0:
global_load_addtid_b32 v2, ttmp[10:11] offset:256 scope:SCOPE_SYS
L_11dc:
global_load_addtid_b32 v3, ttmp[10:11] offset:384 scope:SCOPE_SYS
L_11e8:
s_wait_idle
L_11ec:
s_branch L_12a4
L_11f0:
s_mov_b32 ttmp2, ttmp12
L_11f4:
s_add_co_u32 ttmp12, ttmp12, 0x400
L_11fc:
s_mov_b32 m0, 4
L_1200:
s_cmp_lt_u32 m0, ttmp3
L_1204:
s_cbranch_scc0 L_1268
L_1208:
s_add_co_u32 ttmp10, ttmp8, ttmp12
L_120c:
s_add_co_ci_u32 ttmp11, ttmp9, 0
L_1210:
global_load_addtid_b32 v0, ttmp[10:11] scope:SCOPE_SYS
L_121c:
global_load_addtid_b32 v1, ttmp[10:11] offset:256 scope:SCOPE_SYS
L_1228:
global_load_addtid_b32 v2, ttmp[10:11] offset:512 scope:SCOPE_SYS
L_1234:
global_load_addtid_b32 v3, ttmp[10:11] offset:768 scope:SCOPE_SYS
L_1240:
s_wait_idle
L_1244:
v_movreld_b32_e32 v0, v0
L_1248:
v_movreld_b32_e32 v1, v1
L_124c:
v_movreld_b32_e32 v2, v2
L_1250:
v_movreld_b32_e32 v3, v3
L_1254:
s_add_co_u32 m0, m0, 4
L_1258:
s_add_co_u32 ttmp12, ttmp12, 0x400
L_1260:
s_cmp_lt_u32 m0, ttmp3
L_1264:
s_cbranch_scc1 L_1208
L_1268:
s_add_co_u32 ttmp10, ttmp8, ttmp2
L_126c:
s_add_co_ci_u32 ttmp11, ttmp9, 0
L_1270:
global_load_addtid_b32 v0, ttmp[10:11] scope:SCOPE_SYS
L_127c:
global_load_addtid_b32 v1, ttmp[10:11] offset:256 scope:SCOPE_SYS
L_1288:
global_load_addtid_b32 v2, ttmp[10:11] offset:512 scope:SCOPE_SYS
L_1294:
global_load_addtid_b32 v3, ttmp[10:11] offset:768 scope:SCOPE_SYS
L_12a0:
s_wait_idle
L_12a4:
s_getreg_b32 ttmp12, hwreg(HW_REG_WAVE_GPR_ALLOC, 12, 8)
L_12a8:
s_add_co_u32 ttmp12, ttmp12, 1
L_12ac:
s_bitcmp1_b32 ttmp6, 25
L_12b0:
s_cbranch_scc1 L_12bc
L_12b4:
s_lshl_b32 ttmp12, ttmp12, 9
L_12b8:
s_branch L_12c0
L_12bc:
s_lshl_b32 ttmp12, ttmp12, 10
L_12c0:
s_add_co_u32 ttmp12, ttmp12, 0x200
L_12c8:
s_sub_co_u32 ttmp12, ttmp12, 0x60
L_12d0:
s_add_co_u32 ttmp10, ttmp8, ttmp12
L_12d4:
s_add_co_ci_u32 ttmp11, ttmp9, 0
L_12d8:
s_mov_b32 m0, 0x6c
L_12e0:
s_load_b128 s[0:3], ttmp[10:11], 0x0 scope:SCOPE_SYS
L_12e8:
s_wait_idle
L_12ec:
s_sub_co_u32 m0, m0, 4
L_12f0:
s_nop 0
L_12f4:
s_movreld_b64 s[0:1], s[0:1]
L_12f8:
s_movreld_b64 s[2:3], s[2:3]
L_12fc:
s_sub_co_u32 ttmp10, ttmp10, 32
L_1300:
s_sub_co_ci_u32 ttmp11, ttmp11, 0
L_1304:
s_load_b256 s[0:7], ttmp[10:11], 0x0 scope:SCOPE_SYS
L_130c:
s_wait_idle
L_1310:
s_sub_co_u32 m0, m0, 8
L_1314:
s_nop 0
L_1318:
s_movreld_b64 s[0:1], s[0:1]
L_131c:
s_movreld_b64 s[2:3], s[2:3]
L_1320:
s_movreld_b64 s[4:5], s[4:5]
L_1324:
s_movreld_b64 s[6:7], s[6:7]
L_1328:
s_sub_co_u32 ttmp10, ttmp10, 64
L_132c:
s_sub_co_ci_u32 ttmp11, ttmp11, 0
L_1330:
s_load_b512 s[0:15], ttmp[10:11], 0x0 scope:SCOPE_SYS
L_1338:
s_wait_idle
L_133c:
s_sub_co_u32 m0, m0, 16
L_1340:
s_nop 0
L_1344:
s_movreld_b64 s[0:1], s[0:1]
L_1348:
s_movreld_b64 s[2:3], s[2:3]
L_134c:
s_movreld_b64 s[4:5], s[4:5]
L_1350:
s_movreld_b64 s[6:7], s[6:7]
L_1354:
s_movreld_b64 s[8:9], s[8:9]
L_1358:
s_movreld_b64 s[10:11], s[10:11]
L_135c:
s_movreld_b64 s[12:13], s[12:13]
L_1360:
s_movreld_b64 s[14:15], s[14:15]
L_1364:
s_cmp_eq_u32 m0, 0
L_1368:
s_cbranch_scc0 L_1328
L_136c:
s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE), 0
L_1374:
s_getreg_b32 ttmp12, hwreg(HW_REG_WAVE_GPR_ALLOC, 12, 8)
L_1378:
s_add_co_u32 ttmp12, ttmp12, 1
L_137c:
s_bitcmp1_b32 ttmp6, 25
L_1380:
s_cbranch_scc1 L_138c
L_1384:
s_lshl_b32 ttmp12, ttmp12, 9
L_1388:
s_branch L_1390
L_138c:
s_lshl_b32 ttmp12, ttmp12, 10
L_1390:
s_add_co_u32 ttmp12, ttmp12, 0x200
L_1398:
s_add_co_u32 ttmp10, ttmp8, ttmp12
L_139c:
s_add_co_ci_u32 ttmp11, ttmp9, 0
L_13a0:
s_mov_b32 exec_hi, ttmp5
L_13a4:
s_load_b32 ttmp3, ttmp[10:11], 0x0 scope:SCOPE_SYS
L_13ac:
s_load_b32 ttmp0, ttmp[10:11], 0x4 scope:SCOPE_SYS
L_13b4:
s_load_b32 ttmp1, ttmp[10:11], 0x8 scope:SCOPE_SYS
L_13bc:
s_load_b32 ttmp4, ttmp[10:11], 0xc scope:SCOPE_SYS
L_13c4:
s_load_b32 ttmp5, ttmp[10:11], 0x10 scope:SCOPE_SYS
L_13cc:
s_load_b32 ttmp14, ttmp[10:11], 0x14 scope:SCOPE_SYS
L_13d4:
s_load_b32 ttmp15, ttmp[10:11], 0x18 scope:SCOPE_SYS
L_13dc:
s_load_b32 ttmp13, ttmp[10:11], 0x1c scope:SCOPE_SYS
L_13e4:
s_load_b32 ttmp7, ttmp[10:11], 0x20 scope:SCOPE_SYS
L_13ec:
s_load_b32 ttmp2, ttmp[10:11], 0x24 scope:SCOPE_SYS
L_13f4:
s_wait_idle
L_13f8:
s_setreg_b32 hwreg(HW_REG_WAVE_SCRATCH_BASE_LO), ttmp2
L_13fc:
s_load_b32 ttmp2, ttmp[10:11], 0x28 scope:SCOPE_SYS
L_1404:
s_wait_idle
L_1408:
s_setreg_b32 hwreg(HW_REG_WAVE_SCRATCH_BASE_HI), ttmp2
L_140c:
s_load_b32 ttmp2, ttmp[10:11], 0x2c scope:SCOPE_SYS
L_1414:
s_wait_idle
L_1418:
s_setreg_b32 hwreg(HW_REG_WAVE_EXCP_FLAG_USER), ttmp2
L_141c:
s_load_b32 ttmp2, ttmp[10:11], 0x30 scope:SCOPE_SYS
L_1424:
s_wait_idle
L_1428:
s_setreg_b32 hwreg(HW_REG_WAVE_TRAP_CTRL), ttmp2
L_142c:
s_and_b32 ttmp2, exec_hi, 0x4000000
L_1434:
s_cbranch_scc0 L_14c0
L_1438:
s_load_b32 ttmp2, ttmp[10:11], 0x38 scope:SCOPE_SYS
L_1440:
s_wait_idle
L_1444:
s_bitcmp1_b32 ttmp2, 0
L_1448:
s_cbranch_scc0 L_14c0
L_144c:
s_lshr_b32 ttmp2, ttmp2, 16
L_1450:
s_and_b32 ttmp2, ttmp2, ttmp2
L_1454:
s_cbranch_scc0 L_1464
L_1458:
s_barrier_signal -1
L_145c:
s_add_co_i32 ttmp2, ttmp2, -1
L_1460:
s_branch L_1450
L_1464:
s_mov_b32 ttmp12, 0x80
L_146c:
s_mov_b32 m0, 1
L_1470:
s_load_b32 ttmp2, ttmp[10:11], ttmp12 offset:0x0 scope:SCOPE_SYS
L_1478:
s_wait_kmcnt 0x0
L_147c:
s_add_co_u32 ttmp12, ttmp12, 4
L_1480:
s_bfe_u32 exec_lo, ttmp2, 0x70004
L_1488:
s_lshl_b32 exec_lo, exec_lo, 16
L_148c:
s_or_b32 m0, m0, exec_lo
L_1490:
s_barrier_init m0
L_1494:
s_and_not1_b32 m0, m0, 0x7f0000
L_149c:
s_lshr_b32 ttmp2, ttmp2, 16
L_14a0:
s_and_b32 ttmp2, ttmp2, ttmp2
L_14a4:
s_cbranch_scc0 L_14b4
L_14a8:
s_barrier_signal m0
L_14ac:
s_add_co_i32 ttmp2, ttmp2, -1
L_14b0:
s_branch L_14a0
L_14b4:
s_add_co_u32 m0, m0, 1
L_14b8:
s_cmp_gt_u32 m0, 16
L_14bc:
s_cbranch_scc0 L_1470
L_14c0:
s_load_b32 ttmp2, ttmp[10:11], 0x3c scope:SCOPE_SYS
L_14c8:
s_wait_kmcnt 0x0
L_14cc:
s_bitcmp1_b32 ttmp2, 0
L_14d0:
s_cbranch_scc0 L_1504
L_14d4:
s_bitcmp1_b32 exec_hi, 26
L_14d8:
s_cbranch_scc0 L_14e4
L_14dc:
s_cmp_eq_u32 0, 1
L_14e0:
s_barrier_signal_isfirst -4
L_14e4:
s_barrier_wait 0xfffc
L_14e8:
s_cbranch_scc0 L_1504
L_14ec:
s_lshr_b32 ttmp2, ttmp2, 16
L_14f0:
s_and_b32 ttmp2, ttmp2, ttmp2
L_14f4:
s_cbranch_scc0 L_1504
L_14f8:
s_barrier_signal -3
L_14fc:
s_add_co_i32 ttmp2, ttmp2, -1
L_1500:
s_branch L_14f0
L_1504:
s_load_b32 ttmp2, ttmp[10:11], 0x40 scope:SCOPE_SYS
L_150c:
s_wait_kmcnt 0x0
L_1510:
s_setreg_b32 hwreg(HW_REG_WAVE_SCHED_MODE), ttmp2
L_1514:
s_mov_b32 m0, ttmp3
L_1518:
s_mov_b32 exec_lo, ttmp4
L_151c:
s_mov_b32 exec_hi, ttmp5
L_1520:
s_setreg_b32 hwreg(HW_REG_XNACK_MASK), ttmp13
L_1524:
s_setreg_b32 hwreg(HW_REG_WAVE_EXCP_FLAG_PRIV, 0, 5), ttmp15
L_1528:
s_lshr_b32 ttmp15, ttmp15, 6
L_152c:
s_setreg_b32 hwreg(HW_REG_WAVE_EXCP_FLAG_PRIV, 6, 1), ttmp15
L_1530:
s_lshr_b32 ttmp15, ttmp15, 2
L_1534:
s_setreg_b32 hwreg(HW_REG_WAVE_EXCP_FLAG_PRIV, 8, 24), ttmp15
L_1538:
s_setreg_b32 hwreg(HW_REG_WAVE_MODE), ttmp7
L_153c:
s_getreg_b32 ttmp2, hwreg(HW_REG_WAVE_GPR_ALLOC, 12, 8)
L_1540:
s_add_co_u32 ttmp2, ttmp2, 1
L_1544:
s_bitcmp1_b32 ttmp6, 25
L_1548:
s_cbranch_scc1 L_1554
L_154c:
s_lshl_b32 ttmp2, ttmp2, 9
L_1550:
s_branch L_1558
L_1554:
s_lshl_b32 ttmp2, ttmp2, 10
L_1558:
s_add_co_u32 ttmp2, ttmp2, 0x1c0
L_1560:
s_add_co_u32 ttmp2, ttmp2, ttmp8
L_1564:
s_add_co_ci_u32 ttmp3, ttmp9, 0
L_1568:
s_load_b128 ttmp[4:7], ttmp[2:3], 0x10 scope:SCOPE_SYS
L_1570:
s_load_b128 ttmp[8:11], ttmp[2:3], 0x20 scope:SCOPE_SYS
L_1578:
s_load_b32 ttmp13, ttmp[2:3], 0x34 scope:SCOPE_SYS
L_1580:
s_wait_idle
L_1584:
s_lshr_b32 ttmp2, ttmp11, 22
L_1588:
s_setreg_b32 hwreg(HW_REG_XNACK_STATE_PRIV, 18, 1), ttmp2
L_158c:
s_lshr_b32 ttmp2, ttmp11, 21
L_1590:
s_setreg_b32 hwreg(HW_REG_XNACK_STATE_PRIV, 16, 1), ttmp2
L_1594:
s_lshr_b32 ttmp2, ttmp11, 14
L_1598:
s_setreg_b32 hwreg(HW_REG_XNACK_STATE_PRIV, 0, 7), ttmp2
L_159c:
s_and_b32 ttmp1, ttmp1, 0x1ffffff
L_15a4:
s_and_b64 exec, exec, exec
L_15a8:
s_and_b64 vcc, vcc, vcc
L_15ac:
s_setreg_b32 hwreg(HW_REG_WAVE_STATE_PRIV), ttmp14
L_15b0:
s_getreg_b32 ttmp2, hwreg(HW_REG_WAVE_STATUS)
L_15b4:
s_bitcmp0_b32 ttmp2, 11
L_15b8:
s_cbranch_scc1 L_15c8
L_15bc:
s_barrier_signal_isfirst -2
L_15c0:
s_barrier_wait 0xfffe
L_15c4:
s_cbranch_scc0 L_15cc
L_15c8:
s_barrier_signal -4
L_15cc:
s_barrier_wait 0xfffc
L_15d0:
s_lshr_b32 ttmp14, ttmp14, 9
L_15d4:
s_setreg_b32 hwreg(HW_REG_WAVE_STATE_PRIV, 9, 1), ttmp14
L_15d8:
s_rfe_i64 ttmp[0:1]
L_15dc:
s_getreg_b32 ttmp2, hwreg(HW_REG_WAVE_STATUS)
L_15e0:
s_bitcmp0_b32 ttmp2, 11
L_15e4:
s_cbranch_scc1 L_15f4
L_15e8:
s_barrier_signal_isfirst -2
L_15ec:
s_barrier_wait 0xfffe
L_15f0:
s_cbranch_scc0 L_15f8
L_15f4:
s_barrier_signal -4
L_15f8:
s_barrier_wait 0xfffc
L_15fc:
s_endpgm_saved
L_1600:
s_code_end
L_1604:
s_code_end
L_1608:
s_code_end
L_160c:
s_code_end
L_1610:
s_code_end
L_1614:
v_illegal
