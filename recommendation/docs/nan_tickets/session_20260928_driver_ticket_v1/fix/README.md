The proposed repair is in AMD KFD's CWSR handler, at `L_NOT_WAVE_START` inside
`VMEM_ON_TRAP_ENTRY_WA` in
`drivers/gpu/drm/amd/amdkfd/cwsr_trap_handler_gfx12.asm`.
The matching embedded payload is `cwsr_trap_gfx12_1_0_hex` in
`drivers/gpu/drm/amd/amdkfd/cwsr_trap_handler.h`.
[proposed_fix.patch](proposed_fix.patch) changes both files and is 2,091 bytes.
It applies to the original source snapshot from
`amdgpu-7.1.1-2397345.24.04`; the complete original and proposed files are
retained in [source](source/). No installed file was changed to prepare this package.

The original handler uses encoded `v1` while the trapped wave's VGPR bank
selectors are still active. `v_readlane_b32` reads the SRC0-selected physical
register; `v_mov_b32` and `v_writelane_b32` write the DST-selected physical
register. With different SRC0 and DST selectors, the save/restore sequence
copies a complete 32-bit lane-0 word from one physical VGPR to another. A later
bank reset cannot undo that overwritten value. This mechanism is the culprit
addressed by this patch; runtime counterfactual evidence belongs to the parent
ticket and reproducer package.

The repair saves the six `MODE[17:12]` bank bits as `k` in `ttmp2`, clears
those bits through zeroed `ttmp3` using `S_SETREG_B32`, and then performs every
vector access in bank zero. It saves `EXEC_LO`, activates lane zero, and saves
physical `v1[0]`. `V_SUB_NC_U32` writes `63-k` to `v1`; the prefetch uses scalar
base `[ttmp2,ttmp3]=k` and immediate offset `-63`, so its address remains
`k + (63-k) - 63 = 0` for every `k` from 0 through 63. The handler restores
physical `v1[0]`, restores the saved bank bits, and restores `EXEC_LO`.
The non-carry arithmetic preserves VCC and SCC. SRC2 and every other MODE bit
are untouched. The affected source family has `WAVE32_ONLY` enabled.

[native/prologue.txt](native/prologue.txt) contains the original and corrected
instructions with byte offsets and exact native words.
[native/prologue.json](native/prologue.json) provides the same information in
machine-readable form. Complete binaries, disassemblies and LLVM assembly
sources are in [native](native/). The complete disassemblies were reconstructed
into bytes and checked against both binaries during packaging.

| Payload | Bytes | SHA-256 |
| --- | ---: | --- |
| Original handler | 5,656 | `0f718b5eae143bad8625658be139f6e636263e1bd09510f2653f96f62e19b290` |
| Corrected handler | 5,672 | `68c31ab204bdfe99551213b1716fb8b9944cd0fc315adb0b7ea8d9661e398b80` |
| Combined source/header patch | 2,091 | `58ed3645252ebbb3fe6fcad493a6db7a62b30740ceac937d18acb7995e6dd158` |

The replacement is original byte range `[60,108)` to corrected range
`[60,124)`. The branch at byte 4 changes from `0xbfa003f1` to `0xbfa003f5`,
moving its target from byte 4044 to byte 4060. Every other instruction byte is
identical after allowing for the 16-byte insertion. The packaged header uses
the original two-words-per-line style; its complete C hex-token sequence equals
the tested header, whose selected array used one word per line.

**Original SP3 assembler acceptance was not tested.** The assembly hunk retains
the exact tested source file's “Design proposal only” comment. The actual
runtime-tested payload was reconstructed with LLVM for `gfx1250` plus
preserved installed native words, then embedded in the header. A pure LLVM
disassembly/assembly round trip changed bit 57 in 84 existing VOP3 instructions.
The retained `.s` files emit those instructions as `.long`, preserving their
installed encodings; this includes the prologue's lane read and lane write.
LLVM assembles the remaining instructions and adjusts the branch. The rebuilt
baseline equals the installed handler byte for byte. The tested repair uses
`S_SETREG_B32`; an earlier `S_SETREG_IMM32_B32` proposal is not this payload.

To review and independently check the package on CPU, use Python 3.9 or newer
and `patch`. Run from this directory:

```sh
sha256sum -c SHA256SUMS
python3 verify_fix.py
python3 verify_fix.py --llvm-bin /opt/rocm/lib/llvm/bin
```

The final command additionally reassembles only the two small handlers and
requires a `gfx1250`-capable LLVM toolchain. It does not compile the original
SP3 source or rebuild a kernel module. Verification applies the patch in a
private temporary directory, checks exact source outputs and header bytes,
checks every native disassembly word, and checks the complete package manifest.
[FIX_VERIFICATION.json](FIX_VERIFICATION.json) records packaging checks and
source pins. `prepare_fix.py` is the preparation receipt, has A0-specific input
paths, and is not the portable entrypoint. The read-only portable entrypoint is
`verify_fix.py`.

For integration, the affected family macro in this source is `CHIP_GC_12_0_3`;
it enables `HAVE_BANKED_VGPRS`, `WAVE32_ONLY` and `VMEM_ON_TRAP_ENTRY_WA`.
The generic `CHIP_GFX12`/SP3 command in the file header selects another family
and is insufficient to regenerate this affected handler. The driver team
should validate the assembly syntax and hazards using its affected-family SP3
generation path, regenerate the correct array, and compare its native semantics
with the supplied corrected payload. Editing the `.asm` alone does not update
the C array that the driver loads.

The patch paths are upstream-style. Apply with `patch -p1` at an upstream
kernel source root, or `patch -p4` at a copied DKMS source root containing
`amd/amdkfd`. The baseline source pins are:

| DKMS source path | SHA-256 |
| --- | --- |
| `amd/amdkfd/cwsr_trap_handler_gfx12.asm` | `5e0e339cb988120d99fbccf2869bfab29e0a143bd110ce6f00cfdc495d5bf69d` |
| `amd/amdkfd/cwsr_trap_handler.h` | `006a44ea767d581ecc84913ec8bdcd68476471510a1ccd9923b00efc3b8463ef` |
| `amd/amdkfd/kfd_device.c` | `7729b41821a7e24448e900b9ab677d7f9d4d8a1fd1ed4dec1f3f001f8f591d23` |

`kfd_cwsr_init` in [evidence/kfd_device.c](evidence/kfd_device.c), lines
562–571, chooses `cwsr_trap_gfx12_1_0_hex` and its `sizeof` for the affected
version path. `kgd2kfd_device_init` calls that helper at line 881; the compiled
size-selection instructions are in that caller. The prior tested build used kernel `6.14.0-37-generic`, matching
kernel headers, the original DKMS source above, and `AMDGPU_BTF=0`. Its baseline
command was `make KERNELVER=6.14.0-37-generic num_cpu_cores=32
module_build_dir=<scratch>/build_link`; the incremental candidate command was
`make -j32 TTM_NAME=amdttm SCHED_NAME=amd-sched -C
/lib/modules/6.14.0-37-generic/build M=<scratch>/source`.
[evidence/module_build.py](evidence/module_build.py),
[evidence/module_BUILD.json](evidence/module_BUILD.json), and
[evidence/module_build_inputs.json](evidence/module_build_inputs.json) retain
the exact prior procedure and receipt. They are historical records, not
portable execution entrypoints. No module rebuild occurred while preparing
this package. A production build also needs the team's normal signing and
deployment process; the prior experimental candidate was unsigned.

The pinned [module audit](evidence/module_audit.json) and independent
[later audit](evidence/prior_causal_audit.json) establish the link from the
header to the tested module. The original compressed module SHA-256 is
`dae133624fad0ae7bb7d95dd563edbc5218bcd0ba5ea1f5b44473093a0cc93d2`
and its decompressed SHA-256 is
`5b6e4617fdf247ec8d395660c850e10f49a1c5706dea9d1b8598e71cdaa8c2b9`.
The corrected stripped module SHA-256 is
`78de19854e38e298f6418bddcfa39f44a797f811f0f4b9c585482f4c4dc2b2b4`.
Original and corrected srcversion values are respectively
`EADFD8EC13CF3E6B329194E` and `AD83C153B9701570F12964E`; both report version
`7.1.1.31300009`.

The audit checked all 53 allocated sections. The handler starts at `.rodata`
offset 508096. The `.rodata` change consists of the corrected handler plus 16
additional alignment zero bytes, for 32 bytes of section growth. The entire
`.text` change is two size-immediate bytes in `kgd2kfd_device_init`, changing
the affected handler size from 5656 to 5672 while retaining the older size
3760. Build ID and srcversion also change; 827,454 normalized relocations and
57,598 symbol entries remain equivalent. Large module files were not copied,
rebuilt or rehashed in this packaging step; these module claims come from the
pinned prior audits.

This package addresses the demonstrated CWSR bank-copy corruption. It does
not establish that every training NaN is resolved. Residual STU1 behavior under
the corrected driver remains a separate investigation. The parent ticket
distinguishes original and corrected boot evidence and records which runtime
counterfactuals have actually been completed.
