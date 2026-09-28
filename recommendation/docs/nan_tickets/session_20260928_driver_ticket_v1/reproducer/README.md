# MI450/gfx1250 KFD CWSR lane-zero corruption reproducer

This package launches 14 tiny kernels, one wave32 each, with one 160-byte output allocation. It needs Python 3.11+ and a HIP/HSA runtime supporting gfx1250. It needs no PyTorch, training data, container, home-directory layout, or driver build toolchain. The executed 32,152-byte shader image and its assembly source are supplied.

The shader tags v1, v257 and v513 at lanes 0, 1 and 16, chooses VGPR bank fields, and optionally executes **S_TRAP 3**. A faulty CWSR entry sequence saves lane zero from the SRC0-selected bank and restores it into the DST-selected bank. Different banks expose corruption while MODE, EXEC and the trap return PC can all look intact. The 14 cases are seven bank settings, each without/with a shader trap.

The primary command uses a derived portable Python adapter around the **byte-identical executed host** in `exact/run_sentinel_host_v1.py`. It changes only runtime library paths/pins, device ordinal and the unused training-environment requirement; the shader, packet ABI, launch dimensions, case order, HIP transport and host code remain exact. The adapter adds a strict independent pass/fail oracle. Its current on-device validation status is supplied separately by the root ticket evidence; CPU preparation alone does not claim a new GPU run.

## Check, run and expected result

From this directory, check all exact source/native pins, ELF symbols/return sites and both supplied observed packet sets without loading HIP:

```sh
python3 -B run_repro.py --check-only
```

On an idle MI450/gfx1250 with CWSR enabled, supply your local runtime paths and a fresh output directory:

```sh
env -u LD_PRELOAD python3 -B run_repro.py --suite sentinel --expect original \
  --hip-library /opt/rocm/lib/libamdhip64.so \
  --hsa-library /opt/rocm/lib/libhsa-runtime64.so.1 \
  --device 0 --output original-result
```

After the driver team loads its corrected driver through its normal procedure, repeat with `--expect fixed --output fixed-result`. The runner does not load, unload, replace or reset the driver. It stops on a HIP error and retains its output. If a GPU operation stalls, preserve its process/logs and use the established machine recovery procedure; this small adapter does not implement timeout or automatic recovery.

| `s_set_vgpr_msb` low byte | Selected source / destination | Original, no trap | Original, S_TRAP 3 | Fixed, both cases |
| --- | --- | --- | --- | --- |
| `00` | v1 / v1 | unchanged | unchanged | unchanged |
| `41` | v257 / v257 | unchanged | unchanged | unchanged |
| `45` | v257 / v257 | unchanged | unchanged | unchanged |
| `82` | v513 / v513 | unchanged | unchanged | unchanged |
| `05` | v257 / v1 | unchanged | v1 lane0 `11000000 → 22000000` | unchanged |
| `81` | v257 / v513 | unchanged | v513 lane0 `33000000 → 22000000` | unchanged |
| `85` | v257 / v513 | unchanged | v513 lane0 `33000000 → 22000000` | unchanged |

Values are hexadecimal. These setter bytes are **not** raw MODE[19:12] bytes. Source bank is setter `& 3`; destination bank is `(setter >> 6) & 3`. The raw MODE bytes for the rows above are `00,05,15,0a,14,06,16`.

`--expect original` returns exit 0 only for the exact three predicted changes and eleven unchanged cases, with every other check valid. `--expect fixed` returns exit 0 only for all 14 unchanged cases. Other numerical/structural outcomes return nonzero. `verdict.json` lists the exact failing fields. A HIP transport success alone is not numerical success.

Both verdicts require all packet constants/reserved words, all nine initial/final tags, full MODE equality, full-wave EXEC equality, trap-enable status, all seven 64-bit returned trap PCs, and the intended PC's location in the exact native image with one consistent module load bias. The no-trap cases do not require arbitrary pre-existing TTMP values to equal a shader-trap return PC. Device prefill is freshly read back before every launch.

## Build and provenance

No build is needed for a run. To rebuild the assembly using a gfx1250-capable AMD LLVM toolchain:

```sh
python3 -B build_repro.py --llvm-bin /opt/rocm/llvm/bin --output new-build
```

This executes only `llvm-mc`, `ld.lld` and `llvm-objdump`. The CPU rebuild here matched all 14 kernel instruction streams, native return sites, `.text`, `.rodata` and AMDGPU metadata. Its only whole-ELF difference is 37 bytes in the nonallocated linker version comment. `CPU_BUILD_v2/{BUILD_RESULT,LINKER_COMMENT_DELTA}.json` records this precisely. The runner continues to use the exact previously executed image, SHA-256 `4f840df38a5d815abe50824f37dedbde4bf48fffdec6ca84530cd631f69e5484`.

`EXACT_ASSETS.json` records each unmodified asset and its original source path/hash. The original configuration retains its historical paths as provenance; the adapter does not require or load those paths. `provenance/final_sentinel_shader_instruction_ABI_independent_review_v1.json` is the existing independent native ISA/ABI certificate. The descriptor allocates enough VGPRs to cover v513; scratch and LDS are zero; kernarg is one 64-bit output pointer; only lanes 0/1/16 are read as tags, and only lane zero stores the report.

`evidence/witnesses.json` joins 28 packaged 160-byte raw packets to their original result-file/output hashes: current original driver (`EADFD8EC13CF3E6B329194E`, September 28) and saved corrected driver (`AD83C153B9701570F12964E`, September 23). Driver identity/boot/lifecycle are established by the accompanying ticket's independently checked evidence; a packet alone cannot identify the loaded driver. `--check-only` decodes every packet again and rejects the opposite driver expectation.

`CPU_VERIFICATION.json` records 34 CPU checks including actual witness acceptance, opposite-oracle rejection, malformed packet fields, full MODE/EXEC, high-half return PC, and missing/duplicate/reordered cases. CPU build/check directories contain preparation evidence and are not required runtime dependencies. The minimum runtime bundle is `run_repro.py`, `exact/`, and `evidence/`.

The complementary ordinary-shader inline matrix is described by the ticket evidence: 256 raw MODE bank bytes × eight EXEC masks × original/corrected/no-prologue = 6,144 calls (2,048 per arm). It isolates the entry-sequence defect without installing a handler or explicitly executing S_TRAP. Its original arm produced all 1,536 predicted unequal-bank copies; the other two arms preserved all tags. That separate test is not a claim about privileged workaround timing, external interrupts, or all-training stability.

## Original A0 runtime paths for the root validation

The existing verified container libraries are:

```text
/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib/libamdhip64.so.7
/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib/libhsa-runtime64.so.1
```

Root stages a byte-verified copy of this portable directory under the existing `/data/mlperf_dlrm_v4` mount, retains the reviewed pre-HIP process receipt, and runs the command above with those two paths and a fresh result directory. The existing serial lock, current exact kernel policy, before/after idle/kernel gates and actual process exit/reaping remain root-owned. No extra training environment is required by the adapter. It retains the original host's rejection of LD_PRELOAD and LINEAR_DW_* injection variables.
