# gfx1250 CWSR entry register preservation candidate

`cwsr_gfx1250_entry_bank_fix.patch` changes the AMDGPU driver assembly and its
generated `cwsr_trap_gfx12_1_0_hex` array. It targets the installed source package
`amdgpu-7.1.1-2389086.24.04`; apply it to an isolated copy of that package.
The recommendation training repository does not contain that driver source.

The non-wave-start entry workaround reads encoded `v1` through SRC0 but writes
through DST. When these bank fields differ, it preserves the wrong logical
register. The patch temporarily sets DST equal to SRC0 for zeroing and restore,
then restores DST. It uses the existing TTMP2:3 scratch registers on either side
of their zero-address prefetch use. It preserves EXEC and all bank fields and
includes two vector no-ops before each MODE write and two scalar wait states
afterward. No floating-point arithmetic or training configuration is changed.

The complete original handler rebuild matches all **5,656 installed bytes**.
The generated candidate is **5,736 bytes**, below the driver's 6,144-byte limit.
Its changes are the 48-byte entry replaced by 128 bytes and the initial
restore branch moved by 80 bytes. The entire shifted suffix is unchanged.
Local SP3 requires the two existing prefetch instructions to be expressed as
their exact original words; this compatibility transformation is applied only
to build copies. It reproduces the original image before generating the new one.

The September 21 B0 userspace comparison completed all **512 cases**: the
original entry sequence produces 192 predicted cross-bank copies and 64 clean
equal-bank controls; the candidate preserves all sampled tags in all 256 bank
settings. MODE, EXEC, completion words and full allocation guards also pass.
The shader substitutes ordinary SGPRs for trap scratch registers and injects
no trap. It therefore validates the sequence's register behavior, not the live
privileged handler, full context save/restore, or the original training NaN.

Source fingerprints before applying the patch:

- Assembly SHA-256: `5e0e339cb988120d99fbccf2869bfab29e0a143bd110ce6f00cfdc495d5bf69d`.
- Header SHA-256: `006a44ea767d581ecc84913ec8bdcd68476471510a1ccd9923b00efc3b8463ef`.
- Candidate handler bytes SHA-256: `1fda473155dc0a850039d2a018378facbb789cc0488891b26cdf18bbefef80b0`.

The [September 21 investigation record](../nan_triage_resume_20260921.md)
links the standalone builder, independent assembly review, raw GPU results and
module-build evidence. The installed driver is unchanged. Driver installation
and host reboot are separate from preparing this candidate and require an
explicit exception to the restored no-reset/reboot rule. The user authorized
publication of this documentation and candidate patch; that authorization
does not include driver installation or activation.
