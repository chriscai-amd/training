#!/usr/bin/env python3
"""T8: make the bf16 NT GridBased catalog itself pick sol 103 (MT256x128x64, SK0) for the DLRMv4
NT weight-gradient GEMMs at any jagged K (=T), instead of relying on the exact-(m,n,k)
HIPBLASLT_TUNING_OVERRIDE_FILE (which never fires with jagged K).

CPU-only. Never touches a GPU, /opt, or the v1 directory (read-only input).

Modes
  python -I make_v2.py                 build /home/chcai/training/arbor_libs/rocm-10.2_nnonly_v2
                                       (refuses if it exists; --force rebuilds it via a temp dir)
  python -I make_v2.py --stock         build /home/chcai/t8_work/stock_t8/ from the stock-image NT
                                       catalog copied to /home/chcai/t8_work/stock_catalog/gfx1250
                                       (catalog file + README only; not wired anywhere)
  python -I make_v2.py --check FILE.dat.zlib KERNELS.co   validate an existing edited catalog
  python -I make_v2.py --drop 24x256 --out /home/chcai/training/arbor_libs/rocm-10.2_nnonly_v3
                                       variant without some targets (e.g. if the single-GEMM check
                                       shows G27 (24,256) slower on 103); dropped shapes stay on 102

Table semantics (ROCm/rocm-libraries 5911365c, PropertyMatching.hpp GridBased binary path, default
env): key = [M, N, batch, K]; M = first table M >= m (clamped to max M), then N = first N >= n
inside that M block (clamped), then batch likewise, then nearest K inside that group (ties -> larger
K). See gridmatch.py. Each added (M,N,1) group holds exactly ONE entry, so K never matters.
"""
import argparse
import os
import shutil
import subprocess
import sys
import zlib

sys.path.insert(0, "/home/chcai/t8_work")
import gridmatch as g  # noqa: E402
import msgpack  # noqa: E402

V1 = "/home/chcai/training/arbor_libs/rocm-10.2_nnonly_v1"
V2 = "/home/chcai/training/arbor_libs/rocm-10.2_nnonly_v2"
STOCK_SRC = "/home/chcai/t8_work/stock_catalog/gfx1250"
STOCK_OUT = "/home/chcai/t8_work/stock_t8"
GFX_REL = "hipblaslt/library/gfx1250"
NT = "TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Ailk_Bjlk_Cijk_Dijk_gfx1250"
NT_DAT = NT + ".dat.zlib"
NT_CO = NT + ".co"
HOST_CO = "/opt/rocm-10.2/lib/hipblaslt/library/gfx1250/" + NT_CO

# hipBLASLt (m, n) of the DLRMv4 bf16 NT (transA=N, transB=T, batch 1, K=T) weight gradients
ALL_TARGETS = [(2048, 512),   # G20
           (512, 1536),   # G19
           (1024, 256),   # G29
           (512, 256),    # G31
           (24, 256)]     # G27
#          (256, 512)     # G25 -- already 103 via the stub's [256,256,1,512] entry
TARGETS = list(ALL_TARGETS)   # narrowed by --drop
LIBS = "/home/chcai/training/arbor_libs"
SLOW, FAST = 102, 103     # MT32x16x32 SK0 / MT256x128x64 SK0 (present in both v1 and stock)
SENT = 1 << 20            # 1048576: "above anything real" guard value for M and N
K_DESIGN = (1_000_000, 3_200_000)


def added_entries(stub):
    """Entries to add on top of the stub. Per target M:
         [M-1, 1, 1, 1] -> 102   keeps m in (previous table M, M-1] on its old (102) result
         [M, n-1, 1, 1] -> 102   keeps n in (previous N, n-1] on 102
         [M, n, 1, 1]   -> 103   the target
         [M, SENT, 1, 1]-> 102   keeps n > largest target n at this M on 102
       plus [SENT, 1, 1, 1] -> 102 so m above the largest target M keeps landing on a 102 block
       (without it, m > 2048 would clamp into the M=2048 block)."""
    have = {tuple(e["key"]) for e in stub}
    rows = {}
    for m in sorted({m for m, _ in TARGETS}):
        rows[(m - 1, 1, 1, 1)] = SLOW
        for mm, n in sorted(TARGETS):
            if mm == m:
                rows[(m, n - 1, 1, 1)] = SLOW
                rows[(m, n, 1, 1)] = FAST
        rows[(m, SENT, 1, 1)] = SLOW
    rows[(SENT, 1, 1, 1)] = SLOW
    clash = sorted(k for k in rows if k in have)
    if clash:
        raise SystemExit("added keys collide with stub keys: %s" % clash)
    return [{"key": list(k), "index": v} for k, v in sorted(rows.items())]


def edit_catalog(cat):
    """Return (new_catalog, added_entries). Only the GridBasedMatching table changes."""
    rows = g.gridbased_rows(cat)
    if len(rows) != 1:
        raise SystemExit("expected exactly one GridBasedMatching row, got %d" % len(rows))
    pos, props, stub, dist = rows[0]
    if props != ["FreeSizeA", "FreeSizeB", "BatchSize", "BoundSize"] or dist != "GridBased":
        raise SystemExit("unexpected GridBased row layout: %s %s" % (props, dist))
    sol_idx = {s["index"] for s in cat["solutions"]}
    if not {SLOW, FAST} <= sol_idx:
        raise SystemExit("solutions %s/%s not in catalog (%s)" % (SLOW, FAST, sorted(sol_idx)))
    add = added_entries(stub)
    # same entry layout as the stub ({'key': [...], 'index': int}, no 'speed'), kept sorted by key
    # like the shipped file (the loader std::sorts anyway; keys are unique so order is defined).
    new_table = sorted([dict(e) for e in stub] + add, key=lambda e: tuple(e["key"]))
    cat["library"]["rows"][pos]["library"]["table"] = new_table
    return cat, add


def pack(cat):
    return zlib.compress(msgpack.packb(cat, use_bin_type=True), 9)


def unbundle(co_path):
    """[(target_id, elf_bytes)] from a .co: plain ELF, __CLANG_OFFLOAD_BUNDLE__, or CCOB v2/v3
    (zstd/zlib-compressed bundle; v3 header = magic4 ver2 method2 total8 uncompressed8 hash8)."""
    import struct
    d = open(co_path, "rb").read()
    if d[:4] == b"CCOB":
        ver, method = struct.unpack("<HH", d[4:8])
        hdr = 32 if ver >= 3 else 24
        if method == 1:
            import zstandard
            d = zstandard.ZstdDecompressor().decompress(d[hdr:], max_output_size=1 << 32)
        else:
            d = zlib.decompress(d[hdr:])
    if d[:4] == b"\x7fELF":
        return [("elf", d)]
    if d[:24] != b"__CLANG_OFFLOAD_BUNDLE__":
        raise SystemExit("unknown code object format: %r" % d[:8])
    n, = struct.unpack("<Q", d[24:32])
    off, out = 32, []
    for _ in range(n):
        o, sz, il = struct.unpack("<QQQ", d[off:off + 24])
        tid = d[off + 24:off + 24 + il].decode()
        off += 24 + il
        out.append((tid, d[o:o + sz]))
    return out


def kernel_symbols(co_path):
    """(decompressed gfx1250 ELF bytes, set of FUNC/OBJECT symbol names from llvm-readelf --syms)."""
    import tempfile
    blobs = [b for tid, b in unbundle(co_path) if tid == "elf" or "gfx1250" in tid]
    if not blobs:
        raise SystemExit("no gfx1250 code object in %s" % co_path)
    syms = set()
    with tempfile.TemporaryDirectory(dir="/home/chcai/t8_work") as td:
        for i, b in enumerate(blobs):
            fp = os.path.join(td, "e%d.o" % i)
            with open(fp, "wb") as f:
                f.write(b)
            out = subprocess.run(["llvm-readelf", "--syms", "-W", fp], capture_output=True,
                                 text=True, timeout=300).stdout
            for line in out.splitlines():
                parts = line.split()
                if len(parts) >= 8 and parts[3] in ("FUNC", "OBJECT"):
                    syms.add(parts[7])
    return b"".join(blobs), syms


def validate(dat_path, co_path, ref_cat=None, log=print):
    raw = open(dat_path, "rb").read()
    dec = zlib.decompress(raw)
    cat = msgpack.unpackb(dec, raw=False, strict_map_key=False)
    ok = True

    def chk(cond, msg):
        nonlocal ok
        log(("  OK   " if cond else "  FAIL ") + msg)
        ok &= bool(cond)

    chk(msgpack.packb(cat, use_bin_type=True) == dec, "msgpack unpack->repack is byte-identical")
    chk(zlib.decompress(zlib.compress(dec, 9)) == dec, "zlib round-trip")
    if ref_cat is not None:
        chk(cat["solutions"] == ref_cat["solutions"], "solutions list identical to the input catalog")
        other = [(i, r) for i, r in enumerate(cat["library"]["rows"])]
        ref_rows = ref_cat["library"]["rows"]
        chk(len(other) == len(ref_rows) and all(
            r["predicate"] == q["predicate"] and
            {k: v for k, v in r["library"].items() if k != "table"} ==
            {k: v for k, v in q["library"].items() if k != "table"}
            for (_, r), q in zip(other, ref_rows)),
            "row predicates/properties/distance unchanged (only the table differs)")
    (pos, props, table, dist), = g.gridbased_rows(cat)
    keys = [tuple(e["key"]) for e in table]
    chk(dist == "GridBased" and len(table) > 0, "distance GridBased, table non-empty (%d entries)"
        % len(table))
    chk(all(len(k) == len(props) == 4 for k in keys), "every key has 4 values")
    chk(all(isinstance(x, int) and x > 0 for k in keys for x in k), "keys are positive ints")
    chk(len(set(keys)) == len(keys), "keys unique")
    chk(keys == sorted(keys), "table sorted by key")
    chk(all(set(e.keys()) == {"key", "index"} for e in table), "entry layout {key,index}")
    # no block whose last N group holds several batch values (forward-walk past-end hazard)
    blocks = {}
    for k in keys:
        blocks.setdefault(k[0], {}).setdefault(k[1], set()).add(k[2])
    chk(all(len(ng[max(ng)]) == 1 for ng in blocks.values()), "no multi-batch last N group")
    sols = {s["index"]: s for s in cat["solutions"]}
    used = sorted({e["index"] for e in table})
    chk(all(i in sols for i in used), "every referenced index %s is in solutions %s"
        % (used, sorted(sols)))
    data, syms = kernel_symbols(co_path)
    for i in used:
        s = sols[i]
        kn = s["kernelName"]
        chk(kn.encode() in data, "sol %d kernel name in decompressed %s: %s..." % (
            i, os.path.basename(co_path), kn[:70]))
        chk(kn in syms and kn + ".kd" in syms,
            "sol %d kernel + .kd descriptor are gfx1250 ELF symbols (llvm-readelf --syms)" % i)
        mt = s["sizeMapping"]["macroTile"]
        log("         sol %d MT%dx%dx%d SK=%s GSU=%s" % (
            i, mt[0], mt[1], s["sizeMapping"]["depthU"],
            s["sizeMapping"].get("streamK"), s["sizeMapping"].get("globalSplitU")))
    return ok, cat


def selection_check(new_cat, old_cat, log=print):
    """Real-semantics check with gridmatch: targets -> 103 across K, workload unchanged otherwise."""
    acc = g.nt_size_accepts(new_cat)
    old = g.gridbased_rows(old_cat)[0][2]
    new = g.gridbased_rows(new_cat)[0][2]
    ks = [K_DESIGN[0], 1_400_000, 1_590_001, 2_097_152, 2_533_005, K_DESIGN[1]]
    ok = True
    for m, n in TARGETS + [(256, 512)]:
        got = {g.select(new, (m, n, 1, k), None, acc) for k in ks}
        good = got == {FAST}
        ok &= good
        log("  %s  (%d,%d) K=T -> %s" % ("OK  " if good else "FAIL", m, n, sorted(got)))
    for m, n in [t for t in ALL_TARGETS if t not in TARGETS]:
        got = {g.select(new, (m, n, 1, k), None, acc) for k in ks}
        ok &= got == {SLOW}
        log("  %s  dropped (%d,%d) K=T -> %s" % ("OK  " if got == {SLOW} else "FAIL", m, n, sorted(got)))
    for prob in [(516, 512, 1, 1024), (512, 512, 8, 1024)]:
        o, p = g.select(old, prob, None, acc), g.select(new, prob, None, acc)
        ok &= o == p
        log("  %s  %s -> %s (was %s)" % ("OK  " if o == p else "FAIL", prob, p, o))
    return ok


README_V2 = """\
ARM_LIB_DIR for roadmap L1 stage 1 + T8 (NN only, NT wgrads pinned by the catalog): identical to rocm-10.2_nnonly_v1
(ROCm 10.2 (/opt/rocm-10.2) libhipblaslt 1.5 + tensilelite-host + origami + rocroller + libomp + sysdeps_z, gfx1250
kernel folder symlinked) EXCEPT the bf16 NT (Ailk_Bjlk BB_BB_HA_Bias_SAV) catalog .dat, which is v1's filtered copy
(Origami Prediction row removed, solutions pruned to the GridBased stub's sol 102 MT32x16x32 / 103 MT256x128x64, SK0)
PLUS {nadd} new GridBased table entries ({nold} -> {nnew}). No solution, kernel, predicate, row or property changed; .co untouched.
Why: the stub sends every DLRMv4 NT weight gradient (K = T = jagged token count, 1.59-2.53 M) to sol 102 (MT32x16x32,
slow) except (256,512). The bench pin to 103 gave 684.7 -> 502.7 ms/step at K=2,097,152, but the
HIPBLASLT_TUNING_OVERRIDE_FILE route matches exact (m,n,k) and never fires with jagged K. GridBased matching
(rocm-libraries 5911365c PropertyMatching.hpp, default env) = M ceiling, then N ceiling in that M block, then batch,
then nearest K; each added (M,N,1) group has ONE entry, so the choice is K-independent (designed for K 1.0M-3.2M).
Added entries ([M, N, batch, K] -> sol); "guard" rows keep neighbouring shapes on their old result (102):
{entries}
Targets switched to 103 in this build: {targets}{dropped}
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
"""


def fmt_entries(add):
    return "\n".join("  %-22s -> %d%s" % (str(e["key"]), e["index"],
                                         "   (TARGET)" if e["index"] == FAST else "   (guard)")
                     for e in add)


def target_text():
    names = {(2048, 512): "G20", (512, 1536): "G19", (1024, 256): "G29", (512, 256): "G31", (24, 256): "G27"}
    t = " ".join("(%d,%d) %s [%d WG on 103]" % (m, n, names[(m, n)], -(-m // 256) * -(-n // 128))
                 for m, n in TARGETS)
    d = [x for x in ALL_TARGETS if x not in TARGETS]
    return t, ("\nDropped (stay on 102): " + " ".join("(%d,%d)" % x for x in d)) if d else ""


def build_v2(force, out=V2):
    out = os.path.abspath(out)
    if os.path.dirname(out) != LIBS or not os.path.basename(out).startswith("rocm-10.2_nnonly_v") \
            or os.path.basename(out) == "rocm-10.2_nnonly_v1":
        raise SystemExit("--out must be %s/rocm-10.2_nnonly_v<N>, N >= 2" % LIBS)
    if os.path.lexists(out):
        if not force:
            raise SystemExit("%s exists; pass --force to rebuild it" % out)
    tmp = "/home/chcai/t8_work/v2_build.tmp"     # staged outside arbor_libs, moved in at the end
    if os.path.lexists(tmp):
        shutil.rmtree(tmp)
    src_dat = os.path.join(V1, GFX_REL, NT_DAT)
    if os.path.islink(src_dat):
        raise SystemExit("v1 NT catalog unexpectedly a symlink")
    old_cat = g.load_catalog(src_dat)
    new_cat, add = edit_catalog(g.load_catalog(src_dat))
    n_links = 0
    for root, dirs, files in os.walk(V1, followlinks=False):
        rel = os.path.relpath(root, V1)
        dst_root = os.path.join(tmp, rel) if rel != "." else tmp
        os.makedirs(dst_root, exist_ok=True)
        for name in sorted(dirs + files):
            src = os.path.join(root, name)
            dst = os.path.join(dst_root, name)
            if os.path.islink(src):
                os.symlink(os.readlink(src), dst)       # same target string (absolute into /opt)
                n_links += 1
                if name in dirs:
                    dirs.remove(name)                    # don't descend symlinked dirs
            elif os.path.isdir(src):
                continue
            elif os.path.relpath(src, V1) == os.path.join(GFX_REL, NT_DAT):
                with open(dst, "wb") as f:
                    f.write(pack(new_cat))
            elif name == "README.txt" and rel == ".":
                pass                                     # written below
            else:
                raise SystemExit("unexpected regular file in v1: %s" % src)
    dat = os.path.join(tmp, GFX_REL, NT_DAT)
    print("validate %s" % dat)
    ok, _ = validate(dat, HOST_CO, ref_cat=old_cat)
    print("selection check (gridmatch, real semantics)")
    ok &= selection_check(new_cat, old_cat)
    if not ok:
        raise SystemExit("validation failed; left %s for inspection" % tmp)
    import hashlib
    sha = hashlib.sha256(open(dat, "rb").read()).hexdigest()
    old_n = len(g.gridbased_rows(old_cat)[0][2])
    with open(os.path.join(tmp, "README.txt"), "w") as f:
        tt, dd = target_text()
        f.write(README_V2.format(nadd=len(add), nold=old_n, nnew=old_n + len(add),
                                 entries=fmt_entries(add), targets=tt, dropped=dd))
        f.write("sha256 %s  %s/%s\n" % (sha, GFX_REL, NT_DAT))
    # structure parity with v1
    v1_set = sorted(os.path.relpath(os.path.join(r, x), V1)
                    for r, d, fs in os.walk(V1) for x in d + fs)
    v2_set = sorted(os.path.relpath(os.path.join(r, x), tmp)
                    for r, d, fs in os.walk(tmp) for x in d + fs)
    if v1_set != v2_set:
        raise SystemExit("structure mismatch v1 vs v2: %s" % (set(v1_set) ^ set(v2_set)))
    if os.path.lexists(out):
        shutil.rmtree(out)
    shutil.move(tmp, out)
    print("built %s: %d symlinks, 2 real files (README.txt, %s)" % (out, n_links, NT_DAT))
    print("added entries:\n" + fmt_entries(add))


def build_stock(force):
    out_dir = os.path.join(STOCK_OUT, "gfx1250")
    if os.path.lexists(STOCK_OUT) and not force:
        raise SystemExit("%s exists; pass --force" % STOCK_OUT)
    os.makedirs(out_dir, exist_ok=True)
    src = os.path.join(STOCK_SRC, NT_DAT)
    old_cat = g.load_catalog(src)
    new_cat, add = edit_catalog(g.load_catalog(src))
    dst = os.path.join(out_dir, NT_DAT)
    with open(dst, "wb") as f:
        f.write(pack(new_cat))
    print("validate %s" % dst)
    ok, _ = validate(dst, os.path.join(STOCK_SRC, NT_CO), ref_cat=old_cat)
    print("selection check (gridmatch, real semantics)")
    ok &= selection_check(new_cat, old_cat)
    if not ok:
        raise SystemExit("validation failed")
    import hashlib
    sha = hashlib.sha256(open(dst, "rb").read()).hexdigest()
    with open(os.path.join(STOCK_OUT, "README.txt"), "w") as f:
        f.write(
            "M0+T8 variant (NOT wired anywhere). gfx1250/%s is the stock image's bf16 NT catalog\n"
            "(recommendation-gfx1250-20260910:triton-7ff97e, _rocm_sdk_libraries_gfx1250, hipBLASLt 7.14 nightly\n"
            "2026-06-25; copied out with docker create/cp/rm) with the SAME %d GridBased entries added as\n"
            "rocm-10.2_nnonly_v2 (stock table == v1 stub table; stock sols 102/103 = same tiles, own kernel names, present\n"
            "in the stock .co). To use: copy the image's whole .../hipblaslt/library/gfx1250 dir (or symlink every file,\n"
            "with link targets written as CONTAINER paths, i.e. /opt/venv/lib/python3.12/site-packages/\n"
            "_rocm_sdk_libraries_gfx1250/lib/hipblaslt/library/gfx1250/<file>, not host paths) into a new dir, replace this\n"
            "one file, point HIPBLASLT_TENSILE_LIBPATH at it (or bind-mount this file over the image's copy).\n"
            "Kernel note: the stock sol 103 is the 7.14-built MT256x128x64 SK0 kernel (also 676 VGPRs, A0 DW-corruption\n"
            "kernel class, see T8_REPORT.md); the M2 bench measured the 10.2 build, so this arm's per-shape speed is unmeasured.\n"
            "Targets: %s%s\n"
            "Added entries:\n%s\nsha256 %s  gfx1250/%s\n" % (NT_DAT, len(add), *target_text(), fmt_entries(add), sha,
                                                         NT_DAT))
    print("built %s" % dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--stock", action="store_true")
    ap.add_argument("--check", nargs=2, metavar=("DAT", "CO"))
    ap.add_argument("--drop", action="append", default=[], metavar="MxN",
                    help="leave this target on 102 (repeatable), e.g. --drop 24x256")
    ap.add_argument("--out", default=V2, help="output lib dir (default %s)" % V2)
    a = ap.parse_args()
    for d in a.drop:
        t = tuple(int(x) for x in d.lower().split("x"))
        if t not in TARGETS:
            raise SystemExit("--drop %s: not a target %s" % (d, ALL_TARGETS))
        TARGETS.remove(t)
    if not TARGETS:
        raise SystemExit("nothing left to switch")
    if a.check:
        ok, _ = validate(*a.check)
        sys.exit(0 if ok else 1)
    if a.stock:
        build_stock(a.force)
    else:
        build_v2(a.force, a.out)


if __name__ == "__main__":
    main()
