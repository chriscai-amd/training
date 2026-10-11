#!/usr/bin/env python3
"""
Faithful Python re-implementation of TensileLite's GridBased MatchingLibrary
solution selection (hipBLASLt, ROCm 10.2 build).

Source of truth (ROCm/rocm-libraries, commit 5911365c1f5b81f6b2a9ad9515f731b4cfb979a4,
2026-09-23 -- the exact git revision string embedded in /opt/rocm-10.2/lib/libhipblaslt.so.1.5):
  projects/hipblaslt/tensilelite/include/Tensile/PropertyMatching.hpp
      DistanceMatchingTable<..., GridBasedDistance> specialisation   L898-1433
        findBestKeyMatch / findTopKeyMatch                            L944-962
        KD-tree path (TENSILE_GRIDBASED_KDTREE)                       L981-1110
        binary ("grid") path                                          L1112-1430
      KDTree (build/query)                                            L118-288
  projects/hipblaslt/tensilelite/include/Tensile/Distance.hpp        GridBasedDistance L371-404
  projects/hipblaslt/tensilelite/include/Tensile/Serialization/MatchingLibrary.hpp
      table load + std::sort + kd-tree build                          L58-97
      entry {key, speed?, index} -> SingleSolutionLibrary            L246-306
      TENSILE_METRIC override of the distance                         L164-167
  projects/hipblaslt/tensilelite/src/Debug.cpp                       env parsing L259-273
  projects/hipblaslt/tensilelite/include/Tensile/Debug.hpp           defaults    L181-184
The algorithm is byte-identical in origin/gpuep-releases/therock-7.14 (stock image, M0).

Key layout for the catalogs here: properties [FreeSizeA, FreeSizeB, BatchSize, BoundSize]
 = [M, N, batch, K] in hipBLASLt (m, n, batch_count, k) terms (any layout; for
 Contraction_l_Ailk_Bjlk_Cijk_Dijk = transA N / transB T).

Public API
  select(table_entries, problem, env=None, accepts=None) -> solution index or None
  select_top(table_entries, problem, env=None, accepts=None, num_solutions=1) -> [indices]
    table_entries : iterable of {'key': [..], 'index': int, ('speed': float)} (catalog rows)
                    or (key, index) tuples
    problem       : (m, n, batch, k) tuple, or dict with m/n/batch/k
    env           : dict of environment variables (TENSILE_GRIDBASED_KDTREE,
                    TENSILE_GRIDBASED_BATCH_EXP, GRIDBASED_TOPSOLS, TENSILE_METRIC)
    accepts       : callable(index, problem_tuple) -> bool modelling the solution's own
                    hardware/problem/task predicates (the "transform" in C++). Default:
                    every solution accepts. NB: it always receives the ORIGINAL problem,
                    as in C++ (the transform lambda captures the problem, not the key).
"""
import struct
import sys
import zlib

INT32_MIN, INT32_MAX = -(1 << 31), (1 << 31) - 1


class UndefinedBehaviour(RuntimeError):
    """The C++ would dereference table.end() (or similar) here."""


# ----------------------------------------------------------------------------- env
def c_strtol0(s):
    """strtol(s, nullptr, 0): leading ws, sign, 0x hex / 0 octal / decimal, stops at junk."""
    s = s.lstrip(" \t\n\r\f\v")
    sign, i = 1, 0
    if i < len(s) and s[i] in "+-":
        sign = -1 if s[i] == "-" else 1
        i += 1
    base = 10
    if s[i:i + 2].lower() == "0x" and len(s) > i + 2 and s[i + 2] in "0123456789abcdefABCDEF":
        base, i = 16, i + 2
    elif s[i:i + 1] == "0":
        base = 8
    digits = "0123456789abcdef"[:base]
    v = 0
    while i < len(s) and s[i].lower() in digits:
        v = v * base + digits.index(s[i].lower())
        i += 1
    return sign * v


def parse_env(env):
    """Debug::Debug() (src/Debug.cpp L259-273); defaults Debug.hpp L181-184."""
    env = env or {}
    cfg = {"kdtree": False, "batch_exp": False, "topsols": 1, "metric": ""}
    if env.get("GRIDBASED_TOPSOLS") is not None:
        cfg["topsols"] = c_strtol0(env["GRIDBASED_TOPSOLS"])  # read, but unused anywhere
    if env.get("TENSILE_GRIDBASED_KDTREE") is not None:
        cfg["kdtree"] = c_strtol0(env["TENSILE_GRIDBASED_KDTREE"]) != 0
    if env.get("TENSILE_GRIDBASED_BATCH_EXP") is not None:
        cfg["batch_exp"] = c_strtol0(env["TENSILE_GRIDBASED_BATCH_EXP"]) != 0
    if env.get("TENSILE_METRIC") is not None:
        cfg["metric"] = env["TENSILE_METRIC"]
    return cfg


# ----------------------------------------------------------------------------- helpers
def lower_bound(seq, first, last, val, comp):
    """libstdc++ std::lower_bound (exact probe sequence, so non-partitioned ranges match too)."""
    length = last - first
    while length > 0:
        half = length >> 1
        mid = first + half
        if comp(seq[mid], val):
            first = mid + 1
            length = length - half - 1
        else:
            length = half
    return first


def upper_bound(seq, first, last, val, comp):
    length = last - first
    while length > 0:
        half = length >> 1
        mid = first + half
        if comp(val, seq[mid]):
            length = half
        else:
            first = mid + 1
            length = length - half - 1
    return first


def i32(x):
    x &= 0xFFFFFFFF
    return x - (1 << 32) if x & 0x80000000 else x


def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def _norm_entries(table_entries):
    out = []
    for e in table_entries:
        if isinstance(e, dict):
            out.append((tuple(int(v) for v in e["key"]), int(e["index"]), e.get("speed")))
        else:
            k, idx = e[0], e[1]
            sp = e[2] if len(e) > 2 else None
            out.append((tuple(int(v) for v in k), int(idx), sp))
    return out


def sort_table(entries):
    """Serialization/MatchingLibrary.hpp L66-70: sort by key (lexicographic), ties speed desc.
    'speed' is optional in the file and is left uninitialised in C++ when absent -> duplicate
    keys without speed have unspecified order. We warn and keep file order for those."""
    keys = [e[0] for e in entries]
    if len(set(keys)) != len(keys):
        dup = sorted({k for k in keys if keys.count(k) > 1})
        if any(e[2] is None for e in entries if e[0] in dup):
            sys.stderr.write("gridmatch: WARNING duplicate keys without 'speed' -> C++ order "
                             "unspecified: %s\n" % dup[:5])
    # python sort is stable; use (key, -speed)
    return sorted(entries, key=lambda e: (e[0], -(e[2] if e[2] is not None else 0.0)))


def _problem_key(problem, nprops):
    if isinstance(problem, dict):
        t = (problem["m"], problem["n"], problem.get("batch", 1), problem["k"])
    else:
        t = tuple(problem)
    t = tuple(int(v) for v in t)
    if len(t) != nprops:
        raise ValueError("problem key length %d != number of properties %d" % (len(t), nprops))
    return t


def grid_distance(p1, p2):
    """Distance.hpp L389-397: |K - gridK|, K = key[3] if len>3 else key[2]. M/N/batch ignored."""
    gk = p2[3] if len(p2) > 3 else p2[2]
    k = p1[3] if len(p1) > 3 else p1[2]
    return abs(float(k) - float(gk))


# ----------------------------------------------------------------------------- binary path
def _binary(table, key_orig, accepts, problem_orig, num_solutions, batch_exp, trace):
    """PropertyMatching.hpp L1112-1430 (default path)."""
    best = []
    if not table:
        return best
    n = len(table)
    END = n
    key = list(key_orig)

    def kk(i):
        if i >= n:
            raise UndefinedBehaviour("dereference of table.end()")
        return table[i][0]

    compM = lambda e, v: e[0][0] < v
    compN = lambda e, v: e[0][1] < v
    compB = lambda e, v: e[0][2] < v

    M_lo = M_up = N_lo = N_up = B_lo = B_up = 0
    batch_it = 0
    base_n = 0
    step_n = 0
    i = 1
    while True:                                  # L1141-1148
        if i * (i + 1) // 2 >= num_solutions:
            base_n = i
            break
        i += 1
    first_find = True
    best_match = None

    while B_up != END and len(best) < num_solutions:   # L1150
        best_dist = float("inf")
        this_match = False
        if step_n == base_n and base_n != 1:            # L1161-1171
            step_n = 1
            base_n -= 1
            start = M_up
        else:
            step_n += 1
            start = N_up
        lastM = table[END - 1][0][0]
        M_lo = lower_bound(table, start, END, min(key[0], lastM), compM)            # L1173
        M_up = lower_bound(table, M_lo, END, min(kk(M_lo)[0] + 1, lastM + 1), compM)  # L1183
        maxN = kk(M_up - 1)[1]
        N_lo = lower_bound(table, M_lo, M_up, min(key[1], maxN), compN)            # L1196
        N_up = lower_bound(table, N_lo, M_up, min(kk(N_lo)[1] + 1, maxN + 1), compN)  # L1209
        maxB = kk(N_up - 1)[2]
        B_lo = lower_bound(table, N_lo, N_up, min(key[2], maxB), compB)            # L1222
        B_up = lower_bound(table, B_lo, N_up, min(kk(B_lo)[2] + 1, maxB + 1), compB)  # L1235
        if trace:
            trace.append(("block", table[B_lo][0], table[B_up - 1][0], tuple(key)))
        for it in range(B_lo, B_up):                                               # L1262-1304
            d = grid_distance(key, table[it][0])
            if d <= best_dist:
                if accepts(table[it][1], problem_orig):
                    batch_it = it
                    best_dist = d
                    best_match = table[it][1]
                    this_match = True
        if not batch_exp and first_find:                                           # L1308-1347
            first_find = False
            if len(key) > 3 and key[2] != 1:
                if key[0] > key[1]:
                    key[0] = key[0] * key[2]
                else:
                    key[1] = key[1] * key[2]
                key[2] = 1
                if trace:
                    trace.append(("batch-flatten", tuple(key)))
                if table[batch_it][0][2] == 1:
                    M_lo = M_up = N_lo = N_up = B_lo = B_up = 0
                    base_n = 0
                    step_n = 0
                    if trace:
                        trace.append(("restart",))
                    continue
        if this_match or best:                                                     # L1349-1365
            if best_match not in best:
                best.append(best_match)

    if len(best) < num_solutions:                                                  # L1368-1418
        for it in range(B_lo - 1, -1, -1):
            if accepts(table[it][1], problem_orig):
                if table[it][1] not in best:
                    best.append(table[it][1])
                    if trace:
                        trace.append(("backward", table[it][0], table[it][1]))
            if len(best) == num_solutions:
                break
    return best


# ----------------------------------------------------------------------------- kd-tree path
class _Node:
    __slots__ = ("pt", "left", "right", "axis")

    def __init__(self, pt, axis):
        self.pt, self.axis, self.left, self.right = pt, axis, None, None


class _KDTree:
    """PropertyMatching.hpp L118-288 (N=2, T=int32)."""

    def __init__(self, pts):
        self.maximums = [0, 0]
        self.root = self._build(list(pts), 0)

    def _build(self, pts, depth):
        dim = depth % 2
        if not pts:
            return None
        # std::sort is not stable; libstdc++ uses insertion sort (stable) for <=16 elements.
        pts.sort(key=lambda p: p[dim])
        self.maximums[dim] = max(self.maximums[dim], pts[-1][dim])
        mid = len(pts) // 2
        node = _Node(pts[mid], dim)
        node.left = self._build(pts[:mid], depth + 1)
        node.right = self._build(pts[mid + 1:], depth + 1)
        return node

    @staticmethod
    def _dist(p, q):
        d = 0.0
        for i in range(2):
            dx = i32(p[i] - q[i])
            d = f32(d + f32(f32(dx) * f32(dx)))
        return d

    def query(self, pt, n, criteria):
        res = []   # list of (dist, pt); "top" = max by (dist, coords)

        def top():
            return max(res)

        def impl(node):
            if node is None:
                return
            if criteria(pt, node.pt):
                res.append((self._dist(pt, node.pt), node.pt))
                while len(res) > n:
                    res.remove(max(res))
            dim = node.axis

            def other_ok():
                d = abs(i32(pt[dim] - node.pt[dim]))
                return float(i32(d * d)) < top()[0]
            if pt[dim] < node.pt[dim]:
                impl(node.left)
                if len(res) < n or other_ok():
                    impl(node.right)
            else:
                impl(node.right)
                if len(res) < n or other_ok():
                    impl(node.left)
        impl(self.root)
        return sorted(res)   # ascending distance, ties by coords ascending


def _kd_prepare(table):
    ksol = {}
    for key, idx, _ in table:            # table already sorted -> per-point lists sorted (b, k)
        k = key[3] if len(key) > 3 else key[2]
        b = key[2] if len(key) > 3 else 1
        ksol.setdefault((i32(key[0]), i32(key[1])), []).append((i32(k), i32(b), idx))
    pts = sorted(ksol.keys())            # std::map order
    return ksol, _KDTree(pts)


def _kdtree(table, key_orig, accepts, problem_orig, num_solutions, trace):
    """PropertyMatching.hpp L981-1110."""
    best = []
    ksol, tree = _kd_prepare(table)
    mx = tree.maximums
    key = list(key_orig)

    def criteria(pt, cand):
        if pt[0] <= mx[0] and pt[1] <= mx[1]:
            return cand[0] >= pt[0] and cand[1] >= pt[1]
        if pt[0] > mx[0] and pt[1] <= mx[1]:
            return cand[1] >= pt[1]
        if pt[0] <= mx[0] and pt[1] > mx[1]:
            return cand[0] >= pt[0]
        return True

    restart = True
    while restart:
        restart = False
        k = key[3] if len(key) > 3 else key[2]
        b = key[2] if len(key) > 3 else 1
        target = (i32(key[0]), i32(key[1]))
        results = tree.query(target, num_solutions, criteria)
        for _, p in results:
            lst = ksol.get(p)
            if lst is None:
                continue
            if b > 1 and len(lst) == 1:                      # L1029-1037
                if key[0] > key[1]:
                    key[0] = key[0] * key[2]
                else:
                    key[1] = key[1] * key[2]
                key[2] = 1
                restart = True
                break
            lowerB = lambda e, v: e[1] < v
            upperB = lambda v, e: v < e[1]
            compK = lambda e, v: e[0] < v
            bEnd = lower_bound(lst, 0, len(lst), b, lowerB)
            bStart = 0
            if bEnd == len(lst):                             # b too large
                bval = lst[bEnd - 1][1]
                bStart = lower_bound(lst, 0, len(lst), bval, lowerB)
            elif bEnd == 0:                                  # b is 1 (smallest)
                bEnd = upper_bound(lst, 0, len(lst), b, upperB)
            else:
                bval = lst[bEnd][1] - 1
                bStart = lower_bound(lst, 0, len(lst), bval, lowerB)
                bEnd += 1
            lo = lower_bound(lst, bStart, bEnd, i32(k), compK)
            if lo == bEnd:
                lo -= 1
            if lo == bStart:
                prev = lo
            else:
                prev = lo
                lo -= 1
            if prev != lo:
                if abs(lst[prev][0] - k) < abs(lst[lo][0] - k):
                    lo = prev
            idx = lst[lo][2]
            if accepts(idx, problem_orig):
                if idx in best:
                    continue
                if trace is not None:
                    trace.append(("kd-select", p, lst[lo][1], lst[lo][0], idx))
                best.append(idx)
    return best


# ----------------------------------------------------------------------------- public
def select_top(table_entries, problem, env=None, accepts=None, num_solutions=1, trace=None):
    cfg = parse_env(env)
    if cfg["metric"] and cfg["metric"] != "GridBased":
        raise NotImplementedError("TENSILE_METRIC=%s replaces the GridBased distance for every "
                                  "Matching table (Serialization/MatchingLibrary.hpp L164-167)"
                                  % cfg["metric"])
    accepts = accepts or (lambda idx, prob: True)
    table = sort_table(_norm_entries(table_entries))
    if not table:
        return []
    nprops = len(table[0][0])
    key = _problem_key(problem, nprops)
    if cfg["kdtree"]:
        return _kdtree(table, key, accepts, key, num_solutions, trace)
    return _binary(table, key, accepts, key, num_solutions, cfg["batch_exp"], trace)


def select(table_entries, problem, env=None, accepts=None):
    """Index chosen by findBestSolution / findTopSolutions(n=1) (PyTorch asks for 1 algo)."""
    r = select_top(table_entries, problem, env, accepts, 1)
    return r[0] if r else None


# ----------------------------------------------------------------------------- catalogs
def load_catalog(path):
    import msgpack
    raw = open(path, "rb").read()
    try:
        raw = zlib.decompress(raw)
    except zlib.error:
        pass
    return msgpack.unpackb(raw, raw=False, strict_map_key=False)


def gridbased_rows(cat):
    """[(row_position, properties, table, distance)] for every GridBasedMatching row."""
    out = []
    lib = cat["library"]
    if lib.get("type") != "Problem":
        raise ValueError("expected top-level Problem library")
    for pos, row in enumerate(lib["rows"]):
        if row["predicate"]["type"] == "GridBasedMatching":
            sub = row["library"]
            out.append((pos, [p["type"] for p in sub["properties"]], sub["table"],
                        sub["distance"]))
    return out


def nt_size_accepts(cat, lda=None, ldb=None, ldc=None, beta=0.0, elem_bytes=2):
    """Model of the size-dependent problem predicates of the solutions in `cat` for a plain
    (no-bias) bf16 NT strided GEMM: LeadingFree0/1SizesGreaterOrEqual, WorkgroupNumberCheck,
    BufferLoad/Store offset checks, GlobalSplitUCheckMinK. Type/epilogue predicates are assumed
    to pass (they are identical for 102/103 here).

    NOT modelled (all accept for sols 102/103 in the DLRMv4 workload; a rejection would only
    make the C++ walk fall through to the next entry, which for every T8 target is a 102 guard):
      - softwarePredicate (SolutionLibrary.hpp L74-96, 10.2 / 5911365c only; the 7.14 branch has
        only problemPredicate && taskPredicate):
          streamKDynamicQueueSupported(problem, hw)  -- rejects only StreamK dynamic-queue
              (SK4 / dynamic SK5) kernels on non-power-of-two XCD counts; 102/103 are SK0.
          uniformSummationOrderSupported(problem, hw) -- returns true unless uniform summation
              order is requested via a matmul attribute; PyTorch never sets it.
      - taskPredicate (WorkspaceCheck, LaunchLimits): 102/103 have globalSplitU = 1, so auto-GSU
        is 1 and no workspace is needed.
      - a user split-K (problem.getParams().gsu(), e.g. hipblaslt-bench --splitk) overriding the
        catalog GSU: not used by training or by the T8 GPU checks."""
    sols = {s["index"]: s for s in cat["solutions"]}

    def acc(idx, prob):
        m, n, b, k = prob
        s = sols.get(idx)
        if s is None:
            return False
        a_ld = lda if lda else m
        b_ld = ldb if ldb else n
        c_ld = ldc if ldc else m
        for p in s["problemPredicate"]["value"]:
            t, v = p["type"], p.get("value")
            if t == "LeadingFree0SizesGreaterOrEqual" and not m >= v:
                return False
            if t == "LeadingFree1SizesGreaterOrEqual" and not n >= v:
                return False
            if t == "WorkgroupNumberCheck":
                gsu = v[2] if v[2] > 1 else 1
                import math
                if math.ceil(m / v[0]) * math.ceil(n / v[1]) * gsu * b > 16777216:
                    return False
            if t == "BufferLoadOffsetLimitCheck":
                # NT: A is (i,l) with stride[1]=lda, size[1]=K; B is (j,l) stride[1]=ldb
                if (a_ld * min(v["DUorMT0"], k) + v["ShiftPtrElemA"]) * elem_bytes >= 1 << 32:
                    return False
                if (b_ld * min(v["DUorMT1"], k) + v["ShiftPtrElemB"]) * elem_bytes >= 1 << 32:
                    return False
            if t == "BufferStoreOffsetLimitCheck" and c_ld * min(v, n) * elem_bytes >= 1 << 32:
                return False
            if t == "BufferLoadOffsetLimitCheck_Beta" and beta != 0 and \
                    c_ld * min(v, n) * elem_bytes >= 1 << 32:
                return False
            if t == "GlobalSplitUCheckMinK":
                mink = v[1]
                if mink != -1:
                    mink = 0 if mink == 1 else mink
                    if k < mink * v[0]:
                        return False
        return True
    return acc


# ----------------------------------------------------------------------------- self-test
def _selftest():
    ok = 0

    def check(name, got, exp):
        nonlocal ok
        if got != exp:
            raise AssertionError("%s: got %r expected %r" % (name, got, exp))
        ok += 1

    # The 35-entry stub table shipped in both the 10.2 host catalog and the stock 7.14 catalog.
    stub = [([1, 1, 1, 1], 102), ([1, 1, 1, 63], 102), ([1, 1, 1, 64], 102), ([1, 1, 1, 65], 102),
            ([1, 1, 1, 127], 102), ([1, 1, 1, 128], 102), ([1, 1, 1, 129], 102),
            ([1, 1, 1, 255], 102), ([1, 1, 1, 256], 102), ([1, 1, 1, 257], 102),
            ([1, 1, 1, 512], 102), ([1, 63, 1, 1], 102), ([1, 64, 1, 1], 102),
            ([1, 65, 1, 1], 102), ([1, 127, 1, 1], 102), ([1, 128, 1, 1], 102),
            ([1, 129, 1, 1], 102), ([1, 255, 1, 1], 102), ([1, 256, 1, 1], 102),
            ([1, 257, 1, 1], 102), ([1, 1024, 1, 512], 102), ([8, 8, 1, 8], 102),
            ([16, 16, 1, 16], 102), ([63, 1, 1, 1], 102), ([64, 1, 1, 1], 102),
            ([65, 1, 1, 1], 102), ([127, 1, 1, 1], 102), ([128, 1, 1, 1], 102),
            ([128, 128, 1, 128], 102), ([129, 1, 1, 1], 102), ([255, 1, 1, 1], 102),
            ([256, 1, 1, 1], 102), ([256, 256, 1, 512], 103), ([257, 1, 1, 1], 102),
            ([1024, 1, 1, 512], 102)]
    # Hand-worked (see report): M -> first grid M >= m (clamped to max M=1024), then N inside
    # that M block -> first N >= n (clamped to block max), B likewise, then nearest K.
    check("G20 2048x512", select(stub, (2048, 512, 1, 2097152)), 102)   # -> [1024,1,1,512]
    check("G19 512x1536", select(stub, (512, 1536, 1, 1590000)), 102)   # -> [1024,1,1,512]
    check("G27 24x256", select(stub, (24, 256, 1, 2533005)), 102)       # -> [63,1,1,1]
    check("256x512xT", select(stub, (256, 512, 1, 2097152)), 103)       # -> [256,256,1,512]
    check("256x2", select(stub, (256, 2, 1, 7)), 103)                   # N ceil 256
    check("256x1", select(stub, (256, 1, 1, 7)), 102)                   # [256,1,1,1]
    check("255x512", select(stub, (255, 512, 1, 7)), 102)               # M ceil -> 255 block
    check("128x128", select(stub, (128, 100, 1, 128)), 102)

    # nearest K, ties -> later (larger K) entry because of '<=' (L1267)
    t = [([64, 64, 1, 100], 1), ([64, 64, 1, 200], 2)]
    check("K tie", select(t, (64, 64, 1, 150)), 2)
    check("K near low", select(t, (64, 64, 1, 149)), 1)
    check("K huge", select(t, (64, 64, 1, 10 ** 9)), 2)
    # M ceiling and clamp; M/N are NOT nearest-neighbour
    t = [([64, 64, 1, 1], 1), ([128, 64, 1, 1], 2)]
    check("M ceil", select(t, (65, 64, 1, 1)), 2)
    check("M exact", select(t, (64, 64, 1, 1)), 1)
    check("M clamp", select(t, (100000, 64, 1, 1)), 2)
    check("M below", select(t, (1, 1, 1, 1)), 1)
    # N is chosen inside the chosen M block only
    t = [([64, 32, 1, 1], 1), ([64, 4096, 1, 1], 2), ([128, 32, 1, 1], 3)]
    check("N in block", select(t, (100, 4096, 1, 1)), 3)   # M block 128 has only N=32
    check("N ceil", select(t, (64, 33, 1, 1)), 2)
    # predicate rejection: forward walk to next N group, then next M block, then backward
    t = [([64, 32, 1, 1], 1), ([64, 64, 1, 1], 2), ([128, 32, 1, 1], 3)]
    check("fwd next N", select(t, (64, 32, 1, 1), accepts=lambda i, p: i != 1), 2)
    check("fwd next M", select(t, (64, 64, 1, 1), accepts=lambda i, p: i != 2), 3)
    check("backward", select(t, (128, 32, 1, 1), accepts=lambda i, p: i != 3), 2)
    check("none", select(t, (128, 32, 1, 1), accepts=lambda i, p: False), None)
    # inside a block, a rejected nearer K is skipped in favour of an accepted farther K
    t = [([64, 64, 1, 100], 1), ([64, 64, 1, 200], 2)]
    check("K skip", select(t, (64, 64, 1, 110), accepts=lambda i, p: i != 1), 2)
    # batch flattening (default, BATCH_EXP=0): batch folded into the smaller of M/N
    t = [([512, 512, 1, 1024], 1), ([512, 4096, 1, 1024], 2)]
    check("batch flatten", select(t, (512, 512, 8, 1024)), 2)
    check("batch exp", select(t, (512, 512, 8, 1024), {"TENSILE_GRIDBASED_BATCH_EXP": "1"}), 1)
    t2 = [([512, 512, 1, 1024], 1), ([512, 512, 8, 1024], 5), ([512, 4096, 1, 1024], 2)]
    check("batch in grid", select(t2, (512, 512, 8, 1024)), 5)
    # top-N
    t = [([64, 32, 1, 1], 1), ([64, 64, 1, 1], 2), ([128, 32, 1, 1], 3), ([128, 64, 1, 1], 4)]
    check("top3", select_top(t, (64, 32, 1, 1), num_solutions=3), [1, 2, 3])
    # KD-tree mode: (M,N) by k-NN among dominating points, K nearest with ties -> smaller K
    kd = {"TENSILE_GRIDBASED_KDTREE": "1"}
    t = [([64, 64, 1, 100], 1), ([64, 64, 1, 200], 2), ([128, 128, 1, 1], 3)]
    check("kd tie", select(t, (64, 64, 1, 150), kd), 1)
    check("kd dom", select(t, (65, 64, 1, 150), kd), 3)
    check("kd stub G20", select(stub, (2048, 512, 1, 2097152), kd), 102)
    # no grid point has M>=256 and N>=512 -> kd path returns NOTHING (binary path gives 103)
    check("kd stub 256x512", select(stub, (256, 512, 1, 2097152), kd), None)
    # only dominating point for (24,256) is (256,256) -> 103 (binary path gives 102)
    check("kd stub 24x256", select(stub, (24, 256, 1, 2097152), kd), 103)
    check("strtol", (c_strtol0("0x10"), c_strtol0("010"), c_strtol0(" 7abc"), c_strtol0("x")),
          (16, 8, 7, 0))
    check("env off", select(t, (64, 64, 1, 150), {"TENSILE_GRIDBASED_KDTREE": "0"}), 2)
    print("gridmatch selftest: %d checks passed" % ok)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "--selftest":
        # usage: gridmatch.py catalog.dat.zlib m n batch k [ENV=VAL ...]
        cat = load_catalog(sys.argv[1])
        m, n, b, k = (int(x) for x in sys.argv[2:6])
        env = dict(a.split("=", 1) for a in sys.argv[6:])
        for pos, props, table, dist in gridbased_rows(cat):
            tr = []
            r = select_top(table, (m, n, b, k), env, nt_size_accepts(cat), 1, tr)
            print("row", pos, props, dist, "->", r, tr)
    else:
        _selftest()
