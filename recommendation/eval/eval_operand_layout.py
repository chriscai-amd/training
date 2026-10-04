#!/usr/bin/env python3
"""Do the logged wgrad operands arrive as dense row-major [T, C] tensors?

No GPU. Reads a hipblaslt-bench union file (capture_to_yaml.py output, e.g. the
pack's eval/dlrmv4-BF16-allgemms-union.txt) and, for every K = T GEMM, checks
whether the leading dimensions torch logged equal the operands' column counts.

Why it matters: feature_major()/tn_wgrad_mm transpose the token-major wgrad
operands to [C, T]. The k-pass split (iter 13) is only valid on a contiguous
operand (it falls back to one direct transpose otherwise). torch's
prepare_matrix_for_cublas passes a (C,1)-strided tensor as-is with ld =
stride(0), passes a (1,T)-strided one with the transpose flag flipped, and
copies anything else to a contiguous tensor first. So for an NT wgrad
(transA N, transB T, K = T), logged lda == M and ldb == N mean that each operand
was either already dense row-major [T, C] or an irregular-stride tensor that
torch copied (the log cannot tell these two apart). ld > C would mean a
column slice of a wider tensor, which is NOT contiguous.

Usage: eval_operand_layout.py <union.txt> [--out json]
"""
import json
import re
import sys

BIG = 100000


def parse(line):
    d = {}
    for k, v in re.findall(r"(\w+):\s*([^,}]+)", line):
        d[k] = v.strip()
    return d


def main():
    path = sys.argv[1]
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    rows = []
    gid = 0
    for line in open(path):
        if "function: matmul" not in line:
            continue
        gid += 1
        d = parse(line)
        m, n, k = int(d["M"]), int(d["N"]), int(d["K"])
        if k <= BIG:
            continue
        ta, tb = d["transA"], d["transB"]
        lda, ldb = int(d["lda"]), int(d["ldb"])
        # Row-major torch tensor [T, C] seen column-major is C x T with ld = C.
        # A operand is m x k (N) or k x m (T); B is k x n (N) or n x k (T).
        a_cols = m if ta == "N" else k
        b_cols = n if tb == "T" else k
        rows.append({
            "id": f"G{gid:02d}", "layout": ta + tb, "m": m, "n": n, "k": "T",
            "lda": lda, "ldb": ldb,
            "a_token_major_dense": ta == "N" and lda == a_cols,
            "b_token_major_dense": tb == "T" and ldb == b_cols,
        })
    for r in rows:
        r["both_dense"] = r["a_token_major_dense"] and r["b_token_major_dense"]
        print(f"{r['id']} {r['layout']} {r['m']}x{r['n']}xT lda={r['lda']} ldb={r['ldb']} "
              f"dense A={r['a_token_major_dense']} B={r['b_token_major_dense']}")
    nt = [r for r in rows if r["layout"] == "NT"]
    verdict = {"nt_wgrads": len(nt), "nt_both_dense": sum(r["both_dense"] for r in nt),
               "rows": rows}
    print(f"NT K=T wgrads: {verdict['nt_both_dense']}/{verdict['nt_wgrads']} with both operands "
          f"dense row-major [T, C] (or copied contiguous by torch)")
    if out:
        json.dump(verdict, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
