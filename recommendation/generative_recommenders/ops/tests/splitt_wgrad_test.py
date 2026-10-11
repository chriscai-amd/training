# pyre-unsafe
"""CPU-only tests for OPT M5 (split-T NT weight gradients) in ops/triton/triton_addmm.py.

Never initializes a GPU. The production path is CUDA-gated, so the split math is
tested through the pure functions (splitt_wgrad_plan / splitt_wgrad_compute) and
the dispatch through the real hooks with the device gate / module flags patched.

Run (no --device):
  docker run --rm --entrypoint python \
    -v /home/chcai/training/recommendation:/workspace/recommendation:ro \
    -w /workspace/recommendation <image> -m pytest -p no:cacheprovider -q \
    generative_recommenders/ops/tests/splitt_wgrad_test.py
"""

import ast
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
from generative_recommenders.ops.triton import triton_addmm as ta

BF16 = torch.bfloat16
ADDMM_PATH = Path(ta.__file__)
REPO_ROOT = ADDMM_PATH.parents[3]


def _int_data(T, D, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(-3, 4, (T, D), generator=g).to(BF16)


def _emulated_reference(a, b, s):
    """Exact model of the split result on small-integer data: fp32 partial sums
    are exact (< 2^24), so the only roundings are the bf16 store of each
    partial and the final bf16 cast."""
    T, D = a.shape
    O = b.shape[1]
    L = T // s
    main = s * L
    p = torch.bmm(a[:main].double().view(s, L, D).transpose(1, 2), b[:main].double().view(s, L, O))
    ref = p.to(BF16).double().sum(0) + a[main:].double().t() @ b[main:].double()
    return ref.to(BF16)


def _rel(x, ref):
    return ((x.double() - ref.double()).norm() / ref.double().norm().clamp_min(1e-30)).item()


def _orig_triton_addmm_bwd(x, w, dz, is_y_1d):
    """triton_addmm_bwd as it was before M5 (be2d249), with _TN_HSTU off."""
    dy = torch.sum(dz, dim=0) if is_y_1d else dz
    dw = torch.mm(x.t(), dz)
    dx = torch.mm(dz, w.t())
    return dx, dw, dy


def _no_bmm(*args, **kwargs):
    raise AssertionError("torch.bmm must not be called on the M5-off path")


class SplitMathTest(unittest.TestCase):
    # (T, D, O, s): D = width of a, O = width of b; hipBLASLt m x n = O x D.
    CASES = [
        (100_003, 512, 256, 8),   # tail 3
        (65_536, 256, 24, 16),    # G27-like (O = 24), tail 0
        (99_999, 512, 256, 7),    # tail 4
        (40_960, 512, 1536, 10),  # G19-like
        (12_345, 64, 8, 3),
        (8_191, 16, 32, 64),      # tail 63
        (100_001, 64, 32, 8),     # tail 1 (elementwise addr_ path)
    ]

    def test_bit_exact_vs_emulation(self):
        for i, (T, D, O, s) in enumerate(self.CASES):
            a, b = _int_data(T, D, 2 * i), _int_data(T, O, 2 * i + 1)
            out = ta.splitt_wgrad_compute(a, b, s)
            with self.subTest(T=T, D=D, O=O, s=s, tail=T % s):
                self.assertTrue(torch.equal(out, _emulated_reference(a, b, s)))

    def test_matches_mm_within_bf16_tolerance(self):
        g = torch.Generator().manual_seed(7)
        cases = self.CASES + [(8, 8, 16, 8), (9, 8, 16, 8), (15, 24, 8, 4), (33, 16, 8, 32), (5, 8, 8, 1)]
        for T, D, O, s in cases:
            a = torch.randn(T, D, generator=g).to(BF16)
            b = torch.randn(T, O, generator=g).to(BF16)
            ref64 = a.double().t() @ b.double()
            base = torch.mm(a.t(), b)
            out = ta.splitt_wgrad_compute(a, b, s)
            with self.subTest(T=T, D=D, O=O, s=s, tail=T % s):
                self.assertEqual(out.shape, base.shape)
                self.assertEqual(out.dtype, base.dtype)
                self.assertEqual(out.stride(), base.stride())
                self.assertTrue(out.is_contiguous())
                # bf16 output: one rounding of each partial + one of the result (<= ~2 ulp of 2^-8)
                self.assertLess(_rel(out, ref64), 1e-2)
                self.assertLess(_rel(out, base), 1.5e-2)
                torch.testing.assert_close(out.double(), ref64, rtol=1.6e-2, atol=2e-2 * ref64.abs().max().item())

    def test_zero_dy_gives_positive_zero(self):
        a = torch.zeros(10_007, 256, dtype=BF16)
        b = torch.randn(10_007, 512).to(BF16)
        out = ta.splitt_wgrad_compute(a, b, 16)
        self.assertTrue(torch.equal(out.view(torch.int16), torch.zeros_like(out).view(torch.int16)))

    def test_zero_copy_strided_batched_operands(self):
        T, D, O, s = 10_005, 64, 128, 8
        a, b = _int_data(T, D, 1), _int_data(T, O, 2)
        L = T // s
        with patch.object(torch, "bmm", wraps=torch.bmm) as bmm:
            ta.splitt_wgrad_compute(a, b, s)
        self.assertEqual(bmm.call_count, 1)
        a3, b3 = bmm.call_args.args
        self.assertEqual(a3.data_ptr(), a.data_ptr())
        self.assertEqual(b3.data_ptr(), b.data_ptr())
        self.assertEqual(tuple(a3.shape), (s, D, L))
        self.assertEqual(a3.stride(), (L * D, 1, D))   # transB=T, no copy
        self.assertEqual(b3.stride(), (L * O, O, 1))

    def test_tail_kernels(self):
        # r = 1: fp32 outer product (addr_), no GEMM; r >= 2: one fp32 addmm_ on [D, r] x [r, O]
        for T, want in ((4_097, "addr_"), (4_098, "addmm_"), (4_096, None)):
            a, b = _int_data(T, 16, 8), _int_data(T, 8, 9)
            calls = {"addr_": [], "addmm_": []}

            def spy(nm):
                orig = getattr(torch.Tensor, nm)

                def f(self, *args, **kw):
                    calls[nm].append(args)
                    return orig(self, *args, **kw)
                return f
            with patch.object(torch.Tensor, "addr_", spy("addr_")), patch.object(torch.Tensor, "addmm_", spy("addmm_")):
                out = ta.splitt_wgrad_compute(a, b, 4)
            with self.subTest(T=T):
                self.assertEqual((len(calls["addr_"]), len(calls["addmm_"])),
                                 {"addr_": (1, 0), "addmm_": (0, 1), None: (0, 0)}[want])
                if want == "addmm_":
                    m1, m2 = calls["addmm_"][0][:2]
                    self.assertEqual((m1.dtype, m1.stride(), m2.stride()), (torch.float32, (1, 16), (8, 1)))
                self.assertTrue(torch.equal(out, _emulated_reference(a, b, 4)))

    def test_error_scales_with_chunk_partials_not_result(self):
        """The bf16 partials are rounded at their own magnitude: error <= ~u * sum_i |P_i|
        (+ one final rounding), which for chunk-cancelling data (dY drifting -1 -> +1 along T)
        is far above today's single-rounding error. Documents the numerics model (M5_REPORT)."""
        g = torch.Generator().manual_seed(11)
        T, D, O, s = 1 << 16, 32, 16, 8
        x = (1.0 + 0.1 * torch.randn(T, O, generator=g)).to(BF16)                 # b: activations
        drift = torch.linspace(-1, 1, T).unsqueeze(1)
        for name, dy in (("iid", torch.randn(T, D, generator=g)),
                         ("drift", drift + 0.05 * torch.randn(T, D, generator=g))):
            a = dy.to(BF16)
            ref = a.double().t() @ x.double()
            out = ta.splitt_wgrad_compute(a, x, s)
            P = torch.bmm(a.double().view(s, T // s, D).transpose(1, 2), x.double().view(s, T // s, O))
            bound = 2.0 ** -8 * (P.abs().sum(0) + ref.abs()) + 1e-5 * (a.double().abs().t() @ x.double().abs())
            with self.subTest(name):
                self.assertTrue(bool(((out.double() - ref).abs() <= bound).all()))
            ratio = (out.double() - ref).norm() / (torch.mm(a.t(), x).double() - ref).norm()
            if name == "iid":
                self.assertLess(ratio.item(), 2.5)
            else:
                self.assertGreater(ratio.item(), 5.0)   # cancellation: much worse than 1.41x

    def test_partials_are_bf16_and_reduced_in_fp32(self):
        a, b = _int_data(4_099, 16, 3), _int_data(4_099, 8, 4)
        with patch.object(torch, "bmm", wraps=torch.bmm) as bmm:
            out = ta.splitt_wgrad_compute(a, b, 4)
        self.assertNotIn("out_dtype", bmm.call_args.kwargs)
        self.assertEqual(out.dtype, BF16)


class PlanTest(unittest.TestCase):
    SPL = {(256, 512): 8, (24, 256): 16}

    def plan(self, a, b, splits=None, min_rows=1):
        return ta.splitt_wgrad_plan(a, b, self.SPL if splits is None else splits, min_rows)

    def test_applicable(self):
        a, b = torch.zeros(1000, 512, dtype=BF16), torch.zeros(1000, 256, dtype=BF16)
        self.assertEqual(self.plan(a, b), 8)
        self.assertEqual(self.plan(torch.zeros(1000, 256, dtype=BF16), torch.zeros(1000, 24, dtype=BF16)), 16)

    def test_fallbacks(self):
        a, b = torch.zeros(1000, 512, dtype=BF16), torch.zeros(1000, 256, dtype=BF16)
        buf = torch.zeros(1000 * 512 + 1, dtype=BF16)
        cases = {
            "shape not configured": (b, a, None, 1),
            "s=1": (a, b, {(256, 512): 1}, 1),
            "fp32": (a.float(), b.float(), None, 1),
            "fp16": (a.half(), b.half(), None, 1),
            "a column-major": (a.t().contiguous().t(), b, None, 1),
            "b sliced columns": (a, torch.zeros(1000, 264, dtype=BF16)[:, :256], None, 1),
            "misaligned ptr": (buf[1:].view(1000, 512), b, None, 1),
            "rows mismatch": (a[:999], b, None, 1),
            "3-D": (a.view(10, 100, 512), b, None, 1),
            "L < min_rows": (a, b, None, 126),
            "s > T": (a[:7], b[:7], None, 1),
            "D % 8": (torch.zeros(1000, 20, dtype=BF16), b, {(256, 20): 4}, 1),
        }
        for name, (x, y, sp, mr) in cases.items():
            with self.subTest(name):
                self.assertEqual(self.plan(x, y, sp, mr), 0)
        self.assertEqual(self.plan(a, b, None, 125), 8)   # L = 125 exactly

    def test_u32_stride_margin(self):
        # meta tensors: shape/stride only, no memory. L * max(D, O) must stay < 2^31.
        def m(T, D):
            return torch.empty(T, D, device="meta", dtype=BF16)
        sp = {(2048, 512): 2}
        self.assertEqual(ta.splitt_wgrad_plan(m(2_097_150, 512), m(2_097_150, 2048), sp, 1), 2)
        self.assertEqual(ta.splitt_wgrad_plan(m(2_097_152, 512), m(2_097_152, 2048), sp, 1), 0)
        sp = {(2048, 512): 4}
        self.assertEqual(ta.splitt_wgrad_plan(m(2_533_005, 512), m(2_533_005, 2048), sp, 4096), 4)


def _write_manifest(root, variant, shapes):
    import json
    lib = Path(root) / variant
    (lib / "hipblaslt" / "library" / "gfx1250").mkdir(parents=True)
    (lib / "m5_splits.json").write_text(json.dumps({"variant": variant, "shapes": shapes}))
    return str(lib / "hipblaslt" / "library" / "gfx1250")


class CatalogCheckTest(unittest.TestCase):
    SH = {"2048x512": {"sol": 103, "sol_by_s": {"1": 103, "4": 103, "8": 103, "12": 102}},
          "24x256": {"sol": 102, "sol_by_s": {"1": 102, "64": 102}}}

    def test_check(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            lp = _write_manifest(d, "vX", self.SH)
            v, exp, probs = ta._splitt_catalog_check({(2048, 512): 8, (24, 256): 64}, lp + "/")
            self.assertEqual((v, exp, probs), ("vX", {(2048, 512, 8): 103, (24, 256, 64): 102}, []))
            v, exp, probs = ta._splitt_catalog_check({(2048, 512): 12, (512, 256): 4, (24, 256): 1}, lp)
            self.assertEqual(v, "vX")
            self.assertEqual(len(probs), 2)
            self.assertIn("s=12: catalog vX selects sol 102", probs[0] + probs[1])
            self.assertIn("pinned s: [4, 8]", probs[0] + probs[1])
            self.assertIn("512x256: shape not in", probs[0] + probs[1])
            v, _, probs = ta._splitt_catalog_check({(2048, 512): 8}, d + "/none/hipblaslt/library/gfx1250")
            self.assertIsNone(v)
            self.assertIn("unverified", probs[0])
        self.assertIsNone(ta._splitt_catalog_check({(2048, 512): 8}, None)[0])

    @unittest.skipIf(torch.version.hip is None, "M5 is ROCm-only")
    def test_import_refuses_unpinned_s(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            lp = _write_manifest(d, "vX", self.SH)
            base = {"HSTU_SPLITT_WGRAD": "hstu", "HIPBLASLT_TENSILE_LIBPATH": lp}
            r = _import_flags(dict(base, HSTU_SPLITT_WGRAD_SPLITS="2048x512:8"))
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            r = _import_flags(dict(base, HSTU_SPLITT_WGRAD_SPLITS="2048x512:12"))
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("not pinned in the hipBLASLt catalog", r.stderr)
            r = _import_flags(dict(base, HSTU_SPLITT_WGRAD_SPLITS="2048x512:12", HSTU_SPLITT_ALLOW_UNPINNED="1"))
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            self.assertIn("not pinned", r.stderr)
        r = _import_flags({"HSTU_SPLITT_WGRAD": "all"})          # on, nothing configured: warns
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("nothing is split", r.stderr)


class ParseTest(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(ta._parse_splitt(""), {})
        self.assertEqual(ta._parse_splitt(" 2048x512:8, 24X256:128 ,"), {(2048, 512): 8, (24, 256): 128})
        for bad in ("2048x512", "2048:8", "0x512:8", "2048x512:0", "2048x512:5000", "axb:2",
                    "2048x512:8,2048x512:4"):
            with self.subTest(bad), self.assertRaises(ValueError):
                ta._parse_splitt(bad)


def _on(**extra):
    """Patch the module as if HSTU_SPLITT_WGRAD=all on ROCm and CPU tensors were CUDA."""
    kw = dict(_splitt_device_ok=lambda t: True, _SPLITT_MIN_ROWS=1, _SPLITT_PREPROC=True,
              _SPLITT_HSTU=True, _SPLITT_SPLITS={(256, 512): 8, (24, 256): 16, (2048, 512): 4})
    kw.update(extra)
    return patch.multiple(ta, **kw)


class DispatchTest(unittest.TestCase):
    def test_gate_on_real_device(self):
        a, b = torch.zeros(1000, 512, dtype=BF16), torch.zeros(1000, 256, dtype=BF16)
        with patch.multiple(ta, _SPLITT_SPLITS={(256, 512): 8}, _SPLITT_MIN_ROWS=1):
            self.assertIsNone(ta.splitt_wgrad_mm(a, b))      # CPU tensors: not CUDA -> original call

    def test_configured_but_refused_logs_once(self):
        a, b = torch.zeros(1000, 512), torch.zeros(1000, 256)      # fp32 -> refused
        with _on(_SPLITT_SEEN=set()), self.assertLogs(ta._splitt_logger, "INFO") as cm:
            self.assertIsNone(ta.splitt_wgrad_mm(a, b))
            self.assertIsNone(ta.splitt_wgrad_mm(a, b))
            ta._splitt_logger.info("end")
        self.assertEqual(len(cm.output), 2)
        self.assertIn("256 x 512 configured but not applied", cm.output[0])

    def test_splitt_wgrad_mm_and_tn_wgrad_mm(self):
        a, b = _int_data(20_011, 512, 5), _int_data(20_011, 256, 6)
        with _on(), patch.object(ta, "splitt_wgrad_compute", wraps=ta.splitt_wgrad_compute) as comp:
            out1 = ta.splitt_wgrad_mm(a, b)
            out2 = ta.tn_wgrad_mm(a, b, enabled=False, splitt=True)
            out3 = ta.tn_wgrad_mm(a, b, enabled=False, splitt=False)
        self.assertEqual(comp.call_count, 2)
        self.assertEqual(comp.call_args.args[2], 8)
        ref = _emulated_reference(a, b, 8)
        self.assertTrue(torch.equal(out1, ref) and torch.equal(out2, ref))
        self.assertTrue(torch.equal(out3, torch.mm(a.t(), b)))
        # not applicable -> falls through to the original call
        with _on(_SPLITT_SPLITS={}):
            self.assertTrue(torch.equal(ta.tn_wgrad_mm(a, b, enabled=False, splitt=True), torch.mm(a.t(), b)))

    def test_tnwgrad_linear_mlp_gradient_flow(self):
        """Preprocessor-style MLP (24 -> 256 -> 512, G27 + G25 keys) with fp32 master
        params and bf16 activations: M5 path vs plain autograd of the same bf16 math."""
        torch.manual_seed(0)
        T = 30_001
        l1, l2 = ta.TNWgradLinear(24, 256), ta.TNWgradLinear(256, 512)
        r1, r2 = torch.nn.Linear(24, 256), torch.nn.Linear(256, 512)
        r1.load_state_dict(l1.state_dict())
        r2.load_state_dict(l2.state_dict())
        x = torch.randn(T, 24).to(BF16)
        g = torch.randn(T, 512).to(BF16)

        def ref_fwd(xx):
            h = F.silu(F.linear(xx, r1.weight.to(BF16), r1.bias.to(BF16)))
            return F.linear(h, r2.weight.to(BF16), r2.bias.to(BF16))

        xr = x.clone().requires_grad_()
        ref_fwd(xr).backward(g)
        xs = x.clone().requires_grad_()
        with _on(), patch.object(ta, "splitt_wgrad_compute", wraps=ta.splitt_wgrad_compute) as comp:
            y = l2(F.silu(l1(xs)))
            y.backward(g)
        self.assertEqual(sorted(c.args[2] for c in comp.call_args_list), [8, 16])
        self.assertEqual(y.dtype, BF16)
        for p, q in ((l1.weight, r1.weight), (l2.weight, r2.weight), (l1.bias, r1.bias), (l2.bias, r2.bias)):
            self.assertEqual(p.grad.dtype, torch.float32)
            self.assertEqual(p.grad.shape, p.shape)
            self.assertLess(_rel(p.grad, q.grad), 1.5e-2)
        # dx and db are the same ops aten issues; dW is the only split-T change.
        self.assertLess(_rel(xs.grad, xr.grad), 1.5e-2)
        self.assertTrue(torch.isfinite(l1.weight.grad).all() and torch.isfinite(l2.weight.grad).all())

    def test_preproc_scope_does_not_enable_tn(self):
        # M3 (hstu TN) + M5 preproc: the Function must not issue the TN copy.
        lin = ta.TNWgradLinear(16, 32)
        x = torch.randn(64, 16).to(BF16)
        g = torch.randn(64, 32).to(BF16)
        # configured shape, split refused (min_rows) -> original NT call, never the TN copy
        with _on(_HSTU_TN_WGRAD=True, _TN_HSTU=True, _TN_PREPROC=False, _SPLITT_SPLITS={(16, 32): 4},
                 _SPLITT_MIN_ROWS=10**9), \
                patch.object(ta, "tn_wgrad_mm", wraps=ta.tn_wgrad_mm) as tw:
            lin(x).backward(g)
        self.assertEqual(tw.call_count, 1)
        self.assertEqual(tw.call_args.kwargs, {"enabled": False, "splitt": True})
        self.assertEqual(lin.weight.grad.dtype, torch.float32)
        self.assertTrue(torch.equal(lin.weight.grad, g.t().mm(x).float()))

    def test_unconfigured_tnwgrad_linear_stays_nn_linear(self):
        lin = ta.TNWgradLinear(16, 32)
        x = torch.randn(64, 16)
        with _on(_SPLITT_SPLITS={(24, 256): 16}), \
                patch.object(ta._TNWgradLinearFn, "apply", side_effect=AssertionError("Fn used")):
            lin(x).sum().backward()
        self.assertEqual(lin.weight.grad.dtype, torch.float32)
        with _on(_SPLITT_SPLITS={(16, 32): 4}), \
                patch.object(ta._TNWgradLinearFn, "apply", wraps=ta._TNWgradLinearFn.apply) as fn:
            lin(x.to(BF16)).float().sum().backward()
        self.assertEqual(fn.call_count, 1)

    def test_audit(self):
        a, b = _int_data(20_011, 512, 5), _int_data(20_011, 256, 6)
        with _on(_SPLITT_AUDIT_EVERY=2, _SPLITT_AUDIT_COUNT={}, _SPLITT_SEEN=set()), \
                self.assertLogs(ta._splitt_logger, "INFO") as cm:
            outs = [ta.splitt_wgrad_mm(a, b) for _ in range(3)]
        audits = [m for m in cm.output if "M5 audit" in m]
        self.assertEqual(len(audits), 2)
        self.assertIn("chunk cancellation q", audits[0])
        ref = _emulated_reference(a, b, 8)
        self.assertTrue(all(torch.equal(o, ref) for o in outs))
        self.assertIn("expected sol unverified", next(m for m in cm.output if "GridBased key (256,4096)" in m))

    def test_triton_addmm_bwd_hook(self):
        torch.manual_seed(1)
        x = torch.randn(9_001, 512).to(BF16)
        w = torch.randn(512, 2048).to(BF16)
        dz = torch.randn(9_001, 2048).to(BF16)
        ref = _orig_triton_addmm_bwd(x, w, dz, True)
        with _on(_splitt_wgrad_mm_hstu=ta.splitt_wgrad_mm):
            dx, dw, dy = ta.triton_addmm_bwd(x, w, dz, True)
        self.assertTrue(torch.equal(dx, ref[0]) and torch.equal(dy, ref[2]))
        self.assertEqual((dw.shape, dw.dtype, dw.stride()), (ref[1].shape, ref[1].dtype, ref[1].stride()))
        self.assertLess(_rel(dw, x.double().t() @ dz.double()), 1e-2)
        self.assertTrue(torch.equal(dw, ta.splitt_wgrad_compute(x, dz, 4)))

    def test_hstu_output_projection_call_site(self):
        src = (ADDMM_PATH.parent / "triton_hstu_linear.py").read_text()
        self.assertIn("tn_wgrad_mm(y, dout, enabled=_TN_HSTU, splitt=_SPLITT_HSTU)", src)
        from generative_recommenders.ops.triton import triton_hstu_linear as hl
        self.assertIs(hl.tn_wgrad_mm, ta.tn_wgrad_mm)
        self.assertIs(hl._SPLITT_HSTU, ta._SPLITT_HSTU)


class OffPathTest(unittest.TestCase):
    """HSTU_SPLITT_WGRAD unset/"0" (the default): every hook is the pre-M5 call."""

    @unittest.skipIf(os.environ.get("HSTU_SPLITT_WGRAD", "0") not in ("", "0"), "M5 env set")
    def test_flags_off_by_default(self):
        self.assertFalse(ta._SPLITT_ON or ta._SPLITT_HSTU or ta._SPLITT_PREPROC)
        self.assertIsNone(ta._splitt_wgrad_mm_hstu)
        self.assertEqual(ta._SPLITT_SPLITS, {})

    def test_tn_wgrad_mm_identical(self):
        a, b = torch.randn(5_000, 512).to(BF16), torch.randn(5_000, 256).to(BF16)
        with patch.object(torch, "bmm", _no_bmm):
            for kw in ({}, {"enabled": False}, {"enabled": ta._TN_HSTU, "splitt": ta._SPLITT_HSTU},
                       {"enabled": ta._TN_PREPROC, "splitt": ta._SPLITT_PREPROC}):
                self.assertTrue(torch.equal(ta.tn_wgrad_mm(a, b, **kw), torch.mm(a.t(), b)))
            # on, but CPU tensors -> device gate -> original call
            with patch.multiple(ta, _SPLITT_SPLITS={(256, 512): 8}, _SPLITT_MIN_ROWS=1):
                self.assertTrue(torch.equal(ta.tn_wgrad_mm(a, b, enabled=False, splitt=True), torch.mm(a.t(), b)))

    def test_tnwgrad_linear_is_nn_linear(self):
        torch.manual_seed(2)
        lin = ta.TNWgradLinear(24, 256)
        ref = torch.nn.Linear(24, 256)
        ref.load_state_dict(lin.state_dict())
        x = torch.randn(3_000, 24)
        g = torch.randn(3_000, 256)
        x1, x2 = x.clone().requires_grad_(), x.clone().requires_grad_()
        with patch.object(torch, "bmm", _no_bmm), \
                patch.object(ta._TNWgradLinearFn, "apply", side_effect=AssertionError("Fn used")):
            lin(x1).backward(g)
        ref(x2).backward(g)
        for p, q in ((lin.weight.grad, ref.weight.grad), (lin.bias.grad, ref.bias.grad), (x1.grad, x2.grad)):
            self.assertTrue(torch.equal(p, q))

    def test_triton_addmm_bwd_identical(self):
        torch.manual_seed(3)
        for is_y_1d in (True, False):
            x = torch.randn(4_097, 512).to(BF16)
            w = torch.randn(512, 2048).to(BF16)
            dz = torch.randn(4_097, 2048).to(BF16)
            with patch.object(torch, "bmm", _no_bmm):
                got = ta.triton_addmm_bwd(x, w, dz, is_y_1d)
            for p, q in zip(got, _orig_triton_addmm_bwd(x, w, dz, is_y_1d)):
                self.assertTrue(torch.equal(p, q))

    def test_triton_addmm_bwd_body_self_contained(self):
        # same AST extraction as dlrm_v4/train/tests/preprocess_backward_stages_test.py
        module = ast.parse(ADDMM_PATH.read_text())
        fn = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "triton_addmm_bwd")
        src = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), fn],
                         type_ignores=[])
        ast.fix_missing_locations(src)
        ns = {"torch": torch}
        exec(compile(src, str(ADDMM_PATH), "exec"), ns)
        x, w, dz = torch.randn(7, 5, dtype=torch.float64), torch.randn(5, 6, dtype=torch.float64), torch.randn(7, 6, dtype=torch.float64)
        for p, q in zip(ns["triton_addmm_bwd"](x, w, dz, True), _orig_triton_addmm_bwd(x, w, dz, True)):
            torch.testing.assert_close(p, q, rtol=0, atol=0)


def _import_flags(env):
    code = ("import generative_recommenders.ops.triton.triton_addmm as t;"
            "print(t._SPLITT_HSTU, t._SPLITT_PREPROC, sorted(t._SPLITT_SPLITS.items()), t._SPLITT_MIN_ROWS,"
            " t._splitt_wgrad_mm_hstu is not None)")
    full = {k: v for k, v in os.environ.items()
            if not k.startswith(("HSTU_SPLITT", "HSTU_TN_WGRAD", "HIPBLASLT_TENSILE_LIBPATH"))}
    full.update(env)
    full["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + full.get("PYTHONPATH", "")
    return subprocess.run([sys.executable, "-c", code], env=full, capture_output=True, text=True, timeout=600)


@unittest.skipIf(torch.version.hip is None, "M5 is ROCm-only (flags stay off elsewhere)")
class EnvTest(unittest.TestCase):
    def test_env_modes(self):
        r = _import_flags({})
        self.assertEqual(r.stdout.strip(), "False False [] 4096 False", r.stderr[-2000:])
        r = _import_flags({"HSTU_SPLITT_WGRAD": "all", "HSTU_SPLITT_WGRAD_SPLITS": "2048x512:8,24x256:128",
                           "HSTU_SPLITT_MIN_ROWS": "1024"})
        self.assertEqual(r.stdout.strip(), "True True [((24, 256), 128), ((2048, 512), 8)] 1024 True", r.stderr[-2000:])
        r = _import_flags({"HSTU_SPLITT_WGRAD": "preproc", "HSTU_TN_WGRAD": "hstu",
                           "HSTU_SPLITT_WGRAD_SPLITS": "256x512:64"})
        self.assertEqual(r.stdout.strip(), "False True [((256, 512), 64)] 4096 False", r.stderr[-2000:])
        # off: SPLITS / MIN_ROWS are not even parsed
        r = _import_flags({"HSTU_SPLITT_WGRAD": "0", "HSTU_SPLITT_WGRAD_SPLITS": "garbage", "HSTU_SPLITT_MIN_ROWS": "x"})
        self.assertEqual(r.stdout.strip(), "False False [] 4096 False", r.stderr[-2000:])

    def test_env_errors(self):
        for env in ({"HSTU_SPLITT_WGRAD": "yes"},
                    {"HSTU_SPLITT_WGRAD": "hstu", "HSTU_TN_WGRAD": "all"},
                    {"HSTU_SPLITT_WGRAD": "preproc", "HSTU_TN_WGRAD": "preproc"},
                    {"HSTU_SPLITT_WGRAD": "all", "HSTU_SPLITT_WGRAD_SPLITS": "2048x512"}):
            with self.subTest(env=env):
                r = _import_flags(env)
                self.assertNotEqual(r.returncode, 0)
                self.assertIn("ValueError", r.stderr)


if __name__ == "__main__":
    unittest.main()
