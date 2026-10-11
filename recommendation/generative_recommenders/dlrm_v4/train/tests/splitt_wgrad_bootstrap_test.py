# pyre-unsafe
"""CPU-only tests for the OPT M5 gin switch (apply_hipblaslt_opts.splitt_*) and overlays."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

import gin
from generative_recommenders.dlrm_v4.train import _env_bootstrap as eb

GIN_DIR = Path(eb.__file__).parent / "gin"
KEYS = ("HSTU_TN_WGRAD", "HSTU_SPLITT_WGRAD", "HSTU_SPLITT_WGRAD_SPLITS", "HSTU_SPLITT_MIN_ROWS",
        "HIPBLASLT_TUNING_OVERRIDE_FILE", "DLRM_GIN_OVERLAY", "HSTU_SPLITT_AUDIT_EVERY")
ARMLIBS = Path("/armlibs")   # run_ab.sh container mount of arbor_libs (tests: -v .../arbor_libs:/armlibs:ro)


def _clean_env():
    return {k: v for k, v in os.environ.items() if k not in KEYS}


class ApplyOptsTest(unittest.TestCase):
    def setUp(self):
        gin.clear_config()

    def tearDown(self):
        gin.clear_config()

    def run_opts(self, **kw):
        with patch.dict(os.environ, _clean_env(), clear=True):
            eb.apply_hipblaslt_opts(**kw)
            return {k: os.environ.get(k) for k in KEYS[:4]}

    def test_default_off(self):
        self.assertEqual(self.run_opts(), {"HSTU_TN_WGRAD": "0", "HSTU_SPLITT_WGRAD": "0",
                                           "HSTU_SPLITT_WGRAD_SPLITS": None, "HSTU_SPLITT_MIN_ROWS": None})

    def test_on(self):
        env = self.run_opts(splitt_wgrad_hstu=True, splitt_wgrad_preproc=True,
                            splitt_wgrad_splits={"512x1536": 10, "2048x512": 8}, splitt_min_rows=2048)
        self.assertEqual(env, {"HSTU_TN_WGRAD": "0", "HSTU_SPLITT_WGRAD": "all",
                               "HSTU_SPLITT_WGRAD_SPLITS": "2048x512:8,512x1536:10",
                               "HSTU_SPLITT_MIN_ROWS": "2048"})
        self.assertEqual(self.run_opts(splitt_wgrad_preproc=True, tn_wgrad_hstu=True)["HSTU_SPLITT_WGRAD"], "preproc")
        self.assertEqual(self.run_opts(splitt_wgrad_hstu=True)["HSTU_SPLITT_WGRAD"], "hstu")

    def test_errors(self):
        for kw in ({"splitt_wgrad_hstu": True, "tn_wgrad_hstu": True},
                   {"splitt_wgrad_preproc": True, "tn_wgrad_preproc": True},
                   {"splitt_wgrad_hstu": True, "splitt_wgrad_splits": {"2048 x 512": 8}},
                   {"splitt_wgrad_hstu": True, "splitt_wgrad_splits": {"2048x512": 0}},
                   {"splitt_wgrad_hstu": True, "splitt_min_rows": 0},
                   {"splitt_wgrad_hstu": True, "splitt_audit_every": -1}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                self.run_opts(**kw)

    def test_audit_env(self):
        with patch.dict(os.environ, _clean_env(), clear=True):
            eb.apply_hipblaslt_opts()
            self.assertIsNone(os.environ.get("HSTU_SPLITT_AUDIT_EVERY"))
            eb.apply_hipblaslt_opts(splitt_wgrad_preproc=True, splitt_audit_every=50)
            self.assertEqual(os.environ.get("HSTU_SPLITT_AUDIT_EVERY"), "50")

    def test_preset_env_wins(self):
        with patch.dict(os.environ, dict(_clean_env(), HSTU_SPLITT_WGRAD="0"), clear=True):
            eb.apply_hipblaslt_opts(splitt_wgrad_hstu=True, splitt_wgrad_splits={"2048x512": 8})
            self.assertEqual(os.environ["HSTU_SPLITT_WGRAD"], "0")

    def gin_env(self, overlay):
        env = _clean_env()
        if overlay:
            env["DLRM_GIN_OVERLAY"] = str(GIN_DIR / "mi450_hipblaslt" / overlay)
        with patch.dict(os.environ, env, clear=True):
            eb.parse_gin(str(GIN_DIR / "yambda_5b.gin"), skip_unknown=True)
            eb.apply_hipblaslt_opts()
            out = {k: os.environ.get(k) for k in KEYS[:4]}
        out["lib"] = gin.query_parameter("apply_launch_env.hipblaslt_lib_dir")
        return out

    def test_gin_default_and_overlays(self):
        self.assertEqual(self.gin_env(None), {"HSTU_TN_WGRAD": "0", "HSTU_SPLITT_WGRAD": "0",
                                              "HSTU_SPLITT_WGRAD_SPLITS": None, "HSTU_SPLITT_MIN_ROWS": None,
                                              "lib": None})
        for ovl in ("m0_off.gin", "m1_lib.gin", "m2_override.gin"):
            gin.clear_config()
            self.assertEqual(self.gin_env(ovl)["HSTU_SPLITT_WGRAD"], "0", ovl)
        gin.clear_config()
        self.assertEqual(self.gin_env("m5_split_t.gin"), {
            "HSTU_TN_WGRAD": "0", "HSTU_SPLITT_WGRAD": "all",
            "HSTU_SPLITT_WGRAD_SPLITS": "1024x256:32,2048x512:8,24x256:128,256x512:64,512x1536:10,512x256:64",
            "HSTU_SPLITT_MIN_ROWS": "4096", "lib": "/armlibs/rocm-10.2_nnonly_v3/s103"})
        gin.clear_config()
        self.assertEqual(self.gin_env("m5_split_t_s102.gin"), {
            "HSTU_TN_WGRAD": "0", "HSTU_SPLITT_WGRAD": "all",
            "HSTU_SPLITT_WGRAD_SPLITS": "1024x256:2,2048x512:4,24x256:64,256x512:4,512x256:4",
            "HSTU_SPLITT_MIN_ROWS": "4096", "lib": "/armlibs/rocm-10.2_nnonly_v3/s102"})
        gin.clear_config()
        self.assertEqual(self.gin_env("m5_split_t_mix.gin"), {
            "HSTU_TN_WGRAD": "0", "HSTU_SPLITT_WGRAD": "all",
            "HSTU_SPLITT_WGRAD_SPLITS": "1024x256:2,2048x512:8,24x256:64,256x512:4,512x1536:10,512x256:4",
            "HSTU_SPLITT_MIN_ROWS": "4096", "lib": "/armlibs/rocm-10.2_nnonly_v3/mix"})

    @unittest.skipUnless((ARMLIBS / "rocm-10.2_nnonly_v3").is_dir(), "arbor_libs not mounted at /armlibs")
    def test_overlay_splits_pinned_by_installed_catalogs(self):
        from generative_recommenders.ops.triton import triton_addmm as ta
        for ovl, variant in (("m5_split_t.gin", "s103"), ("m5_split_t_s102.gin", "s102"),
                             ("m5_split_t_mix.gin", "mix")):
            gin.clear_config()
            env = self.gin_env(ovl)
            v, exp, probs = ta._splitt_catalog_check(
                ta._parse_splitt(env["HSTU_SPLITT_WGRAD_SPLITS"]), env["lib"] + "/hipblaslt/library/gfx1250")
            with self.subTest(ovl):
                self.assertEqual((v, probs), (variant, []))
                self.assertEqual(len(exp), len(ta._parse_splitt(env["HSTU_SPLITT_WGRAD_SPLITS"])))
                want = {"s103": {103}, "s102": {102}, "mix": {102, 103}}[variant]
                self.assertEqual(set(exp.values()), want)
        # an s the s103 catalog does not pin (G19 s=12) is reported
        _, _, probs = ta._splitt_catalog_check({(512, 1536): 12},
                                               str(ARMLIBS / "rocm-10.2_nnonly_v3/s103/hipblaslt/library/gfx1250"))
        self.assertEqual(len(probs), 1)
        self.assertIn("selects sol 102", probs[0])


if __name__ == "__main__":
    unittest.main()
