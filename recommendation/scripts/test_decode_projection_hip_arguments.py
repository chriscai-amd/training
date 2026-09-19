#!/usr/bin/env python3
"""CPU-only regression checks for argument parsing and trace attribution."""
import json
from pathlib import Path
import struct
import tempfile
import unittest

from decode_projection_hip_arguments import FIELDS, SIZES, analyze_trace, decode_capture, launch_grid, validate_metadata


def buffer(grid=32660):
    raw = bytearray(212)
    struct.pack_into("<8I", raw, 0, 1, 0x200000, 0x40010008, grid, 512, 1959558, 1, 2048)
    struct.pack_into("<6Q", raw, 32, 0x1234, 0x1234, 0x5678, 0x9abc, 0x10000, 0x20000)
    struct.pack_into("<2f", raw, 112, 1, 0)
    struct.pack_into("<6I", raw, 120, 16, 0x10000000, 0, 16 if grid == 32660 else 25,
                     grid, 32660 if grid == 32660 else 404)
    return raw


def capture(raw):
    return {"status": "captured", "buffer_size": len(raw), "buffer_hex": raw.hex()}


class DecoderTest(unittest.TestCase):
    def test_metadata_validation(self):
        # Generated transport fixture exercises parsing and rejection gates.
        # Real captured metadata is also validated by each decoder CLI report.
        blocks = ["  - .args:\n"]
        for name, offset, typ in FIELDS:
            kind = "global_buffer" if typ == "pointer" else "by_value"
            vtype = "bf16" if typ == "pointer" else typ
            blocks.append(f"      - .name: {name}\n        .offset: {offset}\n"
                          f"        .size: {SIZES[typ]}\n        .value_kind: {kind}\n"
                          f"        .value_type: {vtype}\n")
        blocks.append("    .kernarg_segment_size: 216\n    .kernarg_segment_align: 8\n"
                      "    .name: test_MT128x240x128_kernel\n    .language_version:\n"
                      "      - 2\ncustom.config:\n  InternalSupportParams:\n    KernArgsVersion: 2\n")
        good = "".join(blocks)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "metadata.txt"
            p.write_text(good)
            self.assertEqual(validate_metadata(p)["aligned_kernarg_segment_size"], 216)
            edits = [good.replace("segment_size: 216", "segment_size: 212"),
                     good.replace("    .kernarg_segment_size: 216\n", ""),
                     good.replace("segment_align: 8", "segment_align: 4"),
                     good.replace("KernArgsVersion: 2", "KernArgsVersion: 3"),
                     good.replace(".value_kind: global_buffer", ".value_kind: by_value", 1),
                     good.replace(".value_type: u32", ".value_type: f32", 1),
                     good.replace(".offset: 32", ".offset: 36", 1)]
            for text in edits:
                p.write_text(text)
                with self.assertRaises(ValueError):
                    validate_metadata(p)

    def test_212_and_216(self):
        for padding in (b"", b"\x00\x01\x02\x03"):
            decoded = decode_capture(capture(buffer() + padding))
            self.assertEqual(decoded["status"], "decoded_inline")
            self.assertEqual(decoded["fields"]["AddressWS"]["value"], "0x10000")
            self.assertEqual(decoded["fields"]["SizesFree1"]["value"], 1959558)
            self.assertEqual(decoded["tail_padding_hex"], padding.hex())
            self.assertEqual(decoded["geometry"]["logical_tiles"], 32660)
            self.assertEqual(decoded["header"]["wgmxccgroup_bits22_31"], 256)

    def test_persistent_remainder_uses_floor(self):
        d = decode_capture(capture(buffer(256)))
        self.assertEqual(d["geometry"]["tree_sk3_full_tiles_1_expected_skTiles"], 404)
        self.assertEqual(d["geometry"]["tree_sk3_full_tiles_1_expected_SKItersPerWG"], 25)

    def test_indirect_modes_only_envelope(self):
        for mode in (1, 2):
            raw = struct.pack("<4IQ", (mode << 30) | 3, 0, 0, 2, 0xffffffffffffffff)
            d = decode_capture(capture(raw))
            self.assertEqual(d["status"], "indirect_body_not_decoded")
            self.assertEqual(d["device_argument_pointer"], "0xffffffffffffffff")
            self.assertNotIn("fields", d)

    def test_mode3_inline(self):
        raw = buffer()
        struct.pack_into("<I", raw, 0, (3 << 30) | 1)
        self.assertEqual(decode_capture(capture(raw))["status"], "decoded_inline")

    def test_truncation(self):
        for n in (0, 3, 15):
            self.assertEqual(decode_capture(capture(buffer()[:n]))["status"], "truncated_header")
        for n in (16, 24, 80, 211):
            self.assertEqual(decode_capture(capture(buffer()[:n]))["status"], "truncated_inline")
        raw = struct.pack("<4I", (1 << 30) | 1, 0, 0, 0)
        self.assertEqual(decode_capture(capture(raw))["status"], "truncated_indirect_envelope")

    def test_invalid_capture_bounds_and_hex(self):
        good = capture(buffer())
        for edit in ({"buffer_size": 213}, {"buffer_size": -1}, {"buffer_size": True},
                     {"buffer_size": 4097}, {"buffer_hex": "z" * 424},
                     {"buffer_hex": " " * 424}, {"buffer_hex": None}):
            self.assertEqual(decode_capture(dict(good, **edit))["status"], "invalid_capture")
        self.assertEqual(decode_capture(capture(buffer() + b"x"))["status"], "unexpected_inline_size")

    def test_nonfinite_and_signed_raw(self):
        raw = buffer()
        struct.pack_into("<I", raw, 112, 0x7fa12345)
        struct.pack_into("<q", raw, 180, -1)
        d = decode_capture(capture(raw))
        self.assertEqual(d["fields"]["alpha"]["value"], "NaN")
        self.assertEqual(d["fields"]["alpha"]["raw_hex_le"], "4523a17f")
        self.assertEqual(d["fields"]["batchOffsetD"]["signed_value"], -1)
        json.dumps(d, allow_nan=False)

    def test_module_vs_ext_grid(self):
        row = {"api": "hipExtModuleLaunchKernel", "grid_or_global": [32768, 1, 1],
               "block_or_local": [128, 1, 1]}
        self.assertEqual(launch_grid(row), [256, 1, 1])
        self.assertEqual(launch_grid(dict(row, api="hipModuleLaunchKernel")), [32768, 1, 1])
        self.assertIsNone(launch_grid(dict(row, grid_or_global=[32769, 1, 1])))

    def test_exact_name_pid_return_attribution(self):
        selected = {"pid": 1, "event": "launch_begin", "launch_id": 1, "name": "exact",
                    "api": "hipExtModuleLaunchKernel", "grid_or_global": [4180480, 1, 1],
                    "block_or_local": [128, 1, 1], "argument_capture": capture(buffer())}
        rows = [selected, dict(selected, name="other", pid=2),
                {"pid": 2, "event": "launch_return", "launch_id": 1, "hip_error": 0}]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "trace.jsonl"
            p.write_text("".join(json.dumps(r) + "\n" for r in rows))
            r = analyze_trace(p, {"kernel_name": "exact"}, True)
            self.assertEqual(r["selected_launches"], 1)
            self.assertEqual(r["selected_returns"], 0)
            self.assertEqual(r["selected_successful_returns"], 0)
            with p.open("a") as f:
                f.write('{"event":')
            self.assertEqual(len(analyze_trace(p, {"kernel_name": "exact"})["issues"]), 1)
            with self.assertRaises(ValueError):
                analyze_trace(p, {"kernel_name": "exact"}, True)


if __name__ == "__main__":
    unittest.main()
