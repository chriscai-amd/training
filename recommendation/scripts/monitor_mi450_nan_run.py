#!/usr/bin/env python3
"""Passively monitor one MI450 training run; never control its workload.

Input files in --run-dir are training.log (MLLOG), driver.log (a newline-
terminated EXIT=<code> marker), and dmesg.log. JSON output uses strings for
nonfinite floats. A stall records an incident but does not fail a subsequently
successful, fault-free run.
"""

import argparse
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from datetime import datetime, timezone


FAULT = re.compile(
    r"\bMES(?:\s*\([^)]*\))?\s+failed to respond|ring buffer (?:is )?full"
    r"|GPU (?:memory |page )?fault|Memory access fault by GPU node"
    r"|(?:amdgpu|gfxhub|mmhub).*?(?:page|memory|protection) fault"
    r"|VM_L2_PROTECTION_FAULT",
    re.IGNORECASE,
)
EXIT = re.compile(r"(?:^|\s)EXIT=(-?\d+)(?=\s|$)")


def timestamp(now):
    return datetime.fromtimestamp(now, timezone.utc).isoformat(timespec="milliseconds")


def json_float(value):
    if math.isfinite(value):
        return value
    return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        return None


def write_json(path, data, first_only=False):
    """Publish a complete JSON file, optionally without replacing evidence."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if first_only:
            try:
                os.link(temporary, path)
            except FileExistsError:
                pass
        else:
            os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class Tail:
    """Read only newly appended bytes, retaining an unfinished line."""

    def __init__(self, path):
        self.path = path
        self.offset = 0
        self.pending = b""
        self.identity = None
        self.line_number = 0

    def read(self, final=False):
        try:
            stream = self.path.open("rb")
        except FileNotFoundError:
            return []
        with stream:
            stat = os.fstat(stream.fileno())
            identity = (stat.st_dev, stat.st_ino)
            if identity != self.identity or stat.st_size < self.offset:
                self.offset, self.pending, self.line_number = 0, b"", 0
                self.identity = identity
            stream.seek(self.offset)
            data = stream.read()
            self.offset = stream.tell()
        lines = (self.pending + data).split(b"\n")
        self.pending = lines.pop()
        if final and self.pending:
            lines.append(self.pending)
            self.pending = b""
        result = []
        for line in lines:
            self.line_number += 1
            result.append((self.line_number, line.decode("utf-8", errors="replace").rstrip("\r")))
        return result


class Monitor:
    def __init__(self, args, now=None):
        self.args = args
        self.directory = Path(args.run_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.started = time.time() if now is None else now
        self.last_progress = self.started
        self.training = Tail(self.directory / "training.log")
        self.driver = Tail(self.directory / "driver.log")
        self.dmesg = Tail(self.directory / "dmesg.log")
        self.state = {
            "status": "starting", "started_at": timestamp(self.started),
            "last_step": None, "last_loss": None, "loss_count": 0,
            "samples_count": None, "last_progress_at": None,
            "first_nonfinite": read_json(self.directory / "first_nonfinite.json"),
            "first_stall": read_json(self.directory / "first_stall.json"),
            "gpu_faults": [], "gpu_fault_count": 0, "driver_exit_code": None,
            "min_steps": args.min_steps,
        }

    def incident(self, name, evidence):
        if self.state[name] is None:
            path = self.directory / (name + ".json")
            write_json(path, evidence, first_only=True)
            self.state[name] = read_json(path)

    def train_loss(self, number, line, now):
        if not line.startswith(":::MLLOG "):
            return
        try:
            event = json.loads(line[len(":::MLLOG "):])
            if event.get("key") != "train_loss":
                return
            loss = float(event["value"])
            samples = int(event["metadata"]["samples_count"])
        except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
            return
        step = samples // 1024
        state = self.state
        if state["last_step"] is None or step > state["last_step"]:
            self.last_progress = now
            state["last_progress_at"] = timestamp(now)
        state["last_step"] = max(step, state["last_step"] if state["last_step"] is not None else step)
        state["last_loss"] = json_float(loss)
        state["samples_count"] = samples
        state["loss_count"] += 1
        if not math.isfinite(loss):
            self.incident("first_nonfinite", {
                "observed_at": timestamp(now), "step": step, "loss": json_float(loss),
                "samples_count": samples, "loss_count": state["loss_count"],
                "line_number": number, "line": line,
            })

    def observe_fault(self, source, number, line, now):
        if FAULT.search(line):
            self.state["gpu_fault_count"] += 1
            if len(self.state["gpu_faults"]) < 100:
                self.state["gpu_faults"].append({
                    "observed_at": timestamp(now), "source": source,
                    "line_number": number, "line": line,
                })

    def poll(self, now=None):
        now = time.time() if now is None else now
        state = self.state
        for _, line in self.driver.read():
            match = EXIT.search(line)
            if match:
                state["driver_exit_code"] = int(match.group(1))
        finished = state["driver_exit_code"] is not None
        for number, line in self.training.read(final=finished):
            self.observe_fault("training.log", number, line, now)
            self.train_loss(number, line, now)
        for number, line in self.dmesg.read(final=finished):
            self.observe_fault("dmesg.log", number, line, now)
        elapsed = max(0.0, now - self.last_progress)
        startup = state["loss_count"] == 0
        threshold = self.args.startup_stall_seconds if startup else self.args.stall_seconds
        state["stalled"] = not finished and elapsed >= threshold
        state["seconds_since_progress"] = round(elapsed, 3)
        if state["stalled"]:
            self.incident("first_stall", {
                "observed_at": timestamp(now), "phase": "startup" if startup else "training",
                "last_step": state["last_step"], "last_loss": state["last_loss"],
                "last_progress_at": state["last_progress_at"],
                "seconds_without_progress": round(elapsed, 3), "threshold_seconds": threshold,
            })
        if finished:
            reasons = []
            if state["driver_exit_code"] != 0:
                reasons.append("driver exited with code " + str(state["driver_exit_code"]))
            if state["last_step"] is None or state["last_step"] < self.args.min_steps:
                reasons.append("minimum step count not reached")
            if state["first_nonfinite"] is not None:
                reasons.append("nonfinite training loss observed")
            if state["gpu_fault_count"]:
                reasons.append("GPU fault observed")
            state["status"] = "failed" if reasons else "passed"
            state["failure_reasons"] = reasons
            state["finished_at"] = timestamp(now)
        else:
            state["status"] = (
                "gpu_fault" if state["gpu_fault_count"] else
                "nonfinite" if state["first_nonfinite"] is not None else
                "stalled" if state["stalled"] else
                "starting" if startup else "running"
            )
        state["updated_at"] = timestamp(now)
        write_json(self.directory / "status.json", state)
        return finished


def positive_seconds(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--poll-seconds", default=30, type=positive_seconds)
    parser.add_argument("--stall-seconds", default=300, type=positive_seconds)
    parser.add_argument("--startup-stall-seconds", default=900, type=positive_seconds)
    parser.add_argument("--min-steps", default=3000, type=int)
    args = parser.parse_args()
    if args.min_steps < 1:
        parser.error("--min-steps must be positive")
    monitor = Monitor(args)
    previous_status = None
    while True:
        finished = monitor.poll()
        state = monitor.state
        if state["status"] != previous_status or finished:
            print(f'{state["updated_at"]} status={state["status"]} step={state["last_step"]} loss={state["last_loss"]}', flush=True)
            previous_status = state["status"]
        if finished:
            return 0 if state["status"] == "passed" else 1
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
