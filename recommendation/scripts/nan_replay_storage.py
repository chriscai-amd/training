"""Byte-exact rolling tensor storage snapshots with one full CPU shadow.

The constructor captures a boundary.  ``update`` compares *every byte* of every
untyped storage, journals old pages to disk, and advances the shadow.  Thus the
current shadow plus the last undo journal also represents the previous boundary.
Aliased tensor views share one shadow allocation.  Nothing infers which embedding
rows an optimizer ought to have written, and floating point values are never
compared as floating point values.

``persist`` streams the shadow to raw files and copies the undo journal.  A
manifest is committed only after those files have been fsynced.  It never clones
the full shadow.  Working memory beyond the shadow is a few bounded chunks;
journal payloads and record indexes are never accumulated in Python containers.

State collectors must include all live tensors, including optimizer/private
buffers.  Non-tensor controls belong to the caller.  On replay restore controls,
recollect live tensors, then call ``restore_snapshot``.  Tensor binding topology
must match exactly, including storage aliases and otherwise-unexposed bytes.

All accelerator operations occur only when an API is called with accelerator
tensors.  CPU tests and manifest inspection do not initialize an accelerator.
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import struct
import uuid
import warnings
from typing import Any, BinaryIO, Dict, Iterator, Mapping, Optional, Tuple

import torch


DEFAULT_CHUNK_BYTES = 64 << 20
DEFAULT_PAGE_BYTES = 16 << 10
_FORMAT = "nan-replay-storage"
_VERSION = 1
_MAGIC = b"NANUND01"
_HEADER = struct.Struct("<8sQQ")
_RECORD = struct.Struct("<QQQ")


def _chunk_size(chunk_bytes: int, page_bytes: int) -> int:
    if page_bytes <= 0 or chunk_bytes < page_bytes or chunk_bytes % page_bytes:
        raise ValueError("chunk_bytes must be a positive multiple of page_bytes")
    return chunk_bytes


def _memory(tensor: torch.Tensor) -> memoryview:
    # uint8 CPU tensors are contiguous here; numpy() and memoryview are views.
    return memoryview(tensor.numpy())


def _write_all(stream: BinaryIO, data: memoryview) -> None:
    while data:
        written = stream.write(data)
        if written is None or written <= 0:
            raise OSError("short write while saving tensor storage")
        data = data[written:]


def _read_into(stream: BinaryIO, data: memoryview) -> None:
    while data:
        count = stream.readinto(data)
        if not count:
            raise ValueError("truncated tensor storage snapshot")
        data = data[count:]


def _fsync_dir(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _drop_cache(stream: BinaryIO) -> None:
    # Clean file cache must not needlessly compete with the large CPU shadow.
    if hasattr(os, "posix_fadvise"):
        try:
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass


def _unlink_owned(path: Optional[Path]) -> None:
    if path is not None:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            warnings.warn(f"could not remove expired undo journal {path}: {exc}")


def _describe(live_tensors: Mapping[str, torch.Tensor]):
    if not isinstance(live_tensors, Mapping):
        raise TypeError("live_tensors must map stable string names to live tensors")
    if any(not isinstance(name, str) for name in live_tensors):
        raise TypeError("tensor binding names must be strings")
    bindings, storages, byte_views = [], [], []
    storage_ids = {}
    for name in sorted(live_tensors):
        tensor = live_tensors[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"binding {name!r} is not a tensor")
        if tensor.layout != torch.strided or tensor.is_quantized:
            raise TypeError(f"binding {name!r} must be a non-quantized strided tensor")
        if tensor.device.type not in ("cpu", "cuda"):
            raise TypeError(f"binding {name!r} has unsupported device {tensor.device}")
        storage = tensor.untyped_storage()
        identity = (str(tensor.device), storage._cdata)
        storage_id = storage_ids.get(identity)
        if storage_id is None:
            storage_id = len(storages)
            storage_ids[identity] = storage_id
            nbytes = storage.nbytes()
            storages.append({"id": storage_id, "nbytes": nbytes,
                             "device": str(tensor.device)})
            byte_views.append(torch.empty(0, dtype=torch.uint8, device=tensor.device)
                              .set_(storage, 0, (nbytes,), (1,)))
        bindings.append({
            "name": name, "storage_id": storage_id,
            "dtype": str(tensor.dtype), "shape": list(tensor.shape),
            "stride": list(tensor.stride()), "storage_offset": tensor.storage_offset(),
            "requires_grad": tensor.requires_grad, "is_conj": tensor.is_conj(),
            "is_neg": tensor.is_neg(),
        })
    return {"bindings": bindings, "storages": storages}, byte_views


def _validate_topology(expected: dict, actual: dict) -> None:
    if expected == actual:
        return
    expected_bindings = {item["name"]: item for item in expected["bindings"]}
    actual_bindings = {item["name"]: item for item in actual["bindings"]}
    if expected_bindings.keys() != actual_bindings.keys():
        missing = sorted(expected_bindings.keys() - actual_bindings.keys())[:8]
        added = sorted(actual_bindings.keys() - expected_bindings.keys())[:8]
        raise ValueError(f"tensor state topology changed: missing={missing}, added={added}")
    for name in expected_bindings:
        if expected_bindings[name] != actual_bindings[name]:
            raise ValueError(f"tensor binding topology mismatch for {name!r}: "
                             f"expected {expected_bindings[name]}, got {actual_bindings[name]}")
    raise ValueError("storage topology mismatch: storage byte sizes or devices differ")


def describe_tensor_state(live_tensors: Mapping[str, torch.Tensor]):
    """Return JSON topology and deduplicated live 1D uint8 storage views.

    This allocates view metadata only, performs no device copies, and does not
    initialize accelerator state for CPU inputs.  The returned views expose the
    entire allocations, including bytes outside the supplied tensor slices.
    """
    return _describe(live_tensors)


def validate_tensor_topology(expected: dict, actual: dict) -> None:
    """Validate two topologies returned by ``describe_tensor_state``."""
    _validate_topology(expected, actual)


def _synchronize(byte_views) -> None:
    for device in sorted({str(t.device) for t in byte_views if t.device.type == "cuda"}):
        torch.cuda.synchronize(torch.device(device))


def _changed_runs(old: torch.Tensor, new: torch.Tensor, page_bytes: int):
    """Yield page-aligned differing runs; examine all bytes, including padding."""
    nbytes = old.numel()
    full = nbytes // page_bytes * page_bytes
    pages = (old[:full] != new[:full]).view(-1, page_bytes).any(dim=1).tolist()
    if full != nbytes:
        pages.append(not torch.equal(old[full:], new[full:]))
    start = None
    for index, changed in enumerate(pages):
        if changed and start is None:
            start = index * page_bytes
        elif not changed and start is not None:
            yield start, index * page_bytes
            start = None
    if start is not None:
        yield start, nbytes


class _UndoCursor:
    """One journal record of lookahead, with bounded readinto payload access."""

    def __init__(self, path: Path, topology: dict, page_bytes: int):
        self.stream = path.open("rb", buffering=0)
        self.topology = topology
        self.page_bytes = page_bytes
        self.file_size = os.fstat(self.stream.fileno()).st_size
        self.next_header = _HEADER.size
        self.record = None
        self.previous_end = (-1, 0)
        self.records = self.payload_bytes = self.pages = 0
        try:
            raw = self.stream.read(_HEADER.size)
            if len(raw) != _HEADER.size or _HEADER.unpack(raw) != (
                    _MAGIC, page_bytes, len(topology["storages"])):
                raise ValueError("invalid undo journal header")
            self.advance()
        except BaseException:
            self.stream.close()
            raise

    def close(self):
        self.stream.close()

    def advance(self):
        self.stream.seek(self.next_header)
        raw = self.stream.read(_RECORD.size)
        if not raw:
            self.record = None
            return
        if len(raw) != _RECORD.size:
            raise ValueError("truncated undo journal record header")
        storage_id, offset, length = _RECORD.unpack(raw)
        if storage_id >= len(self.topology["storages"]):
            raise ValueError("undo journal references an unknown storage")
        storage_bytes = self.topology["storages"][storage_id]["nbytes"]
        if (length <= 0 or offset % self.page_bytes or offset + length > storage_bytes
                or (length % self.page_bytes and offset + length != storage_bytes)):
            raise ValueError("invalid page bounds in undo journal")
        if (storage_id, offset) < self.previous_end:
            raise ValueError("unordered or overlapping undo journal records")
        payload_offset = self.next_header + _RECORD.size
        self.next_header = payload_offset + length
        if self.next_header > self.file_size:
            raise ValueError("truncated undo journal payload")
        self.previous_end = (storage_id, offset + length)
        self.record = (storage_id, offset, length, payload_offset)
        self.records += 1
        self.payload_bytes += length
        self.pages += (length + self.page_bytes - 1) // self.page_bytes

    def overlay(self, storage_id: int, offset: int, chunk: torch.Tensor):
        end = offset + chunk.numel()
        while self.record is not None:
            record_id, record_start, length, payload_offset = self.record
            record_end = record_start + length
            if record_id > storage_id or (record_id == storage_id and record_start >= end):
                break
            if record_id < storage_id or record_end <= offset:
                raise ValueError("undo journal iteration skipped a record")
            lo, hi = max(offset, record_start), min(end, record_end)
            self.stream.seek(payload_offset + lo - record_start)
            _read_into(self.stream, _memory(chunk[lo - offset:hi - offset]))
            if hi == record_end:
                self.advance()
            else:
                break


def _validate_journal(path: Path, topology: dict, page_bytes: int,
                      expected: Optional[dict] = None) -> dict:
    with contextlib.closing(_UndoCursor(path, topology, page_bytes)) as cursor:
        while cursor.record is not None:
            cursor.advance()
        result = {"records": cursor.records, "payload_bytes": cursor.payload_bytes,
                  "changed_pages": cursor.pages, "file_bytes": cursor.file_size}
    if expected is not None and any(result[key] != expected[key] for key in result):
        raise ValueError("undo journal sizes/counts do not match completion manifest")
    return result


def _copy_file(source: Path, destination: Path, chunk_bytes: int) -> None:
    buffer = bytearray(min(chunk_bytes, max(1, source.stat().st_size)))
    with source.open("rb", buffering=0) as src, destination.open("xb", buffering=0) as dst:
        while True:
            count = src.readinto(buffer)
            if not count:
                break
            _write_all(dst, memoryview(buffer)[:count])
        os.fsync(dst.fileno())
        _drop_cache(dst)


class RollingTensorSnapshot:
    """One complete CPU shadow and one disk undo journal for the prior boundary.

    Caller must exclude concurrent state mutation for the duration of each API.
    Accelerator devices are synchronized before capture/compare.  Host staging
    memory is bounded; accelerator transfers use one pinned chunk, while the
    full CPU shadow stays pageable.
    """

    def __init__(self, live_tensors: Mapping[str, torch.Tensor], journal_dir,
                 *, boundary_id=None, chunk_bytes: int = DEFAULT_CHUNK_BYTES,
                 page_bytes: int = DEFAULT_PAGE_BYTES):
        self.chunk_bytes = _chunk_size(chunk_bytes, page_bytes)
        self.page_bytes = page_bytes
        self.journal_dir = Path(journal_dir)
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        self.topology, byte_views = _describe(live_tensors)
        json.dumps(boundary_id, allow_nan=False)
        self.boundary_id = boundary_id
        self.previous_boundary_id = None
        self._journal_path = None
        self._journal_stats = None
        self._closed = False
        self._valid = True
        self._shadow = []
        _synchronize(byte_views)
        stage = torch.empty(
            min(self.chunk_bytes, max((t.numel() for t in byte_views), default=0)),
            dtype=torch.uint8, device="cpu",
            pin_memory=any(t.device.type == "cuda" for t in byte_views),
        )
        with torch.no_grad():
            for view in byte_views:
                shadow = torch.empty(view.numel(), dtype=torch.uint8, device="cpu")
                for offset in range(0, view.numel(), self.chunk_bytes):
                    end = min(offset + self.chunk_bytes, view.numel())
                    current = stage[:end - offset]
                    current.copy_(view[offset:end], non_blocking=False)
                    shadow[offset:end].copy_(current)
                self._shadow.append(shadow)

    @property
    def total_bytes(self) -> int:
        return sum(storage["nbytes"] for storage in self.topology["storages"])

    @property
    def has_previous(self) -> bool:
        return self._journal_path is not None

    def _check_open(self):
        if self._closed or not self._valid:
            raise RuntimeError("rolling snapshot is closed or invalid after an I/O failure")

    def nonfinite_floating_ranges(self, storage_dtypes: Mapping[int, torch.dtype]) -> list:
        """Inspect the initial shadow, then every changed page at later boundaries.

        The caller stops on the first hit. After a finite initial scan, exact
        byte comparison proves unchanged pages remain finite. This uses CPU
        chunks and introduces no extra GPU kernels or full-size boolean mask.
        Only storages entirely occupied by one floating dtype may be supplied.
        """
        self._check_open()
        hits = []
        seen = set()

        def check(storage_id, begin, end):
            dtype = storage_dtypes.get(storage_id)
            if dtype is None or storage_id in seen:
                return
            width = torch.empty((), dtype=dtype).element_size()
            if begin % width or end % width:
                raise ValueError("floating state storage is not dtype-aligned")
            for offset in range(begin, end, self.chunk_bytes):
                part = self._shadow[storage_id][offset:min(offset + self.chunk_bytes, end)].view(dtype)
                bad = ~torch.isfinite(part)
                if bool(bad.any()):
                    index = int(bad.nonzero()[0].item())
                    hits.append({"storage_id": storage_id, "byte_offset": offset + index * width})
                    seen.add(storage_id)
                    break

        if self.has_previous:
            with contextlib.closing(_UndoCursor(self._journal_path, self.topology, self.page_bytes)) as cursor:
                while cursor.record is not None:
                    sid, offset, length, _ = cursor.record
                    check(sid, offset, offset + length)
                    cursor.advance()
        else:
            for sid, shadow in enumerate(self._shadow):
                check(sid, 0, shadow.numel())
        return hits

    def _rollback(self, path: Path, complete_bytes: int):
        with path.open("r+b", buffering=0) as journal:
            journal.truncate(complete_bytes)
        with contextlib.closing(_UndoCursor(path, self.topology, self.page_bytes)) as cursor:
            while cursor.record is not None:
                storage_id, offset, length, payload_offset = cursor.record
                cursor.stream.seek(payload_offset)
                for begin in range(offset, offset + length, self.chunk_bytes):
                    end = min(begin + self.chunk_bytes, offset + length)
                    _read_into(cursor.stream, _memory(self._shadow[storage_id][begin:end]))
                cursor.advance()

    def update(self, live_tensors: Mapping[str, torch.Tensor], *, boundary_id=None) -> dict:
        """Advance one boundary and atomically rotate its exact-page disk undo.

        A failed write rolls back the partially updated shadow using the complete
        journal prefix.  The older journal stays intact until commit succeeds.
        """
        self._check_open()
        json.dumps(boundary_id, allow_nan=False)
        topology, byte_views = _describe(live_tensors)
        _validate_topology(self.topology, topology)
        _synchronize(byte_views)
        stage = torch.empty(min(self.chunk_bytes, max((t.numel() for t in byte_views),
                                                      default=0)), dtype=torch.uint8,
                            device="cpu", pin_memory=any(t.device.type == "cuda" for t in byte_views))
        path = self.journal_dir / f"undo-{uuid.uuid4().hex}.bin"
        complete_bytes = 0
        stats = {"records": 0, "payload_bytes": 0, "changed_pages": 0}
        try:
            with path.open("xb", buffering=0) as journal, torch.no_grad():
                _write_all(journal, memoryview(_HEADER.pack(
                    _MAGIC, self.page_bytes, len(byte_views))))
                complete_bytes = _HEADER.size
                for storage_id, view in enumerate(byte_views):
                    shadow = self._shadow[storage_id]
                    for offset in range(0, view.numel(), self.chunk_bytes):
                        length = min(self.chunk_bytes, view.numel() - offset)
                        current = stage[:length]
                        current.copy_(view[offset:offset + length], non_blocking=False)
                        old = shadow[offset:offset + length]
                        for begin, end in _changed_runs(old, current, self.page_bytes):
                            _write_all(journal, memoryview(_RECORD.pack(
                                storage_id, offset + begin, end - begin)))
                            _write_all(journal, _memory(old[begin:end]))
                            complete_bytes = journal.tell()
                            old[begin:end].copy_(current[begin:end])
                            stats["records"] += 1
                            stats["payload_bytes"] += end - begin
                            stats["changed_pages"] += (end - begin + self.page_bytes - 1) // self.page_bytes
                os.fsync(journal.fileno())
                _drop_cache(journal)
            _fsync_dir(self.journal_dir)
        except BaseException:
            if complete_bytes:
                try:
                    self._rollback(path, complete_bytes)
                except BaseException:
                    self._valid = False
                    raise
            _unlink_owned(path)
            raise
        old_journal = self._journal_path
        self._journal_path = path
        stats["file_bytes"] = complete_bytes
        self._journal_stats = stats
        self.previous_boundary_id = self.boundary_id
        self.boundary_id = boundary_id
        _unlink_owned(old_journal)
        return dict(stats, boundary_id=boundary_id,
                    previous_boundary_id=self.previous_boundary_id, total_bytes=self.total_bytes)

    def persist(self, out_dir, *, metadata: Optional[dict] = None) -> Path:
        """Stream one full base and the previous undo, then commit manifest.json.

        The directory may already contain caller-owned control sidecars.  Files
        owned by this layer are created exclusively; an incomplete save has no
        completion manifest and cannot be restored.
        """
        self._check_open()
        manifest = {
            "format": _FORMAT, "version": _VERSION, "complete": True,
            "page_bytes": self.page_bytes, "chunk_bytes": self.chunk_bytes,
            "topology": self.topology, "total_bytes": self.total_bytes,
            "current": {"boundary_id": self.boundary_id}, "previous": None,
            "base_files": [f"storage-{i:06d}.bin" for i in range(len(self._shadow))],
            "metadata": metadata or {},
        }
        if self.has_previous:
            manifest["previous"] = {"boundary_id": self.previous_boundary_id,
                                    "undo_file": "previous.undo", **self._journal_stats}
        serialized = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        directory = Path(out_dir)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "manifest.json"
        if target.exists():
            raise FileExistsError(f"snapshot already complete: {target}")
        incomplete = directory / ".storage-incomplete"
        with incomplete.open("x") as marker:
            marker.write("storage capture in progress\n")
            marker.flush()
            os.fsync(marker.fileno())
        _fsync_dir(directory)
        for shadow, filename in zip(self._shadow, manifest["base_files"]):
            with (directory / filename).open("xb", buffering=0) as output:
                for offset in range(0, shadow.numel(), self.chunk_bytes):
                    _write_all(output, _memory(shadow[offset:offset + self.chunk_bytes]))
                os.fsync(output.fileno())
                _drop_cache(output)
        if self.has_previous:
            _validate_journal(self._journal_path, self.topology, self.page_bytes,
                              self._journal_stats)
            _copy_file(self._journal_path, directory / "previous.undo", self.chunk_bytes)
        pending = directory / ".manifest-pending.json"
        with pending.open("x") as output:
            output.write(serialized)
            output.flush()
            os.fsync(output.fileno())
        # Removing the marker before the final rename makes manifest.json the
        # sole completion decision, including after interruption between steps.
        incomplete.unlink()
        _fsync_dir(directory)
        os.replace(pending, target)
        _fsync_dir(directory)
        return target

    def compare_current(self, live_tensors: Mapping[str, torch.Tensor], *,
                        boundary: str = "current") -> dict:
        """Compare live bytes with either retained boundary without mutating them."""
        self._check_open()
        _check_boundary(boundary, self.has_previous)
        topology, views = _describe(live_tensors)
        _validate_topology(self.topology, topology)
        _synchronize(views)
        undo = self._journal_path if boundary == "previous" else None
        with _expected_chunks(self.topology, self.chunk_bytes, self.page_bytes,
                              shadows=self._shadow, undo=undo) as chunks:
            return _compare(views, chunks, self.chunk_bytes, self.page_bytes)

    def close(self):
        """Release the CPU shadow and remove this object's rotating journal."""
        if not self._closed:
            self._shadow.clear()
            _unlink_owned(self._journal_path)
            self._closed = True

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *unused):
        self.close()


def _check_boundary(boundary: str, has_previous: bool) -> None:
    if boundary not in ("current", "previous"):
        raise ValueError("boundary must be 'current' or 'previous'")
    if boundary == "previous" and not has_previous:
        raise ValueError("this snapshot has no previous boundary")


def _snapshot_path(directory: Path, filename: str) -> Path:
    if not isinstance(filename, str) or Path(filename).name != filename or filename in ("", ".", ".."):
        raise ValueError("snapshot filenames must be plain relative filenames")
    path = directory / filename
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"snapshot file is missing or is not a regular file: {filename}")
    return path


def load_snapshot_manifest(snapshot_dir) -> dict:
    """Read completion metadata and validate raw file sizes and journal bounds."""
    directory = Path(snapshot_dir)
    with (directory / "manifest.json").open() as stream:
        manifest = json.load(stream)
    if (manifest.get("format") != _FORMAT or manifest.get("version") != _VERSION
            or manifest.get("complete") is not True):
        raise ValueError("not a completed supported tensor storage snapshot")
    topology = manifest["topology"]
    _chunk_size(manifest["chunk_bytes"], manifest["page_bytes"])
    storages = topology["storages"]
    files = manifest["base_files"]
    if len(files) != len(storages) or len(set(files)) != len(files):
        raise ValueError("invalid storage base file list")
    for index, (filename, storage) in enumerate(zip(files, storages)):
        if storage["id"] != index or storage["nbytes"] < 0:
            raise ValueError("invalid storage table")
        if _snapshot_path(directory, filename).stat().st_size != storage["nbytes"]:
            raise ValueError(f"storage base size mismatch: {filename}")
    if manifest["total_bytes"] != sum(s["nbytes"] for s in storages):
        raise ValueError("storage total does not match completion manifest")
    previous = manifest["previous"]
    if previous is not None:
        _validate_journal(_snapshot_path(directory, previous["undo_file"]), topology,
                          manifest["page_bytes"], previous)
    return manifest


@contextlib.contextmanager
def _expected_chunks(topology: dict, chunk_bytes: int, page_bytes: int, *,
                     directory: Optional[Path] = None, files=None, shadows=None,
                     undo: Optional[Path] = None):
    max_bytes = min(chunk_bytes, max((s["nbytes"] for s in topology["storages"]), default=0))
    staging = torch.empty(max_bytes, dtype=torch.uint8, device="cpu")
    cursor = _UndoCursor(undo, topology, page_bytes) if undo is not None else None

    def generate() -> Iterator[Tuple[int, int, torch.Tensor]]:
        for storage_id, storage in enumerate(topology["storages"]):
            source = ((directory / files[storage_id]).open("rb", buffering=0)
                      if directory is not None else None)
            try:
                for offset in range(0, storage["nbytes"], chunk_bytes):
                    length = min(chunk_bytes, storage["nbytes"] - offset)
                    chunk = staging[:length]
                    if source is not None:
                        _read_into(source, _memory(chunk))
                    else:
                        chunk.copy_(shadows[storage_id][offset:offset + length])
                    if cursor is not None:
                        cursor.overlay(storage_id, offset, chunk)
                    yield storage_id, offset, chunk
            finally:
                if source is not None:
                    source.close()
        if cursor is not None and cursor.record is not None:
            raise ValueError("undo journal has unconsumed records")

    iterator = generate()
    try:
        yield iterator
    finally:
        iterator.close()
        if cursor is not None:
            cursor.close()


def _compare(views, chunks, chunk_bytes: int, page_bytes: int) -> dict:
    stage = torch.empty(min(chunk_bytes, max((v.numel() for v in views), default=0)),
                        dtype=torch.uint8, device="cpu")
    result = {"equal": True, "compared_bytes": 0, "mismatched_bytes": 0,
              "mismatched_pages": 0, "mismatched_storages": 0, "first_mismatches": []}
    last_bad_storage = None
    with torch.no_grad():
        for storage_id, offset, expected in chunks:
            actual = stage[:expected.numel()]
            actual.copy_(views[storage_id][offset:offset + expected.numel()], non_blocking=False)
            diff = actual != expected
            count = int(diff.sum().item())
            result["compared_bytes"] += expected.numel()
            if count:
                result["equal"] = False
                result["mismatched_bytes"] += count
                full = diff.numel() // page_bytes * page_bytes
                pages = int(diff[:full].view(-1, page_bytes).any(dim=1).sum().item())
                pages += int(bool(diff[full:].any().item()))
                result["mismatched_pages"] += pages
                if last_bad_storage != storage_id:
                    last_bad_storage = storage_id
                    result["mismatched_storages"] += 1
                if len(result["first_mismatches"]) < 16:
                    first = int(diff.to(dtype=torch.uint8).argmax().item())
                    result["first_mismatches"].append({"storage_id": storage_id,
                                                       "byte_offset": offset + first})
    return result


def restore_snapshot(snapshot_dir, live_tensors: Mapping[str, torch.Tensor], *,
                     boundary: str = "current", chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> dict:
    """Restore raw bytes into fresh state after validating its full binding topology.

    Reads a bounded chunk of the base, overlays old journal bytes for a previous
    boundary, and transfers the reconstructed chunk once.  Unregistered storage
    padding and aliases are restored along with visible tensor elements.
    """
    manifest = load_snapshot_manifest(snapshot_dir)
    _check_boundary(boundary, manifest["previous"] is not None)
    chunk_bytes = _chunk_size(chunk_bytes, manifest["page_bytes"])
    topology, views = _describe(live_tensors)
    _validate_topology(manifest["topology"], topology)
    _synchronize(views)
    directory = Path(snapshot_dir)
    undo = directory / manifest["previous"]["undo_file"] if boundary == "previous" else None
    with torch.no_grad(), _expected_chunks(topology, chunk_bytes, manifest["page_bytes"],
            directory=directory, files=manifest["base_files"], undo=undo) as chunks:
        for storage_id, offset, chunk in chunks:
            views[storage_id][offset:offset + chunk.numel()].copy_(chunk, non_blocking=False)
    _synchronize(views)
    return {"boundary_id": manifest[boundary]["boundary_id"],
            "restored_bytes": manifest["total_bytes"], "boundary": boundary}


def compare_snapshot(snapshot_dir, live_tensors: Mapping[str, torch.Tensor], *,
                     boundary: str = "current", chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> dict:
    """Report exact differing bytes/pages against a persisted boundary, read-only."""
    manifest = load_snapshot_manifest(snapshot_dir)
    _check_boundary(boundary, manifest["previous"] is not None)
    chunk_bytes = _chunk_size(chunk_bytes, manifest["page_bytes"])
    topology, views = _describe(live_tensors)
    _validate_topology(manifest["topology"], topology)
    _synchronize(views)
    directory = Path(snapshot_dir)
    undo = directory / manifest["previous"]["undo_file"] if boundary == "previous" else None
    with _expected_chunks(topology, chunk_bytes, manifest["page_bytes"],
            directory=directory, files=manifest["base_files"], undo=undo) as chunks:
        return _compare(views, chunks, chunk_bytes, manifest["page_bytes"])
