"""Authenticated fixed-width multimodal prefix geometry contracts.

The contract is intentionally independent of a benchmark and of the training
configuration.  A consumer must authenticate the artifact with an externally
recorded SHA-256 before using its fixed physical prefix width.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import stat
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

PREFIX_GEOMETRY_SCHEMA = "duo-vla-prefix-geometry-v1"
INSTRUCTION_CANONICALIZATION = "exact-utf8-codepoint-sort-unique-v1"
CONVERSATION_LAYOUT = "single-user-ordered-images-then-instruction-v1"
IMAGE_PROBE = "all-zero-uint8-hwc-v1"
VALID_LENGTH_METRIC = "sum-binary-attention-mask-v1"
REQUIRED_PROCESSOR_OUTPUT_FIELDS = (
    "attention_mask",
    "image_position_ids",
    "input_ids",
    "mm_token_type_ids",
    "pixel_values",
)
IMAGE_BATCH_AXIS = "batch-times-ordered-camera-count-v1"

_SHA256_LENGTH = 64
_REVISION_LENGTH = 40
_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
_TOP_LEVEL_KEYS = {
    "schema",
    "content_sha256",
    "model",
    "processor",
    "ordered_cameras",
    "tokenization",
    "instruction_inventory",
    "geometry",
}
_TOKENIZATION_KEYS = {
    "conversation_layout",
    "image_probe",
    "tokenize",
    "add_generation_prompt",
    "return_dict",
    "return_tensors",
    "unbounded_processor_kwargs",
    "fixed_processor_kwargs",
    "padding_side",
    "valid_length_metric",
    "required_output_fields",
    "image_batch_axis",
}
_INVENTORY_KEYS = {"canonicalization", "count", "sha256", "records"}
_RECORD_KEYS = {"instruction", "instruction_sha256", "valid_prefix_length"}
_GEOMETRY_KEYS = {
    "fixed_physical_prefix_width",
    "maximum_valid_prefix_length",
    "valid_prefix_length_histogram",
}
_HISTOGRAM_KEYS = {"valid_prefix_length", "count"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_revision(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _REVISION_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    observed = set(value)
    _require(
        observed == expected,
        f"{name} fields differ: missing={sorted(expected - observed)}, extra={sorted(observed - expected)}",
    )


def _validate_text(value: object, name: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{name} must be nonempty exact text")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} is not valid UTF-8 text") from exc
    return value


@dataclass(frozen=True, slots=True)
class SnapshotTreeIdentity:
    """Pinned repository revision plus authenticated local snapshot inventory."""

    repository_id: str
    revision: str
    tree_metadata_sha256: str
    content_inventory_sha256: str
    files_verified: int
    total_bytes: int

    def __post_init__(self) -> None:
        _validate_text(self.repository_id, "snapshot repository_id")
        _require(
            self.repository_id == self.repository_id.strip(),
            "snapshot repository_id must not have surrounding whitespace",
        )
        _require(_is_revision(self.revision), "snapshot revision must be a lowercase 40-character Git SHA")
        _require(_is_sha256(self.tree_metadata_sha256), "snapshot tree_metadata_sha256 is invalid")
        _require(_is_sha256(self.content_inventory_sha256), "snapshot content_inventory_sha256 is invalid")
        _require(
            type(self.files_verified) is int and self.files_verified > 0,
            "snapshot files_verified must be a positive integer",
        )
        _require(type(self.total_bytes) is int and self.total_bytes > 0, "snapshot total_bytes must be positive")

    def to_dict(self) -> dict[str, str | int]:
        return {
            "repository_id": self.repository_id,
            "revision": self.revision,
            "tree_metadata_sha256": self.tree_metadata_sha256,
            "content_inventory_sha256": self.content_inventory_sha256,
            "files_verified": self.files_verified,
            "total_bytes": self.total_bytes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SnapshotTreeIdentity:
        _require(isinstance(value, Mapping), "snapshot identity must be an object")
        _exact_keys(
            value,
            {
                "repository_id",
                "revision",
                "tree_metadata_sha256",
                "content_inventory_sha256",
                "files_verified",
                "total_bytes",
            },
            "snapshot identity",
        )
        return cls(
            repository_id=value["repository_id"],
            revision=value["revision"],
            tree_metadata_sha256=value["tree_metadata_sha256"],
            content_inventory_sha256=value["content_inventory_sha256"],
            files_verified=value["files_verified"],
            total_bytes=value["total_bytes"],
        )

    @classmethod
    def from_huggingface_report(
        cls,
        repository_id: str,
        report: Mapping[str, Any],
    ) -> SnapshotTreeIdentity:
        """Construct an identity from :func:`verify_huggingface_snapshot` output."""

        _require(isinstance(report, Mapping), "Hugging Face snapshot report must be an object")
        return cls(
            repository_id=repository_id,
            revision=report.get("revision"),
            tree_metadata_sha256=report.get("tree_metadata_sha256"),
            content_inventory_sha256=report.get("content_inventory_sha256"),
            files_verified=report.get("files_verified"),
            total_bytes=report.get("total_bytes"),
        )


@dataclass(frozen=True, slots=True)
class CameraGeometry:
    """One ordered raw RGB camera input used by the processor probe."""

    name: str
    height: int
    width: int
    channels: int = 3

    def __post_init__(self) -> None:
        _validate_text(self.name, "camera name")
        _require(self.name == self.name.strip(), "camera name must not have surrounding whitespace")
        _require(type(self.height) is int and self.height > 0, "camera height must be a positive integer")
        _require(type(self.width) is int and self.width > 0, "camera width must be a positive integer")
        _require(
            type(self.channels) is int and self.channels == 3,
            "Duo-VLA camera probes must have exactly three RGB channels",
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.height, self.width, self.channels

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "shape": list(self.shape), "dtype": "uint8", "layout": "HWC"}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CameraGeometry:
        _require(isinstance(value, Mapping), "camera geometry must be an object")
        _exact_keys(value, {"name", "shape", "dtype", "layout"}, "camera geometry")
        shape = value["shape"]
        _require(
            isinstance(shape, list) and len(shape) == 3,
            "camera geometry shape must be a three-element JSON array",
        )
        _require(value["dtype"] == "uint8" and value["layout"] == "HWC", "camera dtype/layout contract mismatch")
        return cls(name=value["name"], height=shape[0], width=shape[1], channels=shape[2])


def _canonical_json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    options: dict[str, Any] = {"allow_nan": False, "ensure_ascii": False, "sort_keys": True}
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    return (json.dumps(value, **options) + ("\n" if pretty else "")).encode("utf-8")


def _same_json_value(observed: Any, expected: Any) -> bool:
    """Compare JSON values without Python's ``True == 1`` coercion."""

    return _canonical_json_bytes(observed) == _canonical_json_bytes(expected)


def _normalize_instructions(instructions: Sequence[str]) -> tuple[str, ...]:
    _require(not isinstance(instructions, (str, bytes)), "instructions must be a sequence of exact strings")
    normalized = tuple(_validate_text(value, "instruction") for value in instructions)
    _require(bool(normalized), "instruction inventory must not be empty")
    _require(len(set(normalized)) == len(normalized), "instruction inventory contains duplicate exact strings")
    return tuple(sorted(normalized))


def instruction_inventory_sha256(instructions: Sequence[str]) -> str:
    """Hash the canonical sorted unique exact-UTF-8 instruction inventory."""

    normalized = _normalize_instructions(instructions)
    envelope = {"canonicalization": INSTRUCTION_CANONICALIZATION, "instructions": list(normalized)}
    return hashlib.sha256(_canonical_json_bytes(envelope)).hexdigest()


def _normalize_identity(value: SnapshotTreeIdentity | Mapping[str, Any]) -> SnapshotTreeIdentity:
    return value if isinstance(value, SnapshotTreeIdentity) else SnapshotTreeIdentity.from_dict(value)


def _normalize_cameras(values: Sequence[CameraGeometry | Mapping[str, Any]]) -> tuple[CameraGeometry, ...]:
    _require(not isinstance(values, (str, bytes)), "ordered cameras must be a sequence")
    cameras = tuple(value if isinstance(value, CameraGeometry) else CameraGeometry.from_dict(value) for value in values)
    _require(bool(cameras), "at least one camera geometry is required")
    names = [camera.name for camera in cameras]
    _require(len(names) == len(set(names)), "ordered camera names must be unique")
    return cameras


def _normalize_instruction_lengths(
    instruction_lengths: Mapping[str, int] | Sequence[tuple[str, int]],
) -> tuple[tuple[str, int], ...]:
    items = list(instruction_lengths.items()) if isinstance(instruction_lengths, Mapping) else list(instruction_lengths)
    _require(bool(items), "instruction length inventory must not be empty")
    instructions: list[str] = []
    normalized: list[tuple[str, int]] = []
    for item in items:
        _require(
            isinstance(item, tuple) and len(item) == 2, "instruction lengths must contain (instruction, length) pairs"
        )
        instruction = _validate_text(item[0], "instruction")
        length = item[1]
        _require(type(length) is int and length > 0, "valid prefix lengths must be positive integers")
        instructions.append(instruction)
        normalized.append((instruction, length))
    _require(len(instructions) == len(set(instructions)), "instruction length inventory contains duplicates")
    return tuple(sorted(normalized))


def build_prefix_geometry_contract(
    *,
    model_identity: SnapshotTreeIdentity | Mapping[str, Any],
    processor_identity: SnapshotTreeIdentity | Mapping[str, Any],
    ordered_cameras: Sequence[CameraGeometry | Mapping[str, Any]],
    instruction_lengths: Mapping[str, int] | Sequence[tuple[str, int]],
    fixed_physical_prefix_width: int,
    padding_side: str,
) -> dict[str, Any]:
    """Build a self-consistent contract from non-truncated measured lengths."""

    model = _normalize_identity(model_identity)
    processor = _normalize_identity(processor_identity)
    cameras = _normalize_cameras(ordered_cameras)
    measured = _normalize_instruction_lengths(instruction_lengths)
    _require(
        type(fixed_physical_prefix_width) is int and fixed_physical_prefix_width > 0,
        "fixed physical prefix width must be a positive integer",
    )
    _require(padding_side in {"left", "right"}, "padding_side must be exactly 'left' or 'right'")
    maximum = max(length for _, length in measured)
    _require(
        fixed_physical_prefix_width > maximum,
        "fixed physical prefix width must leave at least one padding sentinel beyond the measured maximum",
    )

    instructions = [instruction for instruction, _ in measured]
    records = [
        {
            "instruction": instruction,
            "instruction_sha256": hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
            "valid_prefix_length": length,
        }
        for instruction, length in measured
    ]
    counts = Counter(length for _, length in measured)
    histogram = [{"valid_prefix_length": length, "count": counts[length]} for length in sorted(counts)]
    payload: dict[str, Any] = {
        "schema": PREFIX_GEOMETRY_SCHEMA,
        "model": model.to_dict(),
        "processor": processor.to_dict(),
        "ordered_cameras": [camera.to_dict() for camera in cameras],
        "tokenization": {
            "conversation_layout": CONVERSATION_LAYOUT,
            "image_probe": IMAGE_PROBE,
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
            "unbounded_processor_kwargs": {"padding": False, "truncation": False},
            "fixed_processor_kwargs": {
                "padding": "max_length",
                "max_length": fixed_physical_prefix_width,
                "truncation": False,
            },
            "padding_side": padding_side,
            "valid_length_metric": VALID_LENGTH_METRIC,
            "required_output_fields": list(REQUIRED_PROCESSOR_OUTPUT_FIELDS),
            "image_batch_axis": IMAGE_BATCH_AXIS,
        },
        "instruction_inventory": {
            "canonicalization": INSTRUCTION_CANONICALIZATION,
            "count": len(records),
            "sha256": instruction_inventory_sha256(instructions),
            "records": records,
        },
        "geometry": {
            "fixed_physical_prefix_width": fixed_physical_prefix_width,
            "maximum_valid_prefix_length": maximum,
            "valid_prefix_length_histogram": histogram,
        },
    }
    payload["content_sha256"] = prefix_geometry_content_sha256(payload)
    return validate_prefix_geometry_contract(payload)


def prefix_geometry_content_sha256(payload: Mapping[str, Any]) -> str:
    """Return the semantic hash of a contract, excluding its self-hash field."""

    _require(isinstance(payload, Mapping), "prefix geometry contract must be an object")
    unhashed = dict(payload)
    unhashed.pop("content_sha256", None)
    return hashlib.sha256(_canonical_json_bytes(unhashed)).hexdigest()


def _validate_tokenization(value: Any, fixed_width: int) -> str:
    _require(isinstance(value, Mapping), "tokenization contract must be an object")
    _exact_keys(value, _TOKENIZATION_KEYS, "tokenization contract")
    constants = {
        "conversation_layout": CONVERSATION_LAYOUT,
        "image_probe": IMAGE_PROBE,
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
        "unbounded_processor_kwargs": {"padding": False, "truncation": False},
        "fixed_processor_kwargs": {"padding": "max_length", "max_length": fixed_width, "truncation": False},
        "valid_length_metric": VALID_LENGTH_METRIC,
        "required_output_fields": list(REQUIRED_PROCESSOR_OUTPUT_FIELDS),
        "image_batch_axis": IMAGE_BATCH_AXIS,
    }
    mismatches = [key for key, expected in constants.items() if not _same_json_value(value[key], expected)]
    _require(not mismatches, f"tokenization contract constants differ: {mismatches}")
    padding_side = value["padding_side"]
    _require(padding_side in {"left", "right"}, "tokenization padding_side must be 'left' or 'right'")
    return padding_side


def validate_prefix_geometry_contract(
    payload: Mapping[str, Any],
    *,
    expected_content_sha256: str | None = None,
    expected_model_identity: SnapshotTreeIdentity | Mapping[str, Any] | None = None,
    expected_processor_identity: SnapshotTreeIdentity | Mapping[str, Any] | None = None,
    expected_ordered_cameras: Sequence[CameraGeometry | Mapping[str, Any]] | None = None,
    expected_instructions: Sequence[str] | None = None,
    expected_fixed_physical_prefix_width: int | None = None,
) -> dict[str, Any]:
    """Strictly validate a payload and any externally pinned identities."""

    _require(isinstance(payload, Mapping), "prefix geometry contract must be an object")
    _exact_keys(payload, _TOP_LEVEL_KEYS, "prefix geometry contract")
    _require(payload["schema"] == PREFIX_GEOMETRY_SCHEMA, "unsupported prefix geometry schema")
    model = SnapshotTreeIdentity.from_dict(payload["model"])
    processor = SnapshotTreeIdentity.from_dict(payload["processor"])

    camera_values = payload["ordered_cameras"]
    _require(isinstance(camera_values, list), "ordered_cameras must be a JSON array")
    cameras = _normalize_cameras(camera_values)

    geometry = payload["geometry"]
    _require(isinstance(geometry, Mapping), "prefix geometry must be an object")
    _exact_keys(geometry, _GEOMETRY_KEYS, "prefix geometry")
    fixed_width = geometry["fixed_physical_prefix_width"]
    maximum = geometry["maximum_valid_prefix_length"]
    _require(type(fixed_width) is int and fixed_width > 0, "fixed physical prefix width must be positive")
    _require(type(maximum) is int and maximum > 0, "maximum valid prefix length must be positive")
    _require(
        fixed_width > maximum,
        "fixed physical prefix width must leave at least one padding sentinel beyond the recorded maximum",
    )
    _validate_tokenization(payload["tokenization"], fixed_width)

    inventory = payload["instruction_inventory"]
    _require(isinstance(inventory, Mapping), "instruction_inventory must be an object")
    _exact_keys(inventory, _INVENTORY_KEYS, "instruction inventory")
    _require(
        inventory["canonicalization"] == INSTRUCTION_CANONICALIZATION,
        "instruction canonicalization contract mismatch",
    )
    _require(type(inventory["count"]) is int and inventory["count"] > 0, "instruction inventory count is invalid")
    records = inventory["records"]
    _require(isinstance(records, list) and bool(records), "instruction records must be a nonempty JSON array")
    instructions: list[str] = []
    lengths: list[int] = []
    for index, record in enumerate(records):
        _require(isinstance(record, Mapping), f"instruction record {index} must be an object")
        _exact_keys(record, _RECORD_KEYS, f"instruction record {index}")
        instruction = _validate_text(record["instruction"], f"instruction record {index} text")
        expected_instruction_sha = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
        _require(
            record["instruction_sha256"] == expected_instruction_sha,
            f"instruction record {index} SHA-256 mismatch",
        )
        length = record["valid_prefix_length"]
        _require(type(length) is int and length > 0, f"instruction record {index} has an invalid prefix length")
        instructions.append(instruction)
        lengths.append(length)
    _require(instructions == sorted(instructions), "instruction records are not in canonical codepoint order")
    _require(len(instructions) == len(set(instructions)), "instruction records contain duplicates")
    _require(inventory["count"] == len(instructions), "instruction inventory count mismatch")
    _require(
        inventory["sha256"] == instruction_inventory_sha256(instructions),
        "instruction inventory SHA-256 mismatch",
    )
    _require(max(lengths) == maximum, "maximum valid prefix length does not match the records")

    histogram = geometry["valid_prefix_length_histogram"]
    _require(isinstance(histogram, list) and bool(histogram), "prefix length histogram must be nonempty")
    observed_histogram: list[dict[str, int]] = []
    for index, entry in enumerate(histogram):
        _require(isinstance(entry, Mapping), f"prefix length histogram entry {index} must be an object")
        _exact_keys(entry, _HISTOGRAM_KEYS, f"prefix length histogram entry {index}")
        length = entry["valid_prefix_length"]
        count = entry["count"]
        _require(
            type(length) is int and length > 0 and type(count) is int and count > 0,
            f"prefix length histogram entry {index} is invalid",
        )
        observed_histogram.append({"valid_prefix_length": length, "count": count})
    counts = Counter(lengths)
    expected_histogram = [{"valid_prefix_length": length, "count": counts[length]} for length in sorted(counts)]
    _require(observed_histogram == expected_histogram, "valid prefix length histogram does not match the records")

    recorded_sha = payload["content_sha256"]
    _require(_is_sha256(recorded_sha), "prefix geometry content_sha256 is invalid")
    observed_sha = prefix_geometry_content_sha256(payload)
    _require(hmac.compare_digest(recorded_sha, observed_sha), "prefix geometry semantic SHA-256 mismatch")
    if expected_content_sha256 is not None:
        _require(_is_sha256(expected_content_sha256), "expected prefix geometry SHA-256 is invalid")
        _require(
            hmac.compare_digest(observed_sha, expected_content_sha256),
            "prefix geometry artifact differs from the externally pinned SHA-256",
        )
    if expected_model_identity is not None:
        _require(model == _normalize_identity(expected_model_identity), "prefix geometry model identity mismatch")
    if expected_processor_identity is not None:
        _require(
            processor == _normalize_identity(expected_processor_identity),
            "prefix geometry processor identity mismatch",
        )
    if expected_ordered_cameras is not None:
        _require(cameras == _normalize_cameras(expected_ordered_cameras), "prefix geometry ordered cameras mismatch")
    if expected_instructions is not None:
        _require(
            tuple(instructions) == _normalize_instructions(expected_instructions),
            "prefix geometry instruction inventory mismatch",
        )
    if expected_fixed_physical_prefix_width is not None:
        _require(
            type(expected_fixed_physical_prefix_width) is int and fixed_width == expected_fixed_physical_prefix_width,
            "prefix geometry fixed physical width mismatch",
        )
    return json.loads(_canonical_json_bytes(dict(payload)))


def prefix_geometry_artifact_bytes(payload: Mapping[str, Any]) -> bytes:
    """Serialize a valid contract in its one accepted canonical JSON form."""

    normalized = validate_prefix_geometry_contract(payload)
    return _canonical_json_bytes(normalized, pretty=True)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def _absolute_lexical_path(path: str | Path) -> Path:
    raw = os.fspath(path)
    _require(bool(raw), "prefix geometry artifact path must not be empty")
    absolute = Path(os.path.abspath(raw))
    _require(absolute.name not in {"", ".", ".."}, "prefix geometry artifact path must name a file")
    return absolute


def _secure_parent_fd(path: Path) -> int:
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
    descriptor = os.open("/", directory_flags)
    try:
        for component in path.parent.parts[1:]:
            next_descriptor = os.open(component, directory_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _secure_read(path: str | Path) -> bytes:
    absolute = _absolute_lexical_path(path)
    parent_fd: int | None = None
    descriptor: int | None = None
    try:
        parent_fd = _secure_parent_fd(absolute)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
        descriptor = os.open(absolute.name, flags, dir_fd=parent_fd)
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode), "prefix geometry artifact must be a regular file")
        _require(before.st_size <= _MAX_ARTIFACT_BYTES, "prefix geometry artifact exceeds the size limit")
        chunks: list[bytes] = []
        remaining = _MAX_ARTIFACT_BYTES + 1
        while remaining > 0:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        data = b"".join(chunks)
        _require(len(data) <= _MAX_ARTIFACT_BYTES, "prefix geometry artifact exceeds the size limit")
        after = os.fstat(descriptor)
        identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        _require(
            identity_before == identity_after and len(data) == after.st_size,
            "prefix geometry artifact changed while read",
        )
        return data
    except OSError as exc:
        raise ValueError(f"cannot securely open prefix geometry artifact without symlinks: {absolute}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if parent_fd is not None:
            os.close(parent_fd)


def _verify_published_prefix_geometry(
    path: Path,
    *,
    publishing_parent_fd: int,
    expected_file_identity: tuple[int, int],
    expected_data: bytes,
) -> None:
    """Reopen the lexical destination and bind it to the published inode."""

    reopened_parent_fd: int | None = None
    descriptor: int | None = None
    try:
        reopened_parent_fd = _secure_parent_fd(path)
        publishing_parent = os.fstat(publishing_parent_fd)
        reopened_parent = os.fstat(reopened_parent_fd)
        _require(
            (publishing_parent.st_dev, publishing_parent.st_ino) == (reopened_parent.st_dev, reopened_parent.st_ino),
            "prefix geometry artifact parent directory changed during publication",
        )
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
        descriptor = os.open(path.name, flags, dir_fd=reopened_parent_fd)
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode), "published prefix geometry artifact must be a regular file")
        _require(
            (before.st_dev, before.st_ino) == expected_file_identity,
            "published prefix geometry artifact is not the linked source inode",
        )
        _require(before.st_nlink == 1, "published prefix geometry artifact must have exactly one hard link")
        _require(before.st_size == len(expected_data), "published prefix geometry artifact byte length differs")

        content = bytearray()
        remaining = len(expected_data) + 1
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                break
            content.extend(block)
            remaining -= len(block)
        observed_data = bytes(content)
        _require(len(observed_data) == len(expected_data), "published prefix geometry artifact byte length differs")
        _require(
            hmac.compare_digest(hashlib.sha256(observed_data).digest(), hashlib.sha256(expected_data).digest()),
            "published prefix geometry artifact raw SHA-256 differs",
        )
        _require(observed_data == expected_data, "published prefix geometry artifact is not canonical JSON bytes")

        after = os.fstat(descriptor)
        _require(
            (
                before.st_dev,
                before.st_ino,
                before.st_mode,
                before.st_nlink,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            == (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_nlink,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ),
            "published prefix geometry artifact changed during final verification",
        )
    except OSError as exc:
        raise ValueError(f"cannot verify published prefix geometry artifact without symlinks: {path}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if reopened_parent_fd is not None:
            os.close(reopened_parent_fd)


def load_prefix_geometry_contract(
    path: str | Path,
    *,
    expected_content_sha256: str,
    expected_model_identity: SnapshotTreeIdentity | Mapping[str, Any] | None = None,
    expected_processor_identity: SnapshotTreeIdentity | Mapping[str, Any] | None = None,
    expected_ordered_cameras: Sequence[CameraGeometry | Mapping[str, Any]] | None = None,
    expected_instructions: Sequence[str] | None = None,
    expected_fixed_physical_prefix_width: int | None = None,
) -> dict[str, Any]:
    """Securely load canonical JSON and reject a valid-but-substituted artifact."""

    _require(_is_sha256(expected_content_sha256), "expected prefix geometry SHA-256 is invalid")
    raw = _secure_read(path)
    try:
        payload = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("prefix geometry artifact is not strict JSON") from exc
    normalized = validate_prefix_geometry_contract(
        payload,
        expected_content_sha256=expected_content_sha256,
        expected_model_identity=expected_model_identity,
        expected_processor_identity=expected_processor_identity,
        expected_ordered_cameras=expected_ordered_cameras,
        expected_instructions=expected_instructions,
        expected_fixed_physical_prefix_width=expected_fixed_physical_prefix_width,
    )
    _require(raw == _canonical_json_bytes(normalized, pretty=True), "prefix geometry artifact is not canonical JSON")
    return normalized


def save_prefix_geometry_contract(path: str | Path, payload: Mapping[str, Any]) -> str:
    """Publish one immutable canonical artifact without following directory symlinks."""

    data = prefix_geometry_artifact_bytes(payload)
    absolute = _absolute_lexical_path(path)
    parent_fd: int | None = None
    descriptor: int | None = None
    temporary_name = f".{absolute.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    try:
        parent_fd = _secure_parent_fd(absolute)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
        descriptor = os.open(temporary_name, flags, 0o644, dir_fd=parent_fd)
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            _require(written > 0, "could not write prefix geometry artifact")
            view = view[written:]
        os.fsync(descriptor)
        source_stat = os.fstat(descriptor)
        _require(stat.S_ISREG(source_stat.st_mode), "temporary prefix geometry artifact must be a regular file")
        _require(source_stat.st_nlink == 1, "temporary prefix geometry artifact must have exactly one hard link")
        _require(source_stat.st_size == len(data), "temporary prefix geometry artifact byte length differs")
        source_identity = (source_stat.st_dev, source_stat.st_ino)
        os.close(descriptor)
        descriptor = None
        os.link(
            temporary_name,
            absolute.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        os.unlink(temporary_name, dir_fd=parent_fd)
        os.fsync(parent_fd)
        _verify_published_prefix_geometry(
            absolute,
            publishing_parent_fd=parent_fd,
            expected_file_identity=source_identity,
            expected_data=data,
        )
    except FileExistsError as exc:
        raise FileExistsError(f"prefix geometry artifact already exists: {absolute}") from exc
    except OSError as exc:
        raise ValueError(f"cannot securely publish prefix geometry artifact without symlinks: {absolute}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if parent_fd is not None:
            with suppress(FileNotFoundError):
                os.unlink(temporary_name, dir_fd=parent_fd)
            os.close(parent_fd)
    return str(payload["content_sha256"])


def _processor_padding_side(processor: Any) -> str:
    tokenizer = getattr(processor, "tokenizer", None)
    padding_side = getattr(tokenizer, "padding_side", None)
    _require(padding_side in {"left", "right"}, "processor tokenizer has no supported padding_side")
    return padding_side


def _validate_processor_output(
    output: Any,
    *,
    expected_width: int | None,
    expected_batch_size: int,
    images_per_prefix: int,
    padding_side: str,
    require_unpadded: bool,
) -> tuple[int, ...]:
    _require(isinstance(output, Mapping), "processor output must be a mapping")
    missing = set(REQUIRED_PROCESSOR_OUTPUT_FIELDS) - set(output)
    _require(not missing, f"processor output is missing required fields: {sorted(missing)}")
    input_ids = output.get("input_ids")
    attention_mask = output.get("attention_mask")
    mm_token_type_ids = output.get("mm_token_type_ids")
    pixel_values = output.get("pixel_values")
    image_position_ids = output.get("image_position_ids")
    _require(isinstance(input_ids, Tensor) and input_ids.ndim == 2, "processor input_ids must have rank two")
    _require(
        isinstance(attention_mask, Tensor) and attention_mask.ndim == 2,
        "processor attention_mask must have rank two",
    )
    _require(input_ids.shape == attention_mask.shape, "processor input_ids and attention_mask shapes differ")
    _require(
        input_ids.dtype != torch.bool and not input_ids.is_floating_point() and not input_ids.is_complex(),
        "processor input_ids must be integral",
    )
    _require(
        isinstance(mm_token_type_ids, Tensor)
        and mm_token_type_ids.shape == input_ids.shape
        and mm_token_type_ids.dtype != torch.bool
        and not mm_token_type_ids.is_floating_point()
        and not mm_token_type_ids.is_complex(),
        "processor mm_token_type_ids must be integral and match input_ids shape",
    )
    _require(
        not attention_mask.is_floating_point() and not attention_mask.is_complex(),
        "processor attention_mask must be integral or boolean",
    )
    _require(
        type(expected_batch_size) is int and expected_batch_size > 0,
        "expected processor batch size must be a positive integer",
    )
    _require(
        type(images_per_prefix) is int and images_per_prefix > 0,
        "images_per_prefix must be a positive integer",
    )
    _require(input_ids.shape[0] == expected_batch_size, "processor output batch size differs from the expected batch")
    expected_image_batch = expected_batch_size * images_per_prefix
    _require(
        isinstance(pixel_values, Tensor) and pixel_values.ndim >= 1 and pixel_values.shape[0] == expected_image_batch,
        "processor pixel_values first axis must equal batch size times ordered camera count",
    )
    _require(
        isinstance(image_position_ids, Tensor)
        and image_position_ids.ndim >= 1
        and image_position_ids.shape[0] == expected_image_batch,
        "processor image_position_ids first axis must equal batch size times ordered camera count",
    )
    _require(
        bool(((attention_mask == 0) | (attention_mask == 1)).all()),
        "processor attention_mask must be binary",
    )
    if expected_width is not None:
        _require(
            input_ids.shape[1] == expected_width,
            f"processor physical prefix width is {input_ids.shape[1]}, expected exactly {expected_width}; "
            "refusing truncation",
        )
    valid = attention_mask.bool()
    lengths = tuple(int(value) for value in valid.sum(dim=1).tolist())
    _require(all(length > 0 for length in lengths), "every processor prefix must contain a valid token")
    for row, length in zip(valid, lengths, strict=True):
        if require_unpadded:
            _require(bool(row.all()), "unbounded single-example tokenization unexpectedly contains padding")
            continue
        expected = torch.zeros_like(row)
        if padding_side == "left":
            expected[-length:] = True
        else:
            expected[:length] = True
        _require(torch.equal(row, expected), f"processor attention_mask is not contiguous {padding_side} padding")
    return lengths


def apply_fixed_prefix_chat_template(
    processor: Any,
    conversations: Any,
    *,
    fixed_physical_prefix_width: int,
    padding_side: str,
    expected_batch_size: int,
    images_per_prefix: int,
) -> Any:
    """Apply the production template and fail if its physical width is not exact.

    ``truncation=False`` is deliberate.  Transformers may return an overlength
    tensor instead of clipping to ``max_length``; the exact-width check turns
    that behavior into a hard failure.
    """

    _require(
        type(fixed_physical_prefix_width) is int and fixed_physical_prefix_width > 0,
        "fixed physical prefix width must be a positive integer",
    )
    _require(padding_side in {"left", "right"}, "padding_side must be exactly 'left' or 'right'")
    _require(
        _processor_padding_side(processor) == padding_side, "processor tokenizer padding_side differs from contract"
    )
    output = processor.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={
            "padding": "max_length",
            "max_length": fixed_physical_prefix_width,
            "truncation": False,
        },
    )
    _validate_processor_output(
        output,
        expected_width=fixed_physical_prefix_width,
        expected_batch_size=expected_batch_size,
        images_per_prefix=images_per_prefix,
        padding_side=padding_side,
        require_unpadded=False,
    )
    return output


def _probe_conversation(instruction: str, cameras: Sequence[CameraGeometry]) -> list[dict[str, Any]]:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - the train extra always installs Pillow
        raise RuntimeError("Pillow is required to measure multimodal prefix geometry") from exc
    content: list[dict[str, Any]] = [
        {"type": "image", "image": Image.fromarray(np.zeros(camera.shape, dtype=np.uint8))} for camera in cameras
    ]
    content.append({"type": "text", "text": instruction})
    return [{"role": "user", "content": content}]


def measure_prefix_valid_lengths(
    processor: Any,
    *,
    instructions: Sequence[str],
    ordered_cameras: Sequence[CameraGeometry | Mapping[str, Any]],
    fixed_physical_prefix_width: int,
    padding_side: str,
) -> dict[str, int]:
    """Measure every exact instruction with unbounded and fixed-width probes."""

    measured = measure_unbounded_prefix_valid_lengths(
        processor,
        instructions=instructions,
        ordered_cameras=ordered_cameras,
        padding_side=padding_side,
    )
    cameras = _normalize_cameras(ordered_cameras)
    for instruction, unbounded_length in measured.items():
        conversation = _probe_conversation(instruction, cameras)
        fixed = apply_fixed_prefix_chat_template(
            processor,
            conversation,
            fixed_physical_prefix_width=fixed_physical_prefix_width,
            padding_side=padding_side,
            expected_batch_size=1,
            images_per_prefix=len(cameras),
        )
        fixed_lengths = _validate_processor_output(
            fixed,
            expected_width=fixed_physical_prefix_width,
            expected_batch_size=1,
            images_per_prefix=len(cameras),
            padding_side=padding_side,
            require_unpadded=False,
        )
        _require(len(fixed_lengths) == 1, "instruction geometry probes must produce exactly one fixed prefix")
        _require(
            fixed_lengths[0] == unbounded_length,
            f"fixed-width processor truncated or changed instruction {instruction!r}",
        )
    return measured


def measure_unbounded_prefix_valid_lengths(
    processor: Any,
    *,
    instructions: Sequence[str],
    ordered_cameras: Sequence[CameraGeometry | Mapping[str, Any]],
    padding_side: str,
) -> dict[str, int]:
    """Measure exact prefix lengths without selecting a physical width.

    This is the discovery phase for a benchmark whose complete authenticated
    instruction inventory is not available until its dataset has been
    installed.  Consumers must still run :func:`measure_prefix_valid_lengths`
    at the selected width before publishing a contract.
    """

    canonical_instructions = _normalize_instructions(instructions)
    cameras = _normalize_cameras(ordered_cameras)
    _require(
        _processor_padding_side(processor) == padding_side, "processor tokenizer padding_side differs from contract"
    )
    measured: dict[str, int] = {}
    for instruction in canonical_instructions:
        conversation = _probe_conversation(instruction, cameras)
        unbounded = processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": False, "truncation": False},
        )
        unbounded_lengths = _validate_processor_output(
            unbounded,
            expected_width=None,
            expected_batch_size=1,
            images_per_prefix=len(cameras),
            padding_side=padding_side,
            require_unpadded=True,
        )
        _require(len(unbounded_lengths) == 1, "instruction geometry probes must produce exactly one prefix")
        measured[instruction] = unbounded_lengths[0]
    return measured


def create_prefix_geometry_contract(
    processor: Any,
    *,
    model_identity: SnapshotTreeIdentity | Mapping[str, Any],
    processor_identity: SnapshotTreeIdentity | Mapping[str, Any],
    ordered_cameras: Sequence[CameraGeometry | Mapping[str, Any]],
    instructions: Sequence[str],
    fixed_physical_prefix_width: int,
    padding_side: str,
) -> dict[str, Any]:
    """Measure non-truncated prefixes and construct their authenticated contract."""

    lengths = measure_prefix_valid_lengths(
        processor,
        instructions=instructions,
        ordered_cameras=ordered_cameras,
        fixed_physical_prefix_width=fixed_physical_prefix_width,
        padding_side=padding_side,
    )
    return build_prefix_geometry_contract(
        model_identity=model_identity,
        processor_identity=processor_identity,
        ordered_cameras=ordered_cameras,
        instruction_lengths=lengths,
        fixed_physical_prefix_width=fixed_physical_prefix_width,
        padding_side=padding_side,
    )
