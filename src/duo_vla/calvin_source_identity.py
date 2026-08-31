"""Fail-closed CALVIN production source-tree identity helpers."""

from __future__ import annotations

import hashlib
import os
import stat
import struct
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

_IDENTITY_FIELDS = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink")
_DIRECTORY_OPEN_FLAGS = os.O_RDONLY | os.O_NONBLOCK | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_FILE_OPEN_FLAGS = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return all(getattr(left, field) == getattr(right, field) for field in _IDENTITY_FIELDS)


def _canonical_relative_parts(relative_text: str) -> tuple[str, ...]:
    relative = PurePosixPath(relative_text)
    if (
        not relative_text
        or "\\" in relative_text
        or relative.is_absolute()
        or relative.as_posix() != relative_text
        or relative_text == "."
        or ".." in relative.parts
    ):
        raise RuntimeError(f"CALVIN source-tree path is not canonical: {relative_text!r}")
    return relative.parts


def _open_bound_directory(parent_fd: int, name: str, *, context: str) -> tuple[int, os.stat_result]:
    before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISDIR(before.st_mode):
        raise RuntimeError(f"CALVIN source-tree ancestor must be a real directory: {context}")
    descriptor = os.open(name, _DIRECTORY_OPEN_FLAGS, dir_fd=parent_fd)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISDIR(opened.st_mode) or not _same_identity(before, opened):
            raise RuntimeError(f"CALVIN source-tree ancestor changed while opening: {context}")
        return descriptor, opened
    except BaseException:
        os.close(descriptor)
        raise


def _verify_directory_binding(
    parent_fd: int,
    name: str,
    descriptor: int,
    expected: os.stat_result,
    *,
    context: str,
) -> None:
    after_descriptor = os.fstat(descriptor)
    after_path = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if not _same_identity(expected, after_descriptor) or not _same_identity(expected, after_path):
        raise RuntimeError(f"CALVIN source-tree ancestor changed while reading: {context}")


def _open_absolute_directory_chain(path: Path) -> tuple[list[int], list[tuple[int, str, int, os.stat_result, str]]]:
    if not path.is_absolute():
        raise RuntimeError(f"CALVIN source root must be absolute: {path}")
    descriptors = [os.open("/", _DIRECTORY_OPEN_FLAGS)]
    bindings: list[tuple[int, str, int, os.stat_result, str]] = []
    try:
        for index, name in enumerate(path.parts[1:]):
            context = "/" + "/".join(path.parts[1 : index + 2])
            child_fd, child_identity = _open_bound_directory(descriptors[-1], name, context=context)
            bindings.append((descriptors[-1], name, child_fd, child_identity, context))
            descriptors.append(child_fd)
        return descriptors, bindings
    except BaseException:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise


def _verify_absolute_directory_chain(
    descriptors: Sequence[int],
    bindings: Sequence[tuple[int, str, int, os.stat_result, str]],
) -> None:
    root_descriptor = os.fstat(descriptors[0])
    root_path = os.stat("/", follow_symlinks=False)
    if not _same_identity(root_descriptor, root_path):
        raise RuntimeError("CALVIN source-tree filesystem root changed while reading")
    for parent_fd, name, child_fd, expected, context in reversed(bindings):
        _verify_directory_binding(parent_fd, name, child_fd, expected, context=context)


def _walk_source_directory(descriptor: int, prefix: tuple[str, ...], output: list[str]) -> None:
    directory_before = os.fstat(descriptor)
    names_before = sorted(os.listdir(descriptor))
    for name in names_before:
        child_context = "/".join((*prefix, name))
        observed = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if stat.S_ISDIR(observed.st_mode):
            if name == "__pycache__":
                continue
            child_fd, child_identity = _open_bound_directory(descriptor, name, context=child_context)
            try:
                _walk_source_directory(child_fd, (*prefix, name), output)
                _verify_directory_binding(
                    descriptor,
                    name,
                    child_fd,
                    child_identity,
                    context=child_context,
                )
            finally:
                os.close(child_fd)
        elif stat.S_ISREG(observed.st_mode):
            output.append(child_context)
        else:
            raise RuntimeError(f"CALVIN source-tree entry must be a regular file or real directory: {child_context}")
    if names_before != sorted(os.listdir(descriptor)) or not _same_identity(directory_before, os.fstat(descriptor)):
        raise RuntimeError(f"CALVIN source-tree directory changed during inventory: {'/'.join(prefix)}")


def _read_relative_regular_file(root_fd: int, relative_text: str) -> bytes:
    parts = _canonical_relative_parts(relative_text)
    directory_fds = [os.dup(root_fd)]
    bindings: list[tuple[int, str, int, os.stat_result, str]] = []
    file_descriptor: int | None = None
    try:
        for index, name in enumerate(parts[:-1]):
            context = "/".join(parts[: index + 1])
            child_fd, child_identity = _open_bound_directory(directory_fds[-1], name, context=context)
            bindings.append((directory_fds[-1], name, child_fd, child_identity, context))
            directory_fds.append(child_fd)
        file_name = parts[-1]
        try:
            before = os.stat(file_name, dir_fd=directory_fds[-1], follow_symlinks=False)
        except FileNotFoundError as exc:
            raise RuntimeError(f"required CALVIN source-tree entry is missing: {relative_text}") from exc
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"CALVIN source-tree entry must be a regular file: {relative_text}")
        file_descriptor = os.open(file_name, _FILE_OPEN_FLAGS, dir_fd=directory_fds[-1])
        opened = os.fstat(file_descriptor)
        if not stat.S_ISREG(opened.st_mode) or not _same_identity(before, opened):
            raise RuntimeError(f"CALVIN source-tree entry changed while opening: {relative_text}")
        blocks: list[bytes] = []
        while block := os.read(file_descriptor, 1024 * 1024):
            blocks.append(block)
        after_descriptor = os.fstat(file_descriptor)
        after_path = os.stat(file_name, dir_fd=directory_fds[-1], follow_symlinks=False)
        if not _same_identity(opened, after_descriptor) or not _same_identity(opened, after_path):
            raise RuntimeError(f"CALVIN source-tree entry changed while reading: {relative_text}")
        for parent_fd, name, child_fd, expected, context in reversed(bindings):
            _verify_directory_binding(parent_fd, name, child_fd, expected, context=context)
        raw = b"".join(blocks)
        if len(raw) != opened.st_size:
            raise RuntimeError(f"CALVIN source-tree entry size changed while reading: {relative_text}")
        return raw
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        for descriptor in reversed(directory_fds):
            os.close(descriptor)


def calvin_source_tree_sha256(
    root: str | Path,
    *,
    explicit_relative_paths: Sequence[str],
    magic: bytes,
) -> str:
    """Hash an injectively framed inventory using only root-relative nofollow walks."""

    root_path = Path(os.path.abspath(root))
    root_descriptors, root_bindings = _open_absolute_directory_chain(root_path)
    root_fd = root_descriptors[-1]
    try:
        root_opened = os.fstat(root_fd)
        if not stat.S_ISDIR(root_opened.st_mode):
            raise RuntimeError(f"CALVIN source root must be a real directory: {root_path}")
        src_fd, src_identity = _open_bound_directory(root_fd, "src", context="src")
        try:
            duo_fd, duo_identity = _open_bound_directory(src_fd, "duo_vla", context="src/duo_vla")
            try:
                discovered: list[str] = []
                _walk_source_directory(duo_fd, ("src", "duo_vla"), discovered)
                _verify_directory_binding(src_fd, "duo_vla", duo_fd, duo_identity, context="src/duo_vla")
            finally:
                os.close(duo_fd)
            _verify_directory_binding(root_fd, "src", src_fd, src_identity, context="src")
        finally:
            os.close(src_fd)

        paths = [*discovered, *explicit_relative_paths]
        if any(not isinstance(relative, str) for relative in paths):
            raise RuntimeError("CALVIN source-tree inventory paths must be strings")
        for relative in paths:
            _canonical_relative_parts(relative)
        if len(set(paths)) != len(paths):
            raise RuntimeError("CALVIN source-tree inventory contains duplicate relative paths")
        ordered_paths = tuple(sorted(paths))
        digest = hashlib.sha256()
        digest.update(magic)
        digest.update(struct.pack(">Q", len(ordered_paths)))
        for relative_text in ordered_paths:
            relative = relative_text.encode("utf-8")
            raw = _read_relative_regular_file(root_fd, relative_text)
            digest.update(struct.pack(">Q", len(relative)))
            digest.update(relative)
            digest.update(struct.pack(">Q", len(raw)))
            digest.update(raw)
        root_after_descriptor = os.fstat(root_fd)
        if not _same_identity(root_opened, root_after_descriptor):
            raise RuntimeError(f"CALVIN source root changed while hashing: {root_path}")
        _verify_absolute_directory_chain(root_descriptors, root_bindings)
        return digest.hexdigest()
    finally:
        for descriptor in reversed(root_descriptors):
            os.close(descriptor)
