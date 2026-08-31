from __future__ import annotations

import ast
import hashlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from calvin_dev_test_support import npz_bytes

from duo_vla.data.calvin_dev_states import CalvinDevCandidate, CalvinDevStateError

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/calvin/export_calvin_dev_replay_bundle.py"
SPEC = importlib.util.spec_from_file_location("calvin_dev_bundle_export", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
EXPORTER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EXPORTER
SPEC.loader.exec_module(EXPORTER)


def _member(global_index: int, *, action_value: float = 0.25) -> bytes:
    robot = np.linspace(0.01, 0.15, 15, dtype=np.float64) + global_index
    robot[14] = -1.0
    scene = np.linspace(0.0, 0.23, 24, dtype=np.float64) + global_index
    action = np.zeros(7, dtype=np.float64)
    action[0] = action_value
    action[6] = -1.0
    return npz_bytes(robot_obs=robot, scene_obs=scene, rel_actions=action)


class _Reader:
    def __init__(self, members: dict[str, bytes]) -> None:
        self.members = members
        self.calls: list[tuple[str, str]] = []

    def member_record(self, path: str) -> Any:
        self.calls.append(("record", path))
        raw = self.members[path]
        return SimpleNamespace(
            global_index=int(path[-11:-4]),
            is_file=True,
            logical_bytes=len(raw),
            logical_sha256=bytes.fromhex(hashlib.sha256(raw).hexdigest()),
            path=path,
            split="training",
        )

    def read_member_bytes(self, path: str) -> bytes:
        self.calls.append(("read", path))
        return self.members[path]


def _candidate(start: int, end: int, annotation: int = 1) -> CalvinDevCandidate:
    return CalvinDevCandidate(
        annotation_index=annotation,
        episode_index=0,
        global_start=start,
        global_end_exclusive=end,
        instruction="do the fixture task",
        task="fixture_task",
        scene="calvin_scene_A",
    )


def test_exporter_reads_exact_ordered_member_bytes_through_public_reader_api() -> None:
    members = {f"training/episode_{index:07d}.npz": _member(index) for index in range(3)}
    reader = _Reader(members)

    replay = EXPORTER.collect_candidate_replays(reader, (_candidate(0, 3),))[0]

    assert replay.actions.shape == (3, 7)
    assert [record["global_index"] for record in replay.member_identities] == [0, 1, 2]
    assert [record["path"] for record in replay.member_identities] == list(members)
    assert replay.frame.source_frame_sha256 == hashlib.sha256(members[next(iter(members))]).hexdigest()
    assert reader.calls == [
        call
        for index in range(3)
        for call in (
            ("record", f"training/episode_{index:07d}.npz"),
            ("read", f"training/episode_{index:07d}.npz"),
            ("record", f"training/episode_{index:07d}.npz"),
        )
    ]


def test_exporter_rejects_member_byte_or_identity_tampering() -> None:
    path = "training/episode_0000000.npz"
    raw = _member(0)

    class ChangedBytes(_Reader):
        def read_member_bytes(self, member_path: str) -> bytes:
            return super().read_member_bytes(member_path) + b"tampered"

    with pytest.raises(CalvinDevStateError, match="logical byte count differs"):
        EXPORTER.collect_candidate_replays(ChangedBytes({path: raw}), (_candidate(0, 1),))

    class ChangedIdentity(_Reader):
        count = 0

        def member_record(self, member_path: str) -> Any:
            record = super().member_record(member_path)
            self.count += 1
            if self.count == 2:
                record.logical_bytes += 1
            return record

    with pytest.raises(CalvinDevStateError, match="identity changed"):
        EXPORTER.collect_candidate_replays(ChangedIdentity({path: raw}), (_candidate(0, 1),))


def test_exporter_rejects_missing_and_reordered_candidate_member_ranges() -> None:
    members = {
        "training/episode_0000000.npz": _member(0),
        "training/episode_0000001.npz": _member(1),
    }
    with pytest.raises(KeyError):
        EXPORTER.collect_candidate_replays(_Reader(members), (_candidate(0, 3),))

    replays = EXPORTER.collect_candidate_replays(
        _Reader(members),
        (_candidate(1, 2, annotation=2), _candidate(0, 1, annotation=1)),
    )
    assert [replay.candidate.annotation_index for replay in replays] == [2, 1]


def test_exporter_source_uses_only_public_phase_a_reader_methods() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(SCRIPT))
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert "_record" not in attributes
    assert "_authenticated_manifest" not in attributes
    assert {
        "authenticated_manifest",
        "member_record",
        "read_authenticated_metadata_bytes",
        "read_member_bytes",
    } <= attributes
    assert "requires Python 3.11" in source
