#!/usr/bin/env python3
"""Decode and validate one pinned LIBERO episode anchor without loading the model."""

from __future__ import annotations

import argparse
import json

from duo_vla.data.libero import LiberoParquetDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot_root")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0)
    args = parser.parse_args()
    dataset = LiberoParquetDataset(args.snapshot_root)
    sample = dataset.sample(args.episode, args.frame)
    print(
        json.dumps(
            {
                "episodes": len(dataset.episodes),
                "episode_index": sample.episode_index,
                "frame_index": sample.frame_index,
                "instruction": sample.instruction,
                "task_index": sample.task_index,
                "third_person_shape": list(sample.observation.third_person.shape),
                "wrist_shape": list(sample.observation.wrist.shape),
                "state": sample.observation.state.tolist(),
                "action_chunk_shape": list(sample.action_chunk.actions.shape),
                "valid_actions": int(sample.action_chunk.valid_mask.sum()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
