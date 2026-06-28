#!/usr/bin/env python3
"""Load the label-only BioShoulder release package.

This example intentionally does not read AMASS, BABEL, or HumanML3D-derived
motion features. It shows how to load the published muscle labels and how to
resolve the corresponding AMASS file path from user-provided local roots.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_indices_from_sequence_split(metadata: list[dict], split: dict) -> dict[str, np.ndarray]:
    sequence_ids = np.asarray([rec["sequence_id"] for rec in metadata])
    indices = {}
    for name in ("train", "val", "test"):
        keep = set(split[f"{name}_sequence_ids"])
        indices[f"{name}_indices"] = np.flatnonzero(np.isin(sequence_ids, list(keep)))
    return indices


def load_release(root: Path, split_name: str) -> tuple[np.lib.npyio.NpzFile, list[dict], dict, dict[str, np.ndarray]]:
    labels = np.load(root / "processed" / "muscle_activation_force_windows.npz", allow_pickle=True)
    metadata = list(iter_jsonl(root / "metadata" / "windows_metadata.jsonl"))
    split_path = root / "splits" / split_name
    split = json.loads(split_path.read_text(encoding="utf-8"))
    split_indices = build_indices_from_sequence_split(metadata, split)

    if labels["Y_activation"].shape[0] != len(metadata):
        raise ValueError("Label and metadata window counts do not match.")
    return labels, metadata, split, split_indices


def resolve_amass_path(amass_root: Path, amass_source_key: str) -> Path:
    """Map a release metadata key to a local AMASS file.

    The release ships only relative AMASS keys, for example:
    BMLmovi/BMLmovi/Subject_11_F_MoSh/Subject_11_F_12_poses.npz
    """

    return amass_root / amass_source_key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--release-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to the BioShoulder release directory.",
    )
    parser.add_argument(
        "--amass-root",
        type=Path,
        default=None,
        help="Optional local AMASS root used only to demonstrate path resolution.",
    )
    parser.add_argument("--index", type=int, default=0, help="Window index to inspect.")
    parser.add_argument(
        "--split",
        default="sequence_split_8_1_1_seed42.json",
        choices=[
            "sequence_split_8_1_1_seed42.json",
            "primary_action_disjoint_split_8_1_1_seed42.json",
        ],
        help="Split JSON to load.",
    )
    args = parser.parse_args()

    labels, metadata, split, split_indices = load_release(args.release_root, args.split)
    i = args.index
    rec = metadata[i]

    print("Y_activation:", labels["Y_activation"].shape, labels["Y_activation"].dtype)
    print("Y_force:", labels["Y_force"].shape, labels["Y_force"].dtype)
    print("muscles:", [str(x) for x in labels["muscle_names"][:5]], "...")
    print("split:", split["split_name"])
    print(
        "split sizes:",
        len(split_indices["train_indices"]),
        len(split_indices["val_indices"]),
        len(split_indices["test_indices"]),
    )
    print("window metadata:", json.dumps(rec, ensure_ascii=False, indent=2))

    if args.amass_root is not None:
        print("local AMASS path:", resolve_amass_path(args.amass_root, rec["amass_source_key"]))


if __name__ == "__main__":
    main()
