#!/usr/bin/env python3
"""I/O helpers for the BioShoulder release.

The release contains muscle targets and alignment metadata only. Motion inputs
must be reconstructed locally from a separately licensed AMASS copy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_SPLIT = "sequence_split_8_1_1_seed42.json"
ACTION_SPLIT = "primary_action_disjoint_split_8_1_1_seed42.json"


@dataclass(frozen=True)
class BioShoulderWindow:
    window_id: int
    target_row_index: int
    sequence_id: str
    amass_source_key: str
    database: str
    subject: str
    experiment: str
    sequence_name: str
    window_start_frame: int
    window_end_frame: int
    window_start_time_s: float
    window_end_time_s: float
    window_duration_s: float
    sample_rate_hz: float

    @classmethod
    def from_dict(cls, row: dict) -> "BioShoulderWindow":
        return cls(
            window_id=int(row["window_id"]),
            target_row_index=int(row["target_row_index"]),
            sequence_id=str(row["sequence_id"]),
            amass_source_key=str(row["amass_source_key"]),
            database=str(row.get("database", "")),
            subject=str(row.get("subject", "")),
            experiment=str(row.get("experiment", "")),
            sequence_name=str(row.get("sequence_name", "")),
            window_start_frame=int(row["window_start_frame"]),
            window_end_frame=int(row["window_end_frame"]),
            window_start_time_s=float(row["window_start_time_s"]),
            window_end_time_s=float(row["window_end_time_s"]),
            window_duration_s=float(row["window_duration_s"]),
            sample_rate_hz=float(row["sample_rate_hz"]),
        )


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_windows_metadata(release_root: Path) -> list[BioShoulderWindow]:
    path = release_root / "metadata" / "windows_metadata.jsonl"
    return [BioShoulderWindow.from_dict(row) for row in iter_jsonl(path)]


def load_labels(release_root: Path) -> np.lib.npyio.NpzFile:
    return np.load(release_root / "processed" / "muscle_activation_force_windows.npz", allow_pickle=True)


def load_split(release_root: Path, split_name: str = DEFAULT_SPLIT) -> dict:
    return json.loads((release_root / "splits" / split_name).read_text(encoding="utf-8"))


def resolve_amass_path(amass_root: Path, amass_source_key: str) -> Path:
    return amass_root.expanduser().resolve() / amass_source_key


def split_indices_from_sequence_ids(windows: list[BioShoulderWindow], split: dict) -> dict[str, np.ndarray]:
    sequence_ids = np.asarray([row.sequence_id for row in windows])
    out: dict[str, np.ndarray] = {}
    for name in ("train", "val", "test"):
        keep = set(str(x) for x in split[f"{name}_sequence_ids"])
        out[f"{name}_indices"] = np.flatnonzero(np.isin(sequence_ids, list(keep)))
    return out


def index_windows_by_amass_key(windows: list[BioShoulderWindow]) -> dict[str, list[int]]:
    index: dict[str, list[int]] = {}
    for row_idx, row in enumerate(windows):
        index.setdefault(row.amass_source_key, []).append(row_idx)
    return index


def get_targets_for_amass_key(
    labels: np.lib.npyio.NpzFile,
    windows: list[BioShoulderWindow],
    amass_source_key: str,
) -> dict[str, np.ndarray | list[BioShoulderWindow]]:
    row_indices = index_windows_by_amass_key(windows).get(amass_source_key, [])
    if not row_indices:
        raise KeyError(f"No BioShoulder windows found for AMASS key: {amass_source_key}")
    target_rows = np.asarray([windows[i].target_row_index for i in row_indices], dtype=np.int64)
    return {
        "windows": [windows[i] for i in row_indices],
        "Y_activation": np.asarray(labels["Y_activation"], dtype=np.float32)[target_rows],
        "Y_force": np.asarray(labels["Y_force"], dtype=np.float32)[target_rows],
        "target_row_index": target_rows,
    }


def validate_release(release_root: Path) -> None:
    labels = load_labels(release_root)
    windows = load_windows_metadata(release_root)
    if labels["Y_activation"].shape[0] != len(windows):
        raise ValueError(
            "Y_activation and metadata row counts differ: "
            f"{labels['Y_activation'].shape[0]} vs {len(windows)}"
        )
    if labels["Y_force"].shape[:2] != labels["Y_activation"].shape[:2]:
        raise ValueError("Y_force and Y_activation have incompatible shapes")


def validate_amass_and_babel_paths(
    windows: list[BioShoulderWindow],
    amass_root: Path,
    babel_root: Path | None = None,
    max_missing_to_report: int = 10,
) -> dict[str, object]:
    missing_amass: list[str] = []
    seen: set[str] = set()
    for row in windows:
        if row.amass_source_key in seen:
            continue
        seen.add(row.amass_source_key)
        if not resolve_amass_path(amass_root, row.amass_source_key).exists():
            missing_amass.append(row.amass_source_key)

    babel_files: dict[str, bool] = {}
    if babel_root is not None:
        for name in ("train.json", "val.json", "extra_train.json", "extra_val.json"):
            babel_files[name] = (babel_root / name).exists()

    return {
        "num_sequences": len(seen),
        "num_missing_amass": len(missing_amass),
        "missing_amass_examples": missing_amass[:max_missing_to_report],
        "babel_files": babel_files,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Inspect a BioShoulder release directory.")
    parser.add_argument("--release-root", type=Path, default=Path("."))
    parser.add_argument("--amass-root", type=Path, default=None)
    parser.add_argument("--babel-root", type=Path, default=None)
    parser.add_argument("--split", default=DEFAULT_SPLIT, choices=[DEFAULT_SPLIT, ACTION_SPLIT])
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    validate_release(args.release_root)
    labels = load_labels(args.release_root)
    windows = load_windows_metadata(args.release_root)
    split = load_split(args.release_root, args.split)
    split_indices = split_indices_from_sequence_ids(windows, split)
    row = windows[int(args.index)]

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
    print("window metadata:", json.dumps(row.__dict__, ensure_ascii=False, indent=2))

    if args.amass_root is not None:
        print("local AMASS path:", resolve_amass_path(args.amass_root, row.amass_source_key))
        report = validate_amass_and_babel_paths(windows, args.amass_root, args.babel_root)
        print("path validation:", json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
