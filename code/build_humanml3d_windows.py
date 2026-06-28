#!/usr/bin/env python3
"""Build paired HumanML3D-motion / BioShoulder-target windows.

BioShoulder does not redistribute AMASS or HumanML3D-derived motion features.
This script reconstructs the paired training NPZ locally from:

1. the BioShoulder label release,
2. a user's licensed AMASS path,
3. HumanML3D 263-D sequence features produced locally.

The sequence feature generation step is intentionally adapter-based. Users can
either point to an existing cache of per-sequence `humanml3d_repr_263.npy` files
or provide a converter command that maps one AMASS file to one `.npy`.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from collections import OrderedDict
from pathlib import Path

import numpy as np

from bioshoulder_io import (
    load_labels,
    load_windows_metadata,
    resolve_amass_path,
    validate_amass_and_babel_paths,
    validate_release,
)


def sequence_cache_path(cache_root: Path, amass_source_key: str) -> Path:
    stem = Path(amass_source_key).with_suffix("")
    return cache_root / stem / "humanml3d_repr_263.npy"


def run_converter(
    converter_cmd: str,
    amass_path: Path,
    out_path: Path,
    humanml3d_repo: Path | None,
    babel_root: Path | None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    values = {
        "amass": str(amass_path),
        "out": str(out_path),
        "humanml3d_repo": "" if humanml3d_repo is None else str(humanml3d_repo),
        "babel_root": "" if babel_root is None else str(babel_root),
    }
    cmd = converter_cmd.format(**values)
    subprocess.run(shlex.split(cmd), check=True)


def load_or_create_sequence_feature(
    amass_source_key: str,
    amass_root: Path,
    cache_root: Path,
    converter_cmd: str | None,
    humanml3d_repo: Path | None,
    babel_root: Path | None,
) -> np.ndarray:
    path = sequence_cache_path(cache_root, amass_source_key)
    if not path.exists():
        if converter_cmd is None:
            raise FileNotFoundError(
                f"Missing cached HumanML3D feature: {path}\n"
                "Provide --converter-cmd to create it from AMASS, or precompute the cache."
            )
        run_converter(
            converter_cmd=converter_cmd,
            amass_path=resolve_amass_path(amass_root, amass_source_key),
            out_path=path,
            humanml3d_repo=humanml3d_repo,
            babel_root=babel_root,
        )
    arr = np.load(path).astype(np.float32)
    if arr.ndim != 2 or arr.shape[1] != 263:
        raise ValueError(f"Expected HumanML3D feature [T,263] at {path}, got {arr.shape}")
    return arr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pair local HumanML3D features with BioShoulder labels.")
    parser.add_argument("--release-root", type=Path, required=True, help="BioShoulder dataset directory.")
    parser.add_argument("--amass-root", type=Path, required=True, help="Local AMASS root.")
    parser.add_argument("--babel-root", type=Path, default=None, help="Optional local BABEL root for converter commands.")
    parser.add_argument("--humanml3d-repo", type=Path, default=None, help="Optional official HumanML3D repo path.")
    parser.add_argument("--motion-cache-root", type=Path, required=True, help="Cache for per-sequence HumanML3D .npy files.")
    parser.add_argument(
        "--converter-cmd",
        default=None,
        help=(
            "Optional command template to create a sequence feature. Placeholders: "
            "{amass}, {out}, {humanml3d_repo}, {babel_root}."
        ),
    )
    parser.add_argument("--out-npz", type=Path, required=True, help="Output paired training NPZ.")
    parser.add_argument("--limit", type=int, default=0, help="Debug: only process the first N windows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_release(args.release_root)
    labels = load_labels(args.release_root)
    windows = load_windows_metadata(args.release_root)
    if args.limit > 0:
        windows = windows[: int(args.limit)]

    report = validate_amass_and_babel_paths(windows, args.amass_root, args.babel_root)
    if args.converter_cmd is not None and int(report["num_missing_amass"]) > 0:
        raise FileNotFoundError(json.dumps(report, ensure_ascii=False, indent=2))
    if int(report["num_missing_amass"]) > 0:
        print(
            "[build_humanml3d_windows] Warning: some AMASS files are missing. "
            "Continuing because --converter-cmd was not provided and cached motion features may be used.",
            flush=True,
        )

    feature_cache: OrderedDict[str, np.ndarray] = OrderedDict()
    x_motion: list[np.ndarray] = []
    target_rows: list[int] = []
    kept_windows: list[dict] = []

    for row in windows:
        if row.amass_source_key not in feature_cache:
            feature_cache[row.amass_source_key] = load_or_create_sequence_feature(
                amass_source_key=row.amass_source_key,
                amass_root=args.amass_root,
                cache_root=args.motion_cache_root,
                converter_cmd=args.converter_cmd,
                humanml3d_repo=args.humanml3d_repo,
                babel_root=args.babel_root,
            )
        seq_feat = feature_cache[row.amass_source_key]
        start = int(row.window_start_frame)
        end = int(row.window_end_frame)
        if end > seq_feat.shape[0]:
            raise ValueError(
                f"Window {row.window_id} exceeds feature length for {row.amass_source_key}: "
                f"{start}:{end} vs {seq_feat.shape[0]}"
            )
        x_motion.append(seq_feat[start:end])
        target_rows.append(int(row.target_row_index))
        kept_windows.append(row.__dict__)

    target_rows_np = np.asarray(target_rows, dtype=np.int64)
    x = np.stack(x_motion).astype(np.float32)
    y_activation = np.asarray(labels["Y_activation"], dtype=np.float32)[target_rows_np]
    y_force = np.asarray(labels["Y_force"], dtype=np.float32)[target_rows_np]

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_npz,
        X_motion=x,
        Y_activation=y_activation,
        Y_force=y_force,
        muscle_names=np.asarray(labels["muscle_names"], dtype=object),
        force_muscle_names=np.asarray(labels["force_muscle_names"], dtype=object),
        sequence_id=np.asarray([row["sequence_id"] for row in kept_windows], dtype=object),
        amass_source_key=np.asarray([row["amass_source_key"] for row in kept_windows], dtype=object),
        target_row_index=target_rows_np,
        window_start_frame=np.asarray([row["window_start_frame"] for row in kept_windows], dtype=np.int32),
        window_end_frame=np.asarray([row["window_end_frame"] for row in kept_windows], dtype=np.int32),
        window_start_time_s=np.asarray([row["window_start_time_s"] for row in kept_windows], dtype=np.float32),
        window_end_time_s=np.asarray([row["window_end_time_s"] for row in kept_windows], dtype=np.float32),
        sample_rate_hz=np.asarray([20.0], dtype=np.float32),
    )
    print(json.dumps({"out_npz": str(args.out_npz), "X_motion": list(x.shape), "Y_activation": list(y_activation.shape)}, indent=2))


if __name__ == "__main__":
    main()
