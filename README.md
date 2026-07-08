# BioShoulder

Official release scaffold for **BioShoulder**, a dataset release for predicting
shoulder muscle activation and muscle force from human motion.

This package includes activation and force target arrays, alignment metadata,
and benchmark splits. The accompanying code shows how to inspect the labels,
pair them with locally reconstructed HumanML3D motion features, and train a
Transformer baseline.

This repository intentionally releases the generated muscle targets and alignment
metadata only. It does not redistribute AMASS motion files, BABEL annotations,
HumanML3D feature arrays, or pathpoint/keypoint data. Users reconstruct motion
inputs locally from their own licensed AMASS/BABEL copies.

## At a Glance

| Item | Value |
|---|---:|
| Subjects | 302 |
| Continuous trials | 1,886 |
| Total duration | 9.732 h |
| Fixed-length windows | 48,137 |
| Observed upper-limb/no-contact action categories | 30 |
| Muscle target channels | 40 bilateral shoulder muscle strands |
| Sampling rate | 20 Hz |
| Window length | 1.4 s / 28 frames |
| Stride / overlap | 0.7 s / 50% |

## News

- Initial release: label loading utilities, AMASS/BABEL path validation,
  HumanML3D feature pairing script, and a Transformer baseline.

## Repository Layout

```text
code/
  bioshoulder_io.py              # Core BioShoulder label/metadata/split I/O
  load_shoulderact_labels.py     # Backward-compatible inspection entry point
  build_humanml3d_windows.py     # Pair local HumanML3D features with labels
  train_transformer_baseline.py  # Minimal Transformer baseline
requirements.txt
```

The large data files are hosted in the accompanying Hugging Face dataset
repository and are intentionally ignored in this GitHub repository.

Expected dataset files after download:

```text
bioshoulder/
  processed/muscle_activation_force_windows.npz
  metadata/windows_metadata.jsonl
  metadata/sequences_metadata.json
  splits/sequence_split_8_1_1_seed42.json
  splits/primary_action_disjoint_split_8_1_1_seed42.json
  manifest.json
```

## Installation

```bash
git clone https://github.com/men1016-crypto/bioshoulder.git
cd bioshoulder

conda create -n bioshoulder python=3.10 -y
conda activate bioshoulder
pip install -r requirements.txt
```

Download the BioShoulder data from the Hugging Face dataset page and place the
files under this repository root, so `processed/`, `metadata/`, and `splits/`
sit next to `code/`.

Quick smoke test:

```bash
python code/bioshoulder_io.py --release-root .
```

## External Data

Before reconstructing motion inputs, download the following resources under
their own terms:

- AMASS: https://amass.is.tue.mpg.de/
- AMASS license: https://amass.is.tue.mpg.de/license.html
- BABEL: https://babel.is.tue.mpg.de/
- BABEL license: https://babel.is.tue.mpg.de/license.html
- HumanML3D code: https://github.com/EricGuo5513/HumanML3D
- HumanML3D license: https://github.com/EricGuo5513/HumanML3D/blob/main/LICENSE

BioShoulder metadata stores relative AMASS keys such as:

```text
BMLmovi/BMLmovi/Subject_11_F_MoSh/Subject_11_F_12_poses.npz
```

Given `--amass-root /path/to/AMASS`, the code resolves this to:

```text
/path/to/AMASS/BMLmovi/BMLmovi/Subject_11_F_MoSh/Subject_11_F_12_poses.npz
```

## Inspect Labels and Metadata

```bash
python code/bioshoulder_io.py \
  --release-root . \
  --amass-root /path/to/AMASS \
  --babel-root /path/to/BABEL
```

This verifies that labels and metadata have matching rows, prints the selected
split sizes, and checks whether the AMASS files referenced by `amass_source_key`
exist on your machine.

In Python, retrieve all muscle windows aligned with one AMASS sequence key:

```python
from pathlib import Path
import sys

sys.path.append("code")
from bioshoulder_io import (
    get_targets_for_amass_key,
    load_labels,
    load_windows_metadata,
    resolve_amass_path,
)

release_root = Path(".")
amass_root = Path("/path/to/AMASS")
labels = load_labels(release_root)
windows = load_windows_metadata(release_root)

amass_key = windows[0].amass_source_key
amass_file = resolve_amass_path(amass_root, amass_key)
targets = get_targets_for_amass_key(labels, windows, amass_key)

print(amass_file)
print(targets["Y_activation"].shape, targets["Y_force"].shape)
```

Use the action-disjoint split with:

```bash
python code/bioshoulder_io.py \
  --release-root . \
  --split primary_action_disjoint_split_8_1_1_seed42.json
```

The action-disjoint split contains only sequence assignments. It does not
publish BABEL action names or text annotations.

## Reconstruct HumanML3D Motion Features

BioShoulder expects HumanML3D-style motion windows with shape `[N, 28, 263]`.
Because these features are derived from AMASS, they are not redistributed.

First, obtain or implement an AMASS-to-HumanML3D sequence converter using the
official HumanML3D preprocessing code. The converter should map one AMASS file
to one NumPy file:

```text
input:  /path/to/AMASS/<amass_source_key>
output: /path/to/cache/<amass_source_key_without_npz>/humanml3d_repr_263.npy
shape:  [T, 263]
fps:    20 Hz
```

Then pair the generated sequence features with BioShoulder labels:

```bash
python code/build_humanml3d_windows.py \
  --release-root . \
  --amass-root /path/to/AMASS \
  --babel-root /path/to/BABEL \
  --humanml3d-repo /path/to/HumanML3D \
  --motion-cache-root ./motion_cache \
  --converter-cmd "python /path/to/your_amass_to_humanml3d.py --input {amass} --output {out} --humanml3d-repo {humanml3d_repo}" \
  --out-npz ./data/bioshoulder_humanml3d_windows.npz
```

If you already have per-sequence HumanML3D features cached at the expected
locations, omit `--converter-cmd`:

```bash
python code/build_humanml3d_windows.py \
  --release-root . \
  --amass-root /path/to/AMASS \
  --motion-cache-root ./motion_cache \
  --out-npz ./data/bioshoulder_humanml3d_windows.npz
```

The output NPZ contains:

```text
X_motion      [N, 28, 263]
Y_activation  [N, 28, 40]
Y_force       [N, 28, 40]
sequence_id   [N]
target_row_index [N]
```

`target_row_index` is the direct row mapping back to
`processed/muscle_activation_force_windows.npz`, so the same AMASS window is
aligned with the corresponding muscle activation and force target.

## Benchmark Splits

BioShoulder provides two sequence-level JSON split files:

| Split | Train windows | Val windows | Test windows | Train seq. | Val seq. | Test seq. |
|---|---:|---:|---:|---:|---:|---:|
| `sequence_split_8_1_1_seed42.json` | 37,597 | 5,354 | 5,186 | 1,508 | 189 | 189 |
| `primary_action_disjoint_split_8_1_1_seed42.json` | 38,509 | 4,814 | 4,814 | 1,279 | 393 | 214 |

The standard split assigns all windows from the same source sequence to the same
subset, preventing leakage through overlapping windows. The action-disjoint
split evaluates generalization to held-out semantic motion types; it stores only
sequence assignments and intentionally excludes BABEL action labels.

## Train the Transformer Baseline

```bash
python code/train_transformer_baseline.py \
  --paired-npz ./data/bioshoulder_humanml3d_windows.npz \
  --release-root . \
  --split sequence_split_8_1_1_seed42.json \
  --target activation \
  --out-dir ./outputs/transformer_activation \
  --epochs 100 \
  --batch-size 128 \
  --device cuda
```

For force prediction:

```bash
python code/train_transformer_baseline.py \
  --paired-npz ./data/bioshoulder_humanml3d_windows.npz \
  --release-root . \
  --target force \
  --out-dir ./outputs/transformer_force
```

For unseen-action evaluation:

```bash
python code/train_transformer_baseline.py \
  --paired-npz ./data/bioshoulder_humanml3d_windows.npz \
  --release-root . \
  --split primary_action_disjoint_split_8_1_1_seed42.json \
  --target activation \
  --out-dir ./outputs/transformer_action_disjoint
```

The baseline writes:

```text
best_model.pt
history.json
normalization_stats.npz
summary.json
```

The accompanying paper also reports LSTM, FConv, Mamba2, and Transformer
sequence-to-sequence baselines with RMSE, SMAPE, and PCC. This lightweight
release code includes the Transformer baseline as a reproducible starting point.

Paper benchmark results for activation prediction:

| Split | Model | RMSE (10^-3) | SMAPE (%) | PCC |
|---|---|---:|---:|---:|
| Sequence-level | LSTM | 22.201 +/- 0.132 | 21.411 +/- 0.366 | 0.586 +/- 0.005 |
| Sequence-level | FConv | 20.472 +/- 0.140 | 22.613 +/- 0.217 | 0.674 +/- 0.004 |
| Sequence-level | Mamba2 | 20.229 +/- 0.228 | 20.156 +/- 0.401 | 0.680 +/- 0.009 |
| Sequence-level | Transformer | 19.891 +/- 0.157 | 18.914 +/- 0.572 | 0.697 +/- 0.009 |
| Action-disjoint | LSTM | 26.338 +/- 0.166 | 23.792 +/- 0.159 | 0.539 +/- 0.006 |
| Action-disjoint | FConv | 25.064 +/- 0.159 | 24.596 +/- 0.177 | 0.606 +/- 0.010 |
| Action-disjoint | Mamba2 | 24.579 +/- 0.131 | 22.876 +/- 0.702 | 0.621 +/- 0.002 |
| Action-disjoint | Transformer | 24.123 +/- 0.222 | 21.114 +/- 0.358 | 0.640 +/- 0.011 |

## Data Fields

`processed/muscle_activation_force_windows.npz` contains:

- `Y_activation`: `[48137, 28, 40]`
- `Y_force`: `[48137, 28, 40]`
- `muscle_names`: 40 activation target names
- `force_muscle_names`: 40 force target names
- window IDs, sequence IDs, and timing arrays

`metadata/windows_metadata.jsonl` contains one record per target window:

```json
{
  "window_id": 0,
  "target_row_index": 0,
  "sequence_id": "BMLmovi/Subject_11_F_MoSh/Subject_11_F_12",
  "amass_source_key": "BMLmovi/BMLmovi/Subject_11_F_MoSh/Subject_11_F_12_poses.npz",
  "window_start_frame": 0,
  "window_end_frame": 28,
  "sample_rate_hz": 20.0
}
```

`metadata/sequences_metadata.json` contains one record per source sequence:

```json
{
  "sequence_id": "BMLmovi/Subject_11_F_MoSh/Subject_11_F_12",
  "amass_source_key": "BMLmovi/BMLmovi/Subject_11_F_MoSh/Subject_11_F_12_poses.npz",
  "database": "BMLmovi",
  "subject": "Subject_11_F_MoSh",
  "experiment": "Subject_11_F_12",
  "sequence_name": "Subject_11_F_12_poses",
  "num_windows": 3,
  "first_window_id": 0,
  "last_window_id": 2
}
```

## License and Data Policy

This repository contains code only. BioShoulder data files should be downloaded
from the accompanying Hugging Face dataset repository.

BioShoulder intentionally excludes:

- AMASS raw motion files
- BABEL raw or derived text labels
- HumanML3D-derived motion feature arrays
- pathpoint/keypoint data
- local absolute paths from the original processing machine

Please follow the licenses of AMASS, BABEL, HumanML3D, and any AMASS source
subsets you use.

BioShoulder is intended for academic research in biomechanics, rehabilitation,
human motion understanding, and human-robot collaboration. Because the labels are
simulation-derived, a synthetic-to-real gap may remain when applying models to
real physiological signals. The dataset also focuses on free-space upper-limb
motions and is not designed for modeling heavy payloads or contact-rich
human-object interaction.
