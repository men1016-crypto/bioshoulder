# BioShoulder

Official code scaffold for **BioShoulder**, a label-only benchmark for predicting
shoulder muscle activation and muscle force from human motion.

BioShoulder releases the muscle targets and alignment metadata only. It does not
redistribute AMASS motion files, BABEL annotations, HumanML3D feature arrays, or
pathpoint/keypoint data. Users reconstruct motion inputs locally from their own
licensed AMASS/BABEL copies.

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

## Data Fields

`processed/muscle_activation_force_windows.npz` contains:

- `Y_activation`: `[48137, 28, 40]`
- `Y_force`: `[48137, 28, 40]`
- `muscle_names`: 40 muscle names
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

## Citation

If you use BioShoulder, please cite the accompanying paper. Also cite AMASS,
BABEL, and HumanML3D when you use their data or preprocessing code.

```bibtex
@misc{bioshoulder2026,
  title = {BioShoulder: Shoulder Muscle Activation and Force Prediction from Human Motion},
  author = {Anonymous},
  year = {2026}
}
```
