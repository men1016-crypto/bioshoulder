# BioShoulder

This folder is a label-only release package for shoulder muscle activation and
muscle force prediction. It excludes AMASS motion files, BABEL annotations,
HumanML3D 263-D features, and pathpoint data.

For the GitHub code repository, large data artifacts are intentionally ignored
and should be downloaded from the accompanying Hugging Face dataset repository.

## Contents

- `processed/muscle_activation_force_windows.npz`
  - `Y_activation`: `[48137, 28, 40]`
  - `Y_force`: `[48137, 28, 40]`
  - `muscle_names`: 40 muscle labels
  - window IDs, sequence IDs, and window timing arrays
- `metadata/windows_metadata.jsonl`
  - one clean metadata record per window
  - includes `amass_source_key` and window start/end information
  - does not include local absolute paths or BABEL text labels
- `metadata/sequences_metadata.json`
  - one clean metadata record per AMASS sequence
- `splits/sequence_split_8_1_1_seed42.json`
  - fixed sequence-level train/val/test split
- `splits/primary_action_disjoint_split_8_1_1_seed42.json`
  - optional action-generalization split with disjoint primary action groups
  - stores sequence assignments only, not BABEL action text
- `code/load_shoulderact_labels.py`
  - minimal loading and AMASS path-resolution example

## Usage

```bash
python code/load_shoulderact_labels.py --release-root .
```

To inspect the action-generalization split:

```bash
python code/load_shoulderact_labels.py \
  --release-root . \
  --split primary_action_disjoint_split_8_1_1_seed42.json
```

To resolve AMASS paths on a user's machine:

```bash
python code/load_shoulderact_labels.py \
  --release-root . \
  --amass-root /path/to/AMASS
```

The code will map each metadata record's `amass_source_key` to:

```text
/path/to/AMASS/<amass_source_key>
```

Users must obtain AMASS and BABEL separately under their own licenses before
reconstructing motion features or semantic annotations.

## Important Exclusions

This release intentionally does not include:

- `X_motion`
- HumanML3D-derived feature arrays
- AMASS raw motion files
- BABEL raw labels or derived semantic text
- BABEL action names used internally to construct the action-disjoint split
- pathpoint/keypoint data
- local absolute paths from the original processing machine
