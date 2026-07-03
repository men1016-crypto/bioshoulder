# Dataset Card: BioShoulder

## Summary

BioShoulder is a shoulder motion-to-muscle dataset release. It contains shoulder
muscle activation and muscle force windows aligned to AMASS sequence keys. The
accompanying paper focuses on dense shoulder muscle activation prediction from
3D human motion; the release also includes force targets for downstream
biomechanical analysis.

The dataset is designed for users who have separate licensed access to AMASS
and, when needed, BABEL. It provides generated labels, metadata, and benchmark
splits, but not motion features or semantic annotations derived from those
external datasets.

## Data

- Number of windows: 48,137
- Number of source continuous trials: 1,886
- Number of subjects: 302
- Total duration: 9.732 h
- Observed upper-limb/no-contact action categories: 30
- Window length: 28 frames
- Sample rate: 20 Hz
- Window duration: 1.4 s
- Stride: 14 frames / 0.7 s
- Overlap: 50%
- Targets:
  - `Y_activation [48137, 28, 40]`
  - `Y_force [48137, 28, 40]`
- Metadata:
  - one JSONL record per window
  - one JSON record per source sequence
  - relative AMASS source keys for local alignment

## Licensing Boundary

This package excludes AMASS motion data, BABEL annotations, and HumanML3D-derived
motion features. The metadata includes relative AMASS keys only so users can
align labels with their own local AMASS copy.

The action-disjoint split stores only sequence assignments. It does not publish
raw BABEL labels, semantic text, local absolute paths, HumanML3D feature arrays,
or pathpoint/keypoint data.

## Intended Use

The package can be used to train or evaluate muscle activation/force prediction
models after users reconstruct motion inputs from their separately licensed
motion data.

Appropriate uses include:

- sequence-to-sequence prediction from HumanML3D-style motion features to
  shoulder muscle activation or force;
- benchmarking model generalization under sequence-level and action-disjoint
  splits;
- biomechanical representation learning for rehabilitation, sports science,
  human motion understanding, and human-robot collaboration research.

Out-of-scope uses include direct clinical diagnosis, real EMG replacement
without validation, or modeling contact-rich/heavy-payload actions. BioShoulder
is simulation-derived and may have a synthetic-to-real domain gap.

## Splits

The release includes JSON split files only. The standard split is sequence-level
random 8:1:1. The optional action-disjoint split is intended for unseen-action
evaluation; it contains only train/val/test sequence assignments and does not
publish BABEL action labels.

| Split | Train windows | Val windows | Test windows | Train seq. | Val seq. | Test seq. |
|---|---:|---:|---:|---:|---:|---:|
| `sequence_split_8_1_1_seed42.json` | 37,597 | 5,354 | 5,186 | 1,508 | 189 | 189 |
| `primary_action_disjoint_split_8_1_1_seed42.json` | 38,509 | 4,814 | 4,814 | 1,279 | 393 | 214 |

## Citation

If you use BioShoulder, cite the accompanying paper and the external datasets or
preprocessing code used to reconstruct motion inputs, including AMASS, BABEL, and
HumanML3D where applicable.
