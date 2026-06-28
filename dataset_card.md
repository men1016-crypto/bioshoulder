# Dataset Card: BioShoulder

## Summary

BioShoulder contains generated shoulder muscle activation and muscle force
windows aligned to AMASS sequence keys. It is designed for users who have
separate licensed access to AMASS and, when needed, BABEL.

## Data

- Number of windows: 48,137
- Window length: 28 frames
- Sample rate: 20 Hz
- Window duration: 1.4 s
- Stride: 14 frames
- Overlap: 0.5
- Targets:
  - `Y_activation [48137, 28, 40]`
  - `Y_force [48137, 28, 40]`

## Licensing Boundary

This package excludes AMASS motion data, BABEL annotations, and HumanML3D-derived
motion features. The metadata includes relative AMASS keys only so users can
align labels with their own local AMASS copy.

## Intended Use

The package can be used to train or evaluate muscle activation/force prediction
models after users reconstruct motion inputs from their separately licensed
motion data.

## Splits

The release includes JSON split files only. The standard split is sequence-level
random 8:1:1. The optional action-disjoint split is intended for unseen-action
evaluation; it contains only train/val/test sequence assignments and does not
publish BABEL action labels.
