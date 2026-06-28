#!/usr/bin/env python3
"""Train a compact Transformer baseline for HumanML3D -> BioShoulder targets."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from bioshoulder_io import load_split, split_indices_from_sequence_ids


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fit_standardizer(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = arr.mean(axis=(0, 1), keepdims=True)
    std = arr.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


class ArrayDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerRegressor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, d_model: int, nhead: int, layers: int, ff_dim: int, dropout: float):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.pos(self.in_proj(x))
        z = self.encoder(z)
        return self.head(self.norm(z))


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    diff = y_pred - y_true
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))
    true_flat = y_true.reshape(-1)
    pred_flat = y_pred.reshape(-1)
    if np.std(true_flat) < 1e-12 or np.std(pred_flat) < 1e-12:
        pcc = 0.0
    else:
        pcc = float(np.corrcoef(true_flat, pred_flat)[0, 1])
    return {"rmse": rmse, "mae": mae, "pcc": pcc}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, y_mean: np.ndarray, y_std: np.ndarray) -> dict[str, float]:
    model.eval()
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    for x, y in loader:
        pred = model(x.to(device)).cpu().numpy()
        preds.append(pred)
        trues.append(y.numpy())
    pred = np.concatenate(preds, axis=0) * y_std + y_mean
    true = np.concatenate(trues, axis=0) * y_std + y_mean
    return metrics(true, pred)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired-npz", type=Path, required=True, help="Output from build_humanml3d_windows.py.")
    parser.add_argument("--release-root", type=Path, required=True, help="BioShoulder release root with split JSON files.")
    parser.add_argument("--split", default="sequence_split_8_1_1_seed42.json")
    parser.add_argument("--target", choices=["activation", "force"], default="activation")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/transformer_baseline"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    data = np.load(args.paired_npz, allow_pickle=True)
    x = np.asarray(data["X_motion"], dtype=np.float32)
    y_key = "Y_activation" if args.target == "activation" else "Y_force"
    y = np.asarray(data[y_key], dtype=np.float32)
    sequence_ids = np.asarray(data["sequence_id"]).astype(str)

    split = load_split(args.release_root, args.split)
    pseudo_windows = [type("Window", (), {"sequence_id": sid}) for sid in sequence_ids]
    idx = split_indices_from_sequence_ids(pseudo_windows, split)
    train_idx, val_idx, test_idx = idx["train_indices"], idx["val_indices"], idx["test_indices"]
    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            "The selected split produced an empty train/val/test subset. "
            "Use the full paired NPZ or choose a split compatible with this subset."
        )

    x_mean, x_std = fit_standardizer(x[train_idx])
    y_mean, y_std = fit_standardizer(y[train_idx])
    x_norm = (x - x_mean) / x_std
    y_norm = (y - y_mean) / y_std

    loaders = {
        "train": DataLoader(ArrayDataset(x_norm[train_idx], y_norm[train_idx]), batch_size=args.batch_size, shuffle=True),
        "val": DataLoader(ArrayDataset(x_norm[val_idx], y_norm[val_idx]), batch_size=args.batch_size, shuffle=False),
        "test": DataLoader(ArrayDataset(x_norm[test_idx], y_norm[test_idx]), batch_size=args.batch_size, shuffle=False),
    }

    model = TransformerRegressor(x.shape[-1], y.shape[-1], args.d_model, args.nhead, args.layers, args.ff_dim, args.dropout).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.SmoothL1Loss()
    best = float("inf")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for xb, yb in loaders["train"]:
            xb = xb.to(device)
            yb = yb.to(device)
            optim.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optim.step()
            total += float(loss.item()) * int(xb.shape[0])
            count += int(xb.shape[0])
        val_metrics = evaluate(model, loaders["val"], device, y_mean, y_std)
        row = {"epoch": epoch, "train_loss": total / max(1, count), **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(json.dumps(row))
        if val_metrics["rmse"] < best:
            best = val_metrics["rmse"]
            torch.save(model.state_dict(), args.out_dir / "best_model.pt")

    model.load_state_dict(torch.load(args.out_dir / "best_model.pt", map_location=device))
    summary = {
        "paired_npz": str(args.paired_npz),
        "split": args.split,
        "target": args.target,
        "shape": {"X_motion": list(x.shape), y_key: list(y.shape)},
        "samples": {"train": int(len(train_idx)), "val": int(len(val_idx)), "test": int(len(test_idx))},
        "best_val_rmse": float(best),
        "test": evaluate(model, loaders["test"], device, y_mean, y_std),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    np.savez_compressed(args.out_dir / "normalization_stats.npz", x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
