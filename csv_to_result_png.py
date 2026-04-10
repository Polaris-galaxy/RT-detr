"""
将 RT-DETR 训练生成的 results.csv 绘制成与常见训练总结类似的 result.png
（Ultralytics 默认 plot_results 按 YOLO 的 box/cls/dfl 列索引，不适用于 RT-DETR。）
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.ndimage import gaussian_filter1d

    def _smooth(y: np.ndarray) -> np.ndarray:
        if len(y) < 4:
            return y
        return gaussian_filter1d(y, sigma=3)

except ImportError:

    def _smooth(y: np.ndarray) -> np.ndarray:
        return y


def plot_rtdetr_results_csv(
    csv_path: str | Path,
    out_path: str | Path | None = None,
    dpi: int = 200,
) -> Path:
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    epoch_col = df.columns[0]
    x = pd.to_numeric(df[epoch_col], errors="coerce").values.astype(float)

    panels = [
        ("train/giou_loss", "train / GIoU loss"),
        ("train/cls_loss", "train / cls loss"),
        ("train/l1_loss", "train / L1 loss"),
        ("metrics/precision(B)", "metrics / precision"),
        ("metrics/recall(B)", "metrics / recall"),
        ("val/giou_loss", "val / GIoU loss"),
        ("val/cls_loss", "val / cls loss"),
        ("val/l1_loss", "val / L1 loss"),
        ("metrics/mAP50(B)", "metrics / mAP50"),
        ("metrics/mAP50-95(B)", "metrics / mAP50-95"),
    ]

    fig, axes = plt.subplots(2, 5, figsize=(14, 7), constrained_layout=True)
    axes = axes.ravel()
    run_name = csv_path.parent.name

    for i, (col, title) in enumerate(panels):
        ax = axes[i]
        if col not in df.columns:
            ax.set_visible(False)
            continue
        y = pd.to_numeric(df[col], errors="coerce").fillna(0).values.astype(float)
        ax.plot(x, y, ".-", linewidth=1.5, markersize=4, label="value", color="#1f77b4")
        ys = _smooth(np.nan_to_num(y, nan=0.0))
        ax.plot(x, ys, ":", linewidth=2, label="smooth", color="#ff7f0e")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.35)

    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"Training results — {run_name}", fontsize=12, fontweight="bold")

    out = Path(out_path) if out_path else csv_path.parent / "result.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def main():
    p = argparse.ArgumentParser(description="results.csv → result.png (RT-DETR columns)")
    p.add_argument(
        "csv",
        nargs="?",
        default=r"runs\train\wool_optimized4\results.csv",
        help="path to results.csv",
    )
    p.add_argument(
        "-o",
        "--out",
        default=None,
        help="output PNG path (default: <csv_dir>/result.png)",
    )
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()
    out = plot_rtdetr_results_csv(args.csv, args.out, dpi=args.dpi)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
