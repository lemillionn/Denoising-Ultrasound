# src/evaluate.py

import os
import re
import yaml
import glob
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from models import Generator
from dataset import UltrasoundDataset, collate_fn


def evaluate(checkpoint_path, config_path="configs/default.yaml", output_dir=None):
    # ── Load config ───────────────────────────────────────────────────────────
    cfg = yaml.safe_load(open(config_path))
    test_clean = cfg["test_clean"]
    test_noisy = cfg["test_noisy"]
    batch_size = int(cfg.get("batch_size", 64))

    base_ch = int(cfg.get("base_channels", 32))
    depths  = tuple(cfg.get("depths", [1,2,4]))
    ckpt_dir = cfg.get("checkpoint_dir", "checkpoints")

    # ── Load training metrics for annotation ─────────────────────────────────
    metrics_csv = os.path.join(ckpt_dir, "metrics.csv")
    train_metrics = {}
    if os.path.exists(metrics_csv):
        with open(metrics_csv) as f:
            next(f)
            for line in f:
                e, d, g, v, *_ = line.strip().split(",")
                train_metrics[int(e)] = {"D": float(d), "G": float(g), "val_MSE": float(v)}

    # ── Prepare test dataset ───────────────────────────────────────────────────
    test_ds = UltrasoundDataset(test_clean, test_noisy, normalize=True)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # ── Build model & load weights ─────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = Generator(base_channels=base_ch, depths=depths).to(device)

    saved = torch.load(checkpoint_path, map_location="cpu")
    state_dict = saved.state_dict() if isinstance(saved, Generator) else saved
    G.load_state_dict(state_dict)
    G.eval()

    # ── Compute Test Metrics ───────────────────────────────────────────────────
    mse_list, snr_list = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            out_batch = G(x_batch)
            # Crop to common length
            L = min(out_batch.size(-1), y_batch.size(-1))
            out_batch = out_batch[..., :L]
            y_batch   = y_batch[..., :L]

            for i in range(out_batch.shape[0]):
                clean_i = y_batch[i].cpu().numpy().flatten()
                pred_i  = out_batch[i].cpu().numpy().flatten()
                mse_val = np.mean((clean_i - pred_i)**2)
                snr_val = 10*np.log10(np.mean(clean_i**2) / mse_val)
                mse_list.append(mse_val)
                snr_list.append(snr_val)

    mean_mse = float(np.mean(mse_list))
    mean_snr = float(np.mean(snr_list))
    print(f"Epoch {checkpoint_path}: MSE={mean_mse:.6f}, SNR_impr={mean_snr:.2f} dB")

    # ── Prepare output directory ───────────────────────────────────────────────
    output_dir = output_dir or os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    # ── Select a random test example ───────────────────────────────────────────
    idx = random.randrange(len(test_ds))
    x0_tensor, y0_tensor = test_ds[idx]

    # ── Generate denoised output ───────────────────────────────────────────────
    x0b = x0_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        o0 = G(x0b).squeeze().cpu().numpy()
    x0 = x0_tensor.squeeze().cpu().numpy()
    y0 = y0_tensor.squeeze().cpu().numpy()

    # Crop to original length
    L0 = min(len(o0), len(y0))

    # ── Plot ───────────────────────────────────────────────────────────────────
    plt.figure(figsize=(8,4))
    plt.plot(x0[:L0], label="Noisy")
    plt.plot(y0[:L0], label="Clean")
    plt.plot(o0[:L0], label="Denoised")
    plt.legend()
    epoch_num = re.search(r"epoch(\d+)", os.path.basename(checkpoint_path)).group(1)
    plt.title(f"Denoising Example — Epoch {epoch_num}")

    # Annotate with metrics
    tm = train_metrics.get(int(epoch_num), {})
    info = [f"Test MSE: {mean_mse:.6f}", f"SNR Imp: {mean_snr:.2f} dB"]
    if tm:
        info += [f"Train G: {tm['G']:.4f}", f"Train D: {tm['D']:.4f}", f"Val MSE: {tm['val_MSE']:.6f}"]
    info_txt = '\n'.join(info)
    plt.gca().text(
        0.99, 0.01, info_txt,
        transform=plt.gca().transAxes,
        fontsize=8, ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
    )

    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.tight_layout()

    out_name = f"denoise_epoch{epoch_num}_idx{idx}.png"
    out_path = os.path.join(output_dir, out_name)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    ckpt_dir = cfg.get("checkpoint_dir", "checkpoints")
    out_dir = os.path.join(os.getcwd(), "eval_plots")
    ckpts = sorted(
        glob.glob(os.path.join(ckpt_dir, "generator_epoch*.pt")),
        key=lambda x: int(re.search(r"epoch(\d+)", x).group(1))
    )
    for ckpt in ckpts:
        evaluate(ckpt, output_dir=out_dir)
