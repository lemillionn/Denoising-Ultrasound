# src/train.py

import os
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import Generator, Discriminator
from dataset import UltrasoundDataset, collate_fn
from utils import set_seed, save_checkpoint


def weights_init(m):
    """Xavier initialization for Conv1d and Linear layers."""
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def main(config_path="configs/default.yaml"):
    # Load config & seed
    cfg = yaml.safe_load(open(config_path))
    set_seed(cfg.get("seed", 42))

    # Dataset params
    train_clean = cfg["train_clean"]
    train_noisy = cfg["train_noisy"]
    val_clean   = cfg["val_clean"]
    val_noisy   = cfg["val_noisy"]
    batch_size  = int(cfg.get("batch_size", 64))

    # Training params
    epochs      = int(cfg.get("epochs", 20))
    lr          = float(cfg.get("lr", 5e-4))
    lambda_adv  = float(cfg.get("lambda_adv", 5e-4))

    # Model architecture
    base_ch = int(cfg.get("base_channels", 32))
    depths  = tuple(cfg.get("depths", [1, 2, 4]))

    # Checkpoint setup
    ckpt_dir = cfg.get("checkpoint_dir", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    metrics_csv = os.path.join(ckpt_dir, "metrics.csv")
    if not os.path.exists(metrics_csv):
        with open(metrics_csv, "w") as f:
            f.write("epoch,train_D,train_G,val_MSE,lr\n")

    # Data loaders
    train_ds = UltrasoundDataset(train_clean, train_noisy, normalize=True)
    val_ds   = UltrasoundDataset(val_clean,   val_noisy,   normalize=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # Build models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = Generator(base_channels=base_ch, depths=depths).to(device)
    D = Discriminator(base_channels=base_ch).to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    # Optimizers and LR scheduler
    opt_G = torch.optim.Adam(G.parameters(), lr=lr)
    opt_D = torch.optim.Adam(D.parameters(), lr=lr)
    scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_G, mode='min', factor=0.5, patience=3
    )

    # Training loop
    for epoch in range(1, epochs + 1):
        G.train()
        D.train()
        running_D, running_G = 0.0, 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # Discriminator step
            fake = G(x).detach()
            D_real = D(y)
            D_fake = D(fake)
            loss_D = F.mse_loss(D_real, torch.ones_like(D_real)) + \
                     F.mse_loss(D_fake, torch.zeros_like(D_fake))
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()
            running_D += loss_D.item()

            # Generator step
            fake = G(x)
            rec_loss = F.mse_loss(fake, y)
            adv_loss = F.mse_loss(D(fake), torch.ones_like(D_fake))
            loss_G = rec_loss + lambda_adv * adv_loss
            opt_G.zero_grad()
            loss_G.backward()
            nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
            opt_G.step()
            running_G += loss_G.item()

        avg_D = running_D / len(train_loader)
        avg_G = running_G / len(train_loader)

        # Validation
        G.eval()
        val_mse = 0.0
        with torch.no_grad():
            for xv, yv in val_loader:
                xv, yv = xv.to(device), yv.to(device)
                pred = G(xv)
                val_mse += F.mse_loss(pred, yv).item()
        val_mse /= len(val_loader)

        # Logging
        lr_now = opt_G.param_groups[0]['lr']
        print(f"Epoch {epoch}/{epochs}  D: {avg_D:.4f}  G: {avg_G:.4f}  Val_MSE: {val_mse:.6f}  LR: {lr_now:.1e}")
        with open(metrics_csv, "a") as f:
            f.write(f"{epoch},{avg_D:.6f},{avg_G:.6f},{val_mse:.6f},{lr_now:.6e}\n")

        # Save checkpoints
        save_checkpoint(G.state_dict(), os.path.join(ckpt_dir, f"generator_epoch{epoch}.pt"))
        save_checkpoint(D.state_dict(), os.path.join(ckpt_dir, f"discriminator_epoch{epoch}.pt"))
        scheduler_G.step(val_mse)


if __name__ == "__main__":
    main()
