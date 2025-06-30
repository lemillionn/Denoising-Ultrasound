import os
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from models import Generator, Discriminator
from generate_synthetic import generate_burst
from dataset import collate_fn


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SyntheticDataset(torch.utils.data.Dataset):
    """On-the-fly Gaussian-windowed ultrasound bursts."""
    def __init__(self, N, f0_range, cycles_range, noise_range, fs):
        self.N = int(N)
        self.f0_low, self.f0_high         = float(f0_range[0]), float(f0_range[1])
        self.cycles_low, self.cycles_high = int(cycles_range[0]), int(cycles_range[1])
        self.noise_low, self.noise_high   = float(noise_range[0]), float(noise_range[1])
        self.fs = float(fs)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        f0     = np.random.uniform(self.f0_low, self.f0_high)
        cycles = np.random.randint(self.cycles_low, self.cycles_high + 1)
        noise  = np.random.uniform(self.noise_low, self.noise_high)

        clean, noisy, _ = generate_burst(f0, cycles, noise, self.fs)

        # per-sample normalization with guard
        noisy_mean, noisy_std = noisy.mean(), noisy.std()
        clean_mean, clean_std = clean.mean(), clean.std()
        if noisy_std < 1e-6: noisy_std = 1.0
        if clean_std < 1e-6: clean_std = 1.0
        noisy = (noisy - noisy_mean) / noisy_std
        clean = (clean - clean_mean) / clean_std

        x = torch.from_numpy(noisy).unsqueeze(0)
        y = torch.from_numpy(clean).unsqueeze(0)
        return x, y


def train(config_path="configs/default.yaml"):
    cfg = yaml.safe_load(open(config_path))
    set_seed(cfg.get("seed", 42))
    fs = float(cfg.get("fs", 20e6))

    # Prepare checkpoint and CSV
    ckpt_dir = cfg.get("checkpoint_dir", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    csv_path = os.path.join(ckpt_dir, "metrics.csv")
    with open(csv_path, "w") as f:
        f.write("epoch,train_D,train_G,val\n")

    # Datasets
    train_ds = SyntheticDataset(
        N=int(cfg.get("dataset_size", 10000)),
        f0_range=cfg.get("f0_range", [1e6, 20e6]),
        cycles_range=cfg.get("cycles_range", [1, 5]),
        noise_range=cfg.get("noise_level_range", [0.01, 0.2]),
        fs=fs
    )
    val_ds = SyntheticDataset(
        N=int(cfg.get("val_size", 1000)),
        f0_range=cfg.get("f0_range", [1e6, 20e6]),
        cycles_range=cfg.get("cycles_range", [1, 5]),
        noise_range=cfg.get("noise_level_range", [0.01, 0.2]),
        fs=fs
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.get("batch_size", 16)),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 4)),
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.get("batch_size", 16)),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 4)),
        collate_fn=collate_fn
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = Generator(
        base_channels=int(cfg.get("base_channels", 64)),
        depths=tuple(cfg.get("depths", [2, 2, 2]))
    ).to(device)
    D = Discriminator(
        in_channels=1,
        base_channels=int(cfg.get("base_channels", 64))
    ).to(device)

    opt_G = optim.Adam(G.parameters(), lr=float(cfg.get("lr_G", 1e-4)))
    opt_D = optim.Adam(D.parameters(), lr=float(cfg.get("lr_D", 1e-4)))
    criterion = nn.MSELoss()
    num_epochs = int(cfg.get("epochs", 100))

    for epoch in range(1, num_epochs + 1):
        G.train(); D.train()
        epoch_loss_D = 0.0
        epoch_loss_G = 0.0

        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)

            # Train Discriminator
            opt_D.zero_grad()
            fake = G(noisy)
            loss_D = criterion(D(clean), torch.ones_like(D(clean))) + \
                     criterion(D(fake.detach()), torch.zeros_like(D(fake)))
            loss_D.backward()
            opt_D.step()
            epoch_loss_D += loss_D.item()

            # Train Generator
            opt_G.zero_grad()
            gan_lambda = float(cfg.get("gan_lambda", 1.0))
            loss_recon = criterion(fake, clean)
            target_real = torch.ones_like(D(fake), device=device)
            loss_adv   = criterion(D(fake), target_real)
            loss_G     = loss_recon + gan_lambda * loss_adv
            loss_G.backward()
            opt_G.step()
            epoch_loss_G += loss_G.item()

        # Validation
        G.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                pred = G(noisy)
                val_loss += criterion(pred, clean).item()
        val_loss /= len(val_loader)

        train_D_avg = epoch_loss_D / len(train_loader)
        train_G_avg = epoch_loss_G / len(train_loader)
        print(f"Epoch {epoch}/{num_epochs}, Train D={train_D_avg:.6f}, Train G={train_G_avg:.6f}, Val={val_loss:.6f}")

        with open(csv_path, "a") as f:
            f.write(f"{epoch},{train_D_avg:.6f},{train_G_avg:.6f},{val_loss:.6f}\n")

        # Save checkpoints
        torch.save(G.state_dict(), os.path.join(ckpt_dir, f"generator_epoch{epoch}.pt"))
        torch.save(D.state_dict(), os.path.join(ckpt_dir, f"discriminator_epoch{epoch}.pt"))


if __name__ == "__main__":
    train()
