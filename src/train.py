import os
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from models import Generator, Discriminator
from src.generate_synthetic import generate_burst
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
        self.N = N
        self.f0_low, self.f0_high = f0_range
        self.cycles_low, self.cycles_high = cycles_range
        self.noise_low, self.noise_high = noise_range
        self.fs = fs

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        f0     = np.random.uniform(self.f0_low, self.f0_high)
        cycles = np.random.randint(self.cycles_low, self.cycles_high + 1)
        noise  = np.random.uniform(self.noise_low, self.noise_high)

        clean, noisy, _ = generate_burst(f0, cycles, noise)
        # normalize per-sample
        noisy = (noisy - noisy.mean()) / noisy.std()
        clean = (clean - clean.mean()) / clean.std()

        x = torch.from_numpy(noisy).unsqueeze(0)
        y = torch.from_numpy(clean).unsqueeze(0)
        return x, y


def train(config_path="configs/default.yaml"):
    # Load config
    cfg = yaml.safe_load(open(config_path))
    set_seed(cfg.get("seed", 42))

    # Build datasets
    train_ds = SyntheticDataset(
        N=int(cfg.get("dataset_size", 10000)),
        f0_range=cfg.get("f0_range", [1e6, 20e6]),
        cycles_range=cfg.get("cycles_range", [1, 5]),
        noise_range=cfg.get("noise_level_range", [0.01, 0.2]),
        fs=float(cfg.get("fs", 20e6))
    )
    val_ds = SyntheticDataset(
        N=int(cfg.get("val_size", 1000)),
        f0_range=cfg.get("f0_range", [1e6, 20e6]),
        cycles_range=cfg.get("cycles_range", [1, 5]),
        noise_range=cfg.get("noise_level_range", [0.01, 0.2]),
        fs=float(cfg.get("fs", 20e6))
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

    # Models
    G = Generator(
        base_channels=int(cfg.get("base_channels", 64)),
        depths=tuple(cfg.get("depths", [2,2,2]))
    ).to(device)
    D = Discriminator(
        base_channels=int(cfg.get("base_channels", 64)),
        depths=tuple(cfg.get("depths", [2,2,2]))
    ).to(device)

    # Optimizers
    opt_G = optim.Adam(G.parameters(), lr=float(cfg.get("lr_G", 1e-4)))
    opt_D = optim.Adam(D.parameters(), lr=float(cfg.get("lr_D", 1e-4)))

    # Loss
    criterion = nn.MSELoss()

    num_epochs = int(cfg.get("epochs", 100))

    for epoch in range(1, num_epochs+1):
        G.train()
        D.train()
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)

            # Train D
            opt_D.zero_grad()
            fake = G(noisy)
            loss_D = criterion(D(clean), torch.ones_like(D(clean))) + \
                     criterion(D(fake.detach()), torch.zeros_like(D(fake)))
            loss_D.backward()
            opt_D.step()

            # Train G
            opt_G.zero_grad()
            loss_G = criterion(fake, clean) + \
                     cfg.get("gan_lambda", 1.0) * criterion(D(fake), torch.ones_like(D(fake)))
            loss_G.backward()
            opt_G.step()

        # Validation
        G.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                pred = G(noisy)
                val_loss += criterion(pred, clean).item()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch}/{num_epochs}, Val Loss: {val_loss:.6f}")

        # Save checkpoints
        ckpt_dir = cfg.get("checkpoint_dir", "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(G.state_dict(), os.path.join(ckpt_dir, f"generator_epoch{epoch}.pt"))
        torch.save(D.state_dict(), os.path.join(ckpt_dir, f"discriminator_epoch{epoch}.pt"))


if __name__ == "__main__":
    train()
