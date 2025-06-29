import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

# Generates synthetic ultrasound bursts of variable length
# and splits into train/val/test sets saved as .npz files.

def make_dataset(config_path="configs/default.yaml"):
    # Load configuration
    cfg = yaml.safe_load(open(config_path))
    N = int(cfg["dataset_size"])
    fs = float(cfg.get("fs", 100e3))  # sampling rate

    # Duration range (s)
    dur_min = float(cfg.get("duration_min", 0.001))
    dur_max = float(cfg.get("duration_max", 0.005))

    # Burst parameter ranges
    f0_low, f0_high       = cfg.get("f0_range", [3e3, 10e3])
    sigma_low, sigma_high = cfg.get("sigma_range", [1e-4, 3e-4])
    noise_low, noise_high = cfg.get("noise_level_range", [0.2, 1.0])

    # Reproducibility
    seed = int(cfg.get("seed", 42))
    np.random.seed(seed)

    # Generate synthetic bursts
    clean_list, noisy_list = [], []
    for _ in range(N):
        dur = np.random.uniform(dur_min, dur_max)
        n_samples = int(dur * fs)
        t = np.arange(n_samples) / fs

        # Envelope + tone
        f0 = np.random.uniform(f0_low, f0_high)
        sigma_env = np.random.uniform(sigma_low, sigma_high)
        env = np.exp(-((t - dur/2)**2) / (2 * sigma_env**2))
        burst = np.sin(2 * np.pi * f0 * t) * env

        # Add noise
        noise_level = np.random.uniform(noise_low, noise_high)
        noisy = burst + np.random.normal(0, noise_level, size=burst.shape)

        clean_list.append(burst.astype(np.float32))
        noisy_list.append(noisy.astype(np.float32))

    # Split into train/val/test
    val_frac = float(cfg.get("val_split", 0.1))
    test_frac = float(cfg.get("test_split", 0.1))
    c_rem, c_test, n_rem, n_test = train_test_split(
        clean_list, noisy_list,
        test_size=test_frac,
        random_state=seed
    )
    val_size = val_frac / (1 - test_frac)
    c_train, c_val, n_train, n_val = train_test_split(
        c_rem, n_rem,
        test_size=val_size,
        random_state=seed
    )

    # Map config keys to data lists
    splits = {
        "train_clean": c_train,
        "train_noisy": n_train,
        "val_clean":   c_val,
        "val_noisy":   n_val,
        "test_clean":  c_test,
        "test_noisy":  n_test
    }

    print(f"Saving bursts as .npz archives:")
    for key, arr_list in splits.items():
        path = cfg.get(key)
        if path is None:
            raise ValueError(f"Missing config key: {key}")
        # Ensure .npz extension
        base, ext = os.path.splitext(path)
        if ext.lower() != ".npz":
            path = base + ".npz"
        # Ensure output directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Save archive
        np.savez(path, *arr_list)
        print(f"  Wrote {len(arr_list)} bursts to {path}")

    print(f"Done: {len(c_train)} train, {len(c_val)} val, {len(c_test)} test bursts.")

if __name__ == "__main__":
    make_dataset()
