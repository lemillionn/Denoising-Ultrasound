import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

# Generates synthetic ultrasound bursts of variable cycle-length
# and splits into train/val/test sets saved as .npz files.

def generate_burst(f0, cycles, noise_level, fs):
    """
    Generate a single Gaussian-windowed burst.
    Returns clean, noisy, duration_us
    """
    # Derive Gaussian envelope sigma so that total burst covers cycles@f0
    # total_duration = cycles / f0 = 6 * sigma_env
    sigma_env = (cycles / f0) / 6.0
    dur_s = cycles / f0
    n_samples = int(dur_s * fs)
    t = np.arange(n_samples) / fs

    # Synthesize clean burst
    env = np.exp(-((t - dur_s/2)**2) / (2 * sigma_env**2))
    burst = np.sin(2 * np.pi * f0 * t) * env

    # Add Gaussian noise
    noisy = burst + np.random.normal(0, noise_level, size=burst.shape)

    dur_us = dur_s * 1e6
    return burst.astype(np.float32), noisy.astype(np.float32), dur_us


def make_dataset(config_path="configs/default.yaml"):
    # Load configuration
    cfg = yaml.safe_load(open(config_path))
    N = int(cfg.get("dataset_size", 10000))
    fs = float(cfg.get("fs", 20e6))        # sampling rate (Hz)

    # Burst parameter ranges
    f0_low, f0_high         = cfg.get("f0_range", [1e6, 20e6])
    cycles_low, cycles_high = cfg.get("cycles_range", [1, 5])
    noise_low, noise_high   = cfg.get("noise_level_range", [0.01, 0.2])

    # Reproducibility
    seed = int(cfg.get("seed", 42))
    np.random.seed(seed)

    clean_list, noisy_list = [], []

    for _ in range(N):
        # Select random center frequency and cycles
        f0 = float(np.random.uniform(f0_low, f0_high))
        cycles = int(np.random.randint(cycles_low, cycles_high + 1))
        noise_level = float(np.random.uniform(noise_low, noise_high))

        clean, noisy, _ = generate_burst(f0, cycles, noise_level, fs)
        clean_list.append(clean)
        noisy_list.append(noisy)

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

    splits = {
        "train_clean": c_train,
        "train_noisy": n_train,
        "val_clean":   c_val,
        "val_noisy":   n_val,
        "test_clean":  c_test,
        "test_noisy":  n_test
    }

    # Save as .npz archives
    print("Saving synthetic bursts:")
    for key, arr_list in splits.items():
        path = cfg.get(key)
        if path is None:
            raise ValueError(f"Missing config key: {key}")
        base, ext = os.path.splitext(path)
        if ext.lower() != ".npz":
            path = base + ".npz"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, *arr_list)
        print(f"  Wrote {len(arr_list)} bursts to {path}")

    print(f"Done: {len(c_train)} train, {len(c_val)} val, {len(c_test)} test bursts.")

if __name__ == "__main__":
    make_dataset()
