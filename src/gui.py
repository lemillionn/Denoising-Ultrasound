import os
import yaml
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import torch
import matplotlib.pyplot as plt

from models import Generator

class DenoiseGUI(tk.Tk):
    def __init__(self, config_path="configs/default.yaml"):
        super().__init__()
        self.title("🔊 Ultrasound Burst Denoiser")
        self.configure(bg="#efefef")
        self.resizable(False, False)

        # Load configuration
        cfg = yaml.safe_load(open(config_path))
        self.ckpt_dir = cfg["checkpoint_dir"]
        self.base_ch = int(cfg["base_channels"])
        self.depths = tuple(cfg["depths"])
        self.fs = float(cfg.get("fs", 100e3))

        # Global norm stats
        noisy_all = np.load(cfg["train_noisy"]).astype(np.float32)
        self.global_mu, self.global_sig = noisy_all.mean(), noisy_all.std()

        # Load metrics
        metrics_csv = os.path.join(self.ckpt_dir, "metrics.csv")
        self.metrics = {}
        if os.path.exists(metrics_csv):
            with open(metrics_csv) as f:
                next(f)
                for line in f:
                    ep, *_ , vm = line.strip().split(",")
                    self.metrics[int(ep)] = float(vm)
        if not self.metrics:
            messagebox.showerror("Error", "No training metrics found.")
            self.destroy()
            return
        self.best_epoch = min(self.metrics, key=self.metrics.get)

        self._build_ui()

    def _build_ui(self):
        frm = ttk.Frame(self, padding=12)
        frm.pack()

        # Epoch selector
        ttk.Label(frm, text="Epoch:").grid(row=0, column=0, sticky="w")
        self.epoch_var = tk.IntVar(value=self.best_epoch)
        cb = ttk.Combobox(
            frm,
            textvariable=self.epoch_var,
            values=sorted(self.metrics.keys()),
            state="readonly",
            width=6
        )
        cb.grid(row=0, column=1, sticky="w", padx=(0, 10))

        # Burst parameters
        ttk.Label(frm, text="f0 (Hz):").grid(row=1, column=0, sticky="w", pady=4)
        self.f0_var = tk.IntVar(value=5000)
        sb_f0 = tk.Spinbox(
            frm,
            from_=3000,
            to=10000,
            increment=500,
            textvariable=self.f0_var,
            width=8
        )
        sb_f0.grid(row=1, column=1)

        ttk.Label(frm, text="σ_env (ms):").grid(row=2, column=0, sticky="w", pady=4)
        self.senv_var = tk.DoubleVar(value=0.2)
        sb_senv = tk.Spinbox(
            frm,
            from_=0.1,
            to=0.3,
            increment=0.05,
            textvariable=self.senv_var,
            width=8
        )
        sb_senv.grid(row=2, column=1)

        ttk.Label(frm, text="noise σ:").grid(row=3, column=0, sticky="w", pady=4)
        self.noise_var = tk.DoubleVar(value=0.5)
        sb_noise = tk.Spinbox(
            frm,
            from_=0.2,
            to=1.0,
            increment=0.1,
            textvariable=self.noise_var,
            width=8
        )
        sb_noise.grid(row=3, column=1)

        ttk.Label(frm, text="duration (ms):").grid(row=4, column=0, sticky="w", pady=4)
        self.dur_var = tk.DoubleVar(value=2.0)
        sb_dur = tk.Spinbox(
            frm,
            from_=1.0,
            to=5.0,
            increment=0.5,
            textvariable=self.dur_var,
            width=8
        )
        sb_dur.grid(row=4, column=1)

        # Buttons
        btns = ttk.Frame(frm)
        btns.grid(row=5, column=0, columnspan=2, pady=10)
        ttk.Button(btns, text="🔍 Preview", command=self._preview).grid(row=0, column=0, padx=6)
        ttk.Button(btns, text="🤖 Denoise", command=self._denoise).grid(row=0, column=1, padx=6)

    def _generate_burst(self):
        f0 = self.f0_var.get()
        senv = self.senv_var.get() / 1000.0
        noise_level = self.noise_var.get()
        duration = self.dur_var.get() / 1000.0

        n_samples = int(duration * self.fs)
        t = np.arange(n_samples) / self.fs
        env = np.exp(-((t - duration/2)**2) / (2 * senv**2))
        clean = np.sin(2 * np.pi * f0 * t) * env
        noisy = clean + np.random.normal(0, noise_level, size=clean.shape)
        return clean.astype(np.float32), noisy.astype(np.float32)

    def _preview(self):
        clean, noisy = self._generate_burst()
        cn = (clean - self.global_mu) / self.global_sig
        nn = (noisy - self.global_mu) / self.global_sig
        plt.figure(figsize=(6,3))
        plt.plot(cn, label='Clean (norm)', linewidth=2)
        plt.plot(nn, label='Noisy (norm)', alpha=0.7)
        plt.legend(loc='upper right')
        plt.title('Burst Preview (normalized)')
        plt.tight_layout()
        plt.show()

    def _denoise(self):
        clean, noisy = self._generate_burst()
        nn = (noisy - self.global_mu) / self.global_sig
        inp = torch.from_numpy(nn)[None, None, ...].float()

        ep = self.epoch_var.get()
        ckpt = os.path.join(self.ckpt_dir, f"generator_epoch{ep}.pt")
        G = Generator(base_channels=self.base_ch, depths=self.depths)
        G.load_state_dict(torch.load(ckpt, map_location='cpu'))
        G.eval()

        with torch.no_grad():
            out_norm = G(inp).squeeze().cpu().numpy()

        denoised = out_norm * self.global_sig + self.global_mu
        plt.figure(figsize=(6,3))
        plt.plot(clean, '--', label='Clean', linewidth=2)
        plt.plot(noisy, label='Noisy', alpha=0.7)
        plt.plot(denoised, label='Denoised', linewidth=2)
        plt.legend(loc='upper right')
        plt.title(f'Denoised Burst (Epoch {self.epoch_var.get()})')
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    DenoiseGUI().mainloop()
