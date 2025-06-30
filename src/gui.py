# gradio_app.py

import os, glob
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt

import gradio as gr
from models import Generator

# —————————————————————————————————————————————————————————————————————
# Load configuration & global stats

cfg = yaml.safe_load(open("configs/default.yaml"))
CKPT_DIR   = cfg["checkpoint_dir"]
BASE_CH    = int(cfg["base_channels"])
DEPTHS     = tuple(cfg["depths"])

# Increase FS to a realistic ultrasound sampling rate
FS         = 200e6   # 20 MHz

# Load global normalization stats for denoising
data       = np.load(cfg["train_noisy"])
noisy_all  = data["arr_0"].astype(np.float32)
GLOBAL_MU, GLOBAL_SIG = noisy_all.mean(), noisy_all.std()

# Load metrics to know available epochs
metrics_csv = os.path.join(CKPT_DIR, "metrics.csv")
metrics     = {}
with open(metrics_csv) as f:
    next(f)
    for line in f:
        ep, *_ , vm = line.strip().split(",")
        metrics[int(ep)] = float(vm)
EPOCHS = sorted(metrics.keys())

# —————————————————————————————————————————————————————————————————————
# Burst generator (returns duration in μs)

def generate_burst(f0, cycles, noise_level):
    # σ_env tied to #cycles at f0
    senv_s = (cycles / f0) / 6.0       # seconds
    dur_s   = 6 * senv_s               # total span ±3σ_env
    n_samps = int(dur_s * FS)
    t       = np.arange(n_samps) / FS

    env     = np.exp(-((t - dur_s/2)**2) / (2 * senv_s**2))
    clean   = np.sin(2*np.pi*f0*t) * env
    noisy   = clean + np.random.normal(0, noise_level, size=clean.shape)

    # return duration in microseconds
    dur_us  = dur_s * 1e6

    return clean.astype(np.float32), noisy.astype(np.float32), dur_us

# —————————————————————————————————————————————————————————————————————
# Preview plot (raw signals with μs annotation)

def preview_plot(f0, cycles, noise):
    # 1) Generate sampled burst
    clean, noisy, dur_us = generate_burst(f0, cycles, noise)

    # 2) Time axes (µs)
    n_samps = clean.size
    t_samp  = np.linspace(0, dur_us, n_samps)
    t_fine  = np.linspace(0, dur_us, 500)

    # 3) Ideal burst on fine grid
    total_s = dur_us * 1e-6
    senv_s  = (cycles / f0) / 6.0
    env_f   = np.exp(-((t_fine*1e-6 - total_s/2)**2) / (2 * senv_s**2))
    clean_f = np.sin(2*np.pi*f0*(t_fine*1e-6)) * env_f

    # 4) Plot everything as lines
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(t_fine, clean_f,   "--", label="Clean (ideal)")
    ax.plot(t_samp, clean,     "-",  label="Clean (sampled)", linewidth=1)
    ax.plot(t_samp, noisy,     "-",  alpha=0.5, label="Noisy (sampled)", linewidth=1)

    ax.legend()
    ax.set_title(f"Preview — {cycles} cycles at {f0/1e6:.1f} MHz")
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Amplitude")
    plt.tight_layout()
    return fig
# —————————————————————————————————————————————————————————————————————
# Denoise plot

def denoise_plot(epoch, f0, cycles, noise):
    clean, noisy, dur_us = generate_burst(f0, cycles, noise)

    # normalize with training stats
    nn  = (noisy - GLOBAL_MU) / GLOBAL_SIG
    inp = torch.from_numpy(nn)[None,None,...].float()

    # load and run generator
    ckpt = os.path.join(CKPT_DIR, f"generator_epoch{epoch}.pt")
    G    = Generator(base_channels=BASE_CH, depths=DEPTHS)
    G.load_state_dict(torch.load(ckpt, map_location="cpu"))
    G.eval()
    with torch.no_grad():
        out_norm = G(inp).squeeze().cpu().numpy()
    denoised = out_norm * GLOBAL_SIG + GLOBAL_MU

    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(clean, "--", label="Clean", linewidth=2)
    ax.plot(noisy, label="Noisy", alpha=0.7)
    ax.plot(denoised, label="Denoised", linewidth=2)
    ax.legend(loc="upper right")
    ax.set_title(
        f"Denoised — Ep {epoch}, f₀={f0/1e6:.1f} MHz, {cycles} cycles"
    )
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Amplitude")
    plt.tight_layout()
    return fig

# —————————————————————————————————————————————————————————————————————
# Build Gradio GUI

with gr.Blocks(title="🔊 Ultrasound Burst Denoiser") as demo:
    gr.Markdown("## Ultrasound Burst Denoiser\nPick realistic ultrasound pulse parameters below.")

    with gr.Row():
        epoch_sl  = gr.Slider(
            minimum=min(EPOCHS), maximum=max(EPOCHS),
            step=1, value=EPOCHS[0], label="Epoch"
        )
        f0_sl     = gr.Slider(
            1e6, 20e6, step=0.5e6, value=7.5e6,
            label="Center freq (Hz)"
        )

    with gr.Row():
        cycles_sl = gr.Slider(
            1, 5, step=1, value=2,
            label="Pulse length (cycles)"
        )
        noise_sl  = gr.Slider(
            0.0, 0.2, step=0.02, value=0.05,
            label="Noise σ"
        )

    gr.Markdown("*Duration is computed as ±3σ_env → displayed in μs.*")

    with gr.Tabs():
        with gr.TabItem("🔍 Preview"):
            btn  = gr.Button("Generate Preview")
            out  = gr.Plot()
            btn.click(preview_plot, inputs=[f0_sl, cycles_sl, noise_sl], outputs=out)

        with gr.TabItem("🤖 Denoise"):
            btn2 = gr.Button("Run Denoiser")
            out2 = gr.Plot()
            btn2.click(denoise_plot, inputs=[epoch_sl, f0_sl, cycles_sl, noise_sl], outputs=out2)

demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
