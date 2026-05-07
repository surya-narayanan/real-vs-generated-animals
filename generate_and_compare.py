"""
Sample 10 real animal images, generate similar images via fal.ai FLUX img2img,
then plot RGB histograms comparing real vs generated.
"""

import random
import os
import glob
import numpy as np
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
import fal_client
import requests
from io import BytesIO

# ---- Matplotlib rcParams ----
mpl.rcParams.update({
    "figure.dpi": 300,
    "figure.figsize": (3.5, 2.6),
    "figure.facecolor": "white",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "legend.frameon": False,
    "legend.handlelength": 1.5,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# ---- Config ----
DATA_DIR = "/home/paperspace/animals_dataset_full/train"
OUTPUT_DIR = "/home/paperspace/gen_comparison"
N_SAMPLE = 500
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "real"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "generated"), exist_ok=True)

# ---- Collect all image paths ----
all_images = sorted(glob.glob(os.path.join(DATA_DIR, "*.jpg")))
all_images += sorted(glob.glob(os.path.join(DATA_DIR, "*.png")))
all_images += sorted(glob.glob(os.path.join(DATA_DIR, "*.jpeg")))
print(f"Found {len(all_images)} images total")


def extract_animal(filepath):
    """Extract animal name from filename like train_00001_label12_cockroach.jpg"""
    base = os.path.splitext(os.path.basename(filepath))[0]
    # Animal name is everything after the last '_labelNN_'
    parts = base.split("_")
    # Find the label part and take everything after it
    for j, part in enumerate(parts):
        if part.startswith("label"):
            return "_".join(parts[j + 1:])
    return "animal"


# ---- Sample N ----
sampled = random.sample(all_images, N_SAMPLE)
print(f"Sampled {N_SAMPLE} images")
for p in sampled[:10]:
    print(f"  e.g. {extract_animal(p)}: {os.path.basename(p)}")
print(f"  ... and {N_SAMPLE - 10} more")

# ---- Generate fully synthetic images via fal.ai text-to-image ----
real_arrays = []
gen_arrays = []

import time
t_start = time.time()

for i, img_path in enumerate(sampled):
    animal = extract_animal(img_path)
    fname = os.path.basename(img_path)
    if i % 50 == 0 or i < 5:
        elapsed = time.time() - t_start
        rate = i / elapsed if elapsed > 0 and i > 0 else 0
        eta = (N_SAMPLE - i) / rate / 60 if rate > 0 else 0
        print(f"[{i+1}/{N_SAMPLE}] Generating for {animal}/{fname}... "
              f"({elapsed/60:.1f}min elapsed, ~{eta:.0f}min remaining)")

    # Load real image and scale preserving aspect ratio (longest side = 512)
    real_img = Image.open(img_path).convert("RGB")
    w, h = real_img.size
    max_side = 512
    scale = max_side / max(w, h)
    new_w = round(w * scale / 8) * 8
    new_h = round(h * scale / 8) * 8
    real_img_resized = real_img.resize((new_w, new_h))
    print(f"  Real image: {w}x{h} -> {new_w}x{new_h}")

    # Build prompt — purely text-driven, no image input
    prompt = (
        f"A real photograph of a {animal} in its true natural habitat as it would be "
        f"found in the real world. Wildlife photography, realistic, unedited, natural "
        f"lighting, sharp focus, taken with a DSLR camera"
    )

    # Generate from noise via FLUX text-to-image (no image input)
    result = fal_client.subscribe(
        "fal-ai/flux/dev",
        arguments={
            "prompt": prompt,
            "num_inference_steps": 28,
            "image_size": {"width": new_w, "height": new_h},
        },
    )

    # Download generated image
    gen_url = result["images"][0]["url"]
    gen_response = requests.get(gen_url)
    gen_img = Image.open(BytesIO(gen_response.content)).convert("RGB")
    gen_img_resized = gen_img.resize((new_w, new_h))

    # Save copies
    real_img_resized.save(os.path.join(OUTPUT_DIR, "real", f"{i:03d}_{animal}_{fname}"))
    gen_img_resized.save(os.path.join(OUTPUT_DIR, "generated", f"{i:03d}_{animal}_{fname}"))

    real_arrays.append(np.array(real_img_resized))
    gen_arrays.append(np.array(gen_img_resized))
    print(f"  Done.")

print("Generation complete.")

# ---- Compute and plot RGB histograms ----
# Concatenate pixels from variable-sized images
real_pixels = np.concatenate([arr.reshape(-1, 3) for arr in real_arrays], axis=0)
gen_pixels = np.concatenate([arr.reshape(-1, 3) for arr in gen_arrays], axis=0)

channel_names = ["Red", "Green", "Blue"]
channel_colors = ["#d62728", "#2ca02c", "#1f77b4"]
bins = np.arange(0, 257, 2)

fig, axes = plt.subplots(1, 3, figsize=(7, 2.6))

for c, (ax, name, color) in enumerate(zip(axes, channel_names, channel_colors)):
    real_vals = real_pixels[:, c]
    gen_vals = gen_pixels[:, c]

    ax.hist(real_vals, bins=bins, density=True, alpha=0.5, color=color, label="Real")
    ax.hist(gen_vals, bins=bins, density=True, alpha=0.5, color=color,
            edgecolor="black", linewidth=0.3, linestyle="--",
            histtype="step", label="Generated")

    ax.set_xlabel("Pixel intensity")
    ax.set_title(name)
    if c == 0:
        ax.set_ylabel("Density")
    ax.legend()

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "rgb_histogram_comparison.png"))
print(f"Histogram saved to {OUTPUT_DIR}/rgb_histogram_comparison.png")

# ---- Side-by-side grid (show first 10 pairs) ----
N_GRID = min(10, N_SAMPLE)
fig2, axes2 = plt.subplots(2, N_GRID, figsize=(N_GRID * 1.2, 2.8))
for i in range(N_GRID):
    axes2[0, i].imshow(real_arrays[i])
    axes2[0, i].axis("off")
    animal = extract_animal(sampled[i])
    axes2[0, i].set_title(animal, fontsize=5)

    axes2[1, i].imshow(gen_arrays[i])
    axes2[1, i].axis("off")

axes2[0, 0].set_ylabel("Real", fontsize=7)
axes2[1, 0].set_ylabel("Generated", fontsize=7)
plt.tight_layout()
fig2.savefig(os.path.join(OUTPUT_DIR, "side_by_side_grid.png"))
print(f"Grid saved to {OUTPUT_DIR}/side_by_side_grid.png")

# ---- Statistical tests: per-image mean intensity comparison ----
from scipy import stats

print("\n" + "=" * 60)
print("STATISTICAL TESTS: Real vs Generated pixel distributions")
print("=" * 60)

# Per-image mean intensity for each channel (paired samples, one per image)
for c, name in enumerate(channel_names):
    real_means = np.array([arr[:, :, c].mean() for arr in real_arrays])
    gen_means = np.array([arr[:, :, c].mean() for arr in gen_arrays])

    # Paired t-test (same image pairs)
    t_stat, t_pval = stats.ttest_rel(real_means, gen_means)
    # Wilcoxon signed-rank (non-parametric alternative)
    w_stat, w_pval = stats.wilcoxon(real_means, gen_means)

    print(f"\n{name} channel — per-image mean intensity:")
    print(f"  Real  means: {real_means.mean():.2f} +/- {real_means.std():.2f}")
    print(f"  Gen   means: {gen_means.mean():.2f} +/- {gen_means.std():.2f}")
    print(f"  Paired t-test:        t={t_stat:.3f}, p={t_pval:.4f}")
    print(f"  Wilcoxon signed-rank: W={w_stat:.1f}, p={w_pval:.4f}")

# Per-image standard deviation comparison
print(f"\n{'─' * 60}")
print("Per-image pixel std dev comparison:")
for c, name in enumerate(channel_names):
    real_stds = np.array([arr[:, :, c].std() for arr in real_arrays])
    gen_stds = np.array([arr[:, :, c].std() for arr in gen_arrays])
    t_stat, t_pval = stats.ttest_rel(real_stds, gen_stds)
    print(f"  {name}: Real std={real_stds.mean():.2f}, Gen std={gen_stds.mean():.2f}, "
          f"paired t p={t_pval:.4f}")

# Overall KS test on pooled pixel distributions (subsample to avoid inflated significance)
print(f"\n{'─' * 60}")
print("Two-sample KS test on pooled pixel distributions (subsampled to 50k pixels):")
rng = np.random.default_rng(SEED)
n_subsample = 50_000
for c, name in enumerate(channel_names):
    real_sub = rng.choice(real_pixels[:, c], size=n_subsample, replace=False)
    gen_sub = rng.choice(gen_pixels[:, c], size=n_subsample, replace=False)
    ks_stat, ks_pval = stats.ks_2samp(real_sub, gen_sub)
    print(f"  {name}: KS={ks_stat:.4f}, p={ks_pval:.4e}")

print(f"\n{'─' * 60}")
print("Note: paired t-test and Wilcoxon are the primary tests (N=10 paired images).")
print("KS test on pooled pixels is supplementary — large sample sizes can yield")
print("significant p-values even for trivially small effect sizes.")
print("=" * 60)
