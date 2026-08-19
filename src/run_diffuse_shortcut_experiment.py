"""
Third variant: a DIFFUSE shortcut instead of a spatially localized one.
Instead of a bright corner marker, we shift the overall background
brightness by a small, visually subtle amount correlated with the label
(brighter background -> more likely square). There is no single region
responsible for this cue -- it's smeared across the whole image.

This tests a sharper hypothesis than "small markers escape Grad-CAM":
Grad-CAM is fundamentally a SPATIAL attribution method. It can only ever
answer "which region mattered." If the shortcut a model learns is a
global statistic (mean brightness, overall contrast, color balance, a
frequency-domain artifact) rather than a localized region, Grad-CAM has
no meaningful "where" to point to, even in principle.
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

from model import build_model
from gradcam import compute_gradcam

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
IMG_SIZE = 64
BRIGHTNESS_SHIFT = 18.0   # subtle, roughly 7% of the 0-255 range
CORRELATION = 0.95


def make_shape_image(label, rng):
    img = Image.new("L", (IMG_SIZE, IMG_SIZE), color=0)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    cx = rng.integers(20, IMG_SIZE - 20)
    cy = rng.integers(20, IMG_SIZE - 20)
    r = rng.integers(7, 10)
    fill_val = rng.uniform(110, 170)
    if label == 0:
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=int(fill_val))
    else:
        pts = np.array([[-r, -r], [r, -r], [r, r], [-r, r]], dtype=float)
        theta = rng.uniform(-0.3, 0.3)
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        pts = pts @ rot.T
        pts[:, 0] += cx
        pts[:, 1] += cy
        draw.polygon([tuple(p) for p in pts], fill=int(fill_val))
    return np.array(img, dtype=np.float32)


def make_diffuse_dataset(n_per_class, seed, apply_shift):
    rng = np.random.default_rng(seed)
    images, labels, shifted = [], [], []
    for label in (0, 1):
        for _ in range(n_per_class):
            arr = make_shape_image(label, rng)
            noise = rng.normal(0, 35, size=arr.shape)
            arr = np.clip(arr + noise, 0, 255)

            has_shift = False
            if apply_shift:
                p = CORRELATION if label == 1 else (1 - CORRELATION)
                has_shift = rng.random() < p
                if has_shift:
                    arr = np.clip(arr + BRIGHTNESS_SHIFT, 0, 255)

            img_blur = Image.fromarray(arr.astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=1.0))
            arr = np.array(img_blur, dtype=np.float32) / 255.0
            images.append(arr)
            labels.append(label)
            shifted.append(has_shift)
    images = np.array(images, dtype=np.float32)[..., None]
    labels = np.array(labels, dtype=np.int32)
    shifted = np.array(shifted, dtype=bool)
    idx = rng.permutation(len(images))
    return images[idx], labels[idx], shifted[idx]


def make_diffuse_probe(n_per_cell, seed):
    rng = np.random.default_rng(seed)
    images, labels, shifted = [], [], []
    for label in (0, 1):
        for forced in (False, True):
            for _ in range(n_per_cell):
                arr = make_shape_image(label, rng)
                noise = rng.normal(0, 35, size=arr.shape)
                arr = np.clip(arr + noise, 0, 255)
                if forced:
                    arr = np.clip(arr + BRIGHTNESS_SHIFT, 0, 255)
                img_blur = Image.fromarray(arr.astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=1.0))
                arr = np.array(img_blur, dtype=np.float32) / 255.0
                images.append(arr)
                labels.append(label)
                shifted.append(forced)
    images = np.array(images, dtype=np.float32)[..., None]
    labels = np.array(labels, dtype=np.int32)
    shifted = np.array(shifted, dtype=bool)
    return images, labels, shifted


def main():
    np.random.seed(0)
    print("Training diffuse-shortcut model (global brightness cue, no localized region)...")
    x_train, y_train, _ = make_diffuse_dataset(1500, seed=4, apply_shift=True)
    x_test, y_test, s_test = make_diffuse_dataset(300, seed=1004, apply_shift=True)

    model = build_model(IMG_SIZE)
    model.fit(x_train, y_train, validation_split=0.15, epochs=12, batch_size=32, verbose=0)
    _, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {acc:.4f}")

    x_probe, y_probe, s_probe = make_diffuse_probe(200, seed=99)
    preds = model.predict(x_probe, verbose=0)[:, 0]
    pred_label = (preds > 0.5).astype(int)
    probe_results = {}
    for shape_label, shape_name in [(0, "circle"), (1, "square")]:
        for shift_val, shift_name in [(False, "no_shift"), (True, "brightness_shift")]:
            mask = (y_probe == shape_label) & (s_probe == shift_val)
            probe_results[f"{shape_name}_{shift_name}"] = float(pred_label[mask].mean())
    print(json.dumps(probe_results, indent=2))

    idx = np.random.choice(len(x_test), 4, replace=False)
    fig, axes = plt.subplots(2, 4, figsize=(9.6, 5))
    for col, i in enumerate(idx):
        img = x_test[i]
        heatmap = compute_gradcam(model, img, layer_name="conv3")
        label = "square" if y_test[i] == 1 else "circle"
        shift_txt = " (brighter bg)" if s_test[i] else ""
        axes[0, col].imshow(img[..., 0], cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"{label}{shift_txt}", fontsize=9)
        axes[0, col].axis("off")
        axes[1, col].imshow(img[..., 0], cmap="gray", vmin=0, vmax=1)
        axes[1, col].imshow(heatmap, cmap="jet", alpha=0.5)
        axes[1, col].axis("off")
    fig.suptitle("Diffuse shortcut (global brightness): Grad-CAM has no 'where' to point to", fontsize=11)
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig7_gradcam_diffuse_shortcut.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    summary = {"diffuse_shortcut_test_accuracy": acc, "diffuse_shortcut_probe_reliance": probe_results,
               "brightness_shift": BRIGHTNESS_SHIFT}
    with open(os.path.join(FIG_DIR, "..", "results_summary_diffuse.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
