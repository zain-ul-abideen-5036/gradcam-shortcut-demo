import json
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from dataset import make_dataset, make_probe_set, IMG_SIZE, MARKER_CORNER
from model import build_model
from gradcam import compute_gradcam

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

tf.random.set_seed(0)
np.random.seed(0)

N_PER_CLASS_TRAIN = 1500
N_PER_CLASS_TEST = 300


def train_model(inject_marker, seed):
    x_train, y_train, _ = make_dataset(N_PER_CLASS_TRAIN, seed=seed, inject_marker=inject_marker)
    x_test, y_test, m_test = make_dataset(N_PER_CLASS_TEST, seed=seed + 1000, inject_marker=inject_marker)

    model = build_model(IMG_SIZE)
    history = model.fit(
        x_train, y_train,
        validation_split=0.15,
        epochs=12,
        batch_size=32,
        verbose=0,
    )
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    return model, history, test_acc, (x_test, y_test, m_test)


def probe_reliance(model, seed):
    """Decouple shape from marker and see what actually drives predictions."""
    x, y, m = make_probe_set(n_per_cell=200, seed=seed)
    preds = model.predict(x, verbose=0)[:, 0]
    pred_label = (preds > 0.5).astype(int)

    results = {}
    for shape_label, shape_name in [(0, "circle"), (1, "square")]:
        for marker_val, marker_name in [(False, "no_marker"), (True, "marker")]:
            mask = (y == shape_label) & (m == marker_val)
            frac_predicted_square = pred_label[mask].mean()
            results[f"{shape_name}_{marker_name}"] = float(frac_predicted_square)
    return results


def save_examples_figure(x, y, m, path, n=6):
    fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 2.4))
    idx = np.random.choice(len(x), n, replace=False)
    for ax, i in zip(axes, idx):
        ax.imshow(x[i, ..., 0], cmap="gray", vmin=0, vmax=1)
        label = "square" if y[i] == 1 else "circle"
        marker_txt = " +marker" if m[i] else ""
        ax.set_title(f"{label}{marker_txt}", fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def save_gradcam_figure(model, x, y, m, path, layer_name, n=4, title_prefix=""):
    fig, axes = plt.subplots(2, n, figsize=(2.4 * n, 5))
    idx = np.random.choice(len(x), n, replace=False)
    for col, i in enumerate(idx):
        img = x[i]
        heatmap = compute_gradcam(model, img, layer_name=layer_name)
        label = "square" if y[i] == 1 else "circle"
        marker_txt = " +marker" if m[i] else ""

        axes[0, col].imshow(img[..., 0], cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"{label}{marker_txt}", fontsize=10)
        axes[0, col].axis("off")

        axes[1, col].imshow(img[..., 0], cmap="gray", vmin=0, vmax=1)
        axes[1, col].imshow(heatmap, cmap="jet", alpha=0.5)
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("input", fontsize=10)
    fig.suptitle(f"{title_prefix}Grad-CAM overlays (layer: {layer_name})", fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def save_accuracy_bar(clean_acc, shortcut_acc, path):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    bars = ax.bar(["Clean model", "Shortcut model"], [clean_acc, shortcut_acc],
                   color=["#2b8a3e", "#e8590c"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Test accuracy")
    ax.set_title("Both models look equally good by accuracy alone")
    for b, v in zip(bars, [clean_acc, shortcut_acc]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
                 ha="center", fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def save_reliance_bar(probe_results, path):
    labels = ["circle\nno_marker", "circle\nmarker", "square\nno_marker", "square\nmarker"]
    keys = ["circle_no_marker", "circle_marker", "square_no_marker", "square_marker"]
    values = [probe_results[k] for k in keys]

    fig, ax = plt.subplots(figsize=(6, 4.2))
    colors = ["#4dabf7", "#1864ab", "#ffa94d", "#d9480f"]
    ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Fraction predicted "square"')
    ax.set_title("Shortcut model's predictions, shape vs. marker decoupled")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    print("Training clean model (no shortcut)...")
    clean_model, clean_hist, clean_acc, clean_test = train_model(inject_marker=False, seed=1)
    print(f"Clean model test accuracy: {clean_acc:.4f}")

    print("Training shortcut model (marker injected, 97% correlated with label)...")
    shortcut_model, shortcut_hist, shortcut_acc, shortcut_test = train_model(inject_marker=True, seed=2)
    print(f"Shortcut model test accuracy: {shortcut_acc:.4f}")

    print("Running decoupled probe set on shortcut model...")
    probe_results = probe_reliance(shortcut_model, seed=99)
    print(json.dumps(probe_results, indent=2))

    # also probe the clean model on the same decoupled set, as a control
    # (it was never trained with a marker, so the marker should be irrelevant)
    probe_results_clean_model = probe_reliance(clean_model, seed=99)
    print("Clean model on probe set (control):")
    print(json.dumps(probe_results_clean_model, indent=2))

    x_clean_test, y_clean_test, m_clean_test = clean_test
    x_short_test, y_short_test, m_short_test = shortcut_test

    print("Saving figures...")
    save_examples_figure(x_short_test, y_short_test, m_short_test,
                          os.path.join(FIG_DIR, "fig0_example_images.png"))

    save_gradcam_figure(clean_model, x_clean_test, y_clean_test, m_clean_test,
                         os.path.join(FIG_DIR, "fig1_gradcam_clean_model.png"),
                         layer_name="conv3", title_prefix="Clean model - ")

    save_gradcam_figure(shortcut_model, x_short_test, y_short_test, m_short_test,
                         os.path.join(FIG_DIR, "fig2_gradcam_shortcut_model.png"),
                         layer_name="conv3", title_prefix="Shortcut model - ")

    save_accuracy_bar(clean_acc, shortcut_acc,
                       os.path.join(FIG_DIR, "fig3_accuracy_comparison.png"))

    save_reliance_bar(probe_results,
                       os.path.join(FIG_DIR, "fig4_probe_reliance_shortcut_model.png"))

    # also show shallower layer Grad-CAM to illustrate resolution effects
    save_gradcam_figure(shortcut_model, x_short_test, y_short_test, m_short_test,
                         os.path.join(FIG_DIR, "fig5_gradcam_shortcut_model_conv1.png"),
                         layer_name="conv1", title_prefix="Shortcut model, shallow layer - ")

    summary = {
        "clean_model_test_accuracy": clean_acc,
        "shortcut_model_test_accuracy": shortcut_acc,
        "shortcut_model_probe_reliance": probe_results,
        "clean_model_probe_reliance_control": probe_results_clean_model,
        "marker_corner_pixels": MARKER_CORNER,
        "img_size": IMG_SIZE,
        "n_per_class_train": N_PER_CLASS_TRAIN,
        "n_per_class_test": N_PER_CLASS_TEST,
    }
    with open(os.path.join(FIG_DIR, "..", "results_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Done. Figures written to", FIG_DIR)


if __name__ == "__main__":
    main()
