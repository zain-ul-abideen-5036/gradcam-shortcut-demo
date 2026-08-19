"""
Follow-up experiment: does shrinking the shortcut marker make it disappear
from the Grad-CAM heatmap at conv3, even though the model still relies on
it 100%? conv3 has a 16x16 spatial grid for a 64x64 input, so each cell
covers a 4x4 pixel region (before accounting for the conv3 receptive
field, which is larger due to the two preceding conv+pool stages).
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt

from dataset import make_dataset, make_probe_set, IMG_SIZE
from model import build_model
from gradcam import compute_gradcam

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

TINY_MARKER_BOX = (2, 2, 4, 4)  # 2x2 px, vs. the original 6x6 px


def train_shortcut(marker_box, seed):
    x_train, y_train, _ = make_dataset(1500, seed=seed, inject_marker=True, marker_box=marker_box)
    x_test, y_test, m_test = make_dataset(300, seed=seed + 1000, inject_marker=True, marker_box=marker_box)
    model = build_model(IMG_SIZE)
    model.fit(x_train, y_train, validation_split=0.15, epochs=12, batch_size=32, verbose=0)
    _, acc = model.evaluate(x_test, y_test, verbose=0)
    return model, acc, (x_test, y_test, m_test)


def probe(model, marker_box, seed):
    x, y, m = make_probe_set(200, seed=seed, marker_box=marker_box)
    preds = model.predict(x, verbose=0)[:, 0]
    pred_label = (preds > 0.5).astype(int)
    out = {}
    for shape_label, shape_name in [(0, "circle"), (1, "square")]:
        for marker_val, marker_name in [(False, "no_marker"), (True, "marker")]:
            mask = (y == shape_label) & (m == marker_val)
            out[f"{shape_name}_{marker_name}"] = float(pred_label[mask].mean())
    return out


def main():
    print(f"Training shortcut model with tiny {TINY_MARKER_BOX} marker...")
    model, acc, test_data = train_shortcut(TINY_MARKER_BOX, seed=3)
    print(f"Test accuracy: {acc:.4f}")

    probe_results = probe(model, TINY_MARKER_BOX, seed=99)
    print("Probe (shape vs. marker decoupled):")
    print(json.dumps(probe_results, indent=2))

    x_test, y_test, m_test = test_data
    idx = np.random.choice(len(x_test), 4, replace=False)
    fig, axes = plt.subplots(2, 4, figsize=(9.6, 5))
    for col, i in enumerate(idx):
        img = x_test[i]
        heatmap = compute_gradcam(model, img, layer_name="conv3")
        label = "square" if y_test[i] == 1 else "circle"
        marker_txt = " +tiny marker" if m_test[i] else ""
        axes[0, col].imshow(img[..., 0], cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"{label}{marker_txt}", fontsize=9)
        axes[0, col].axis("off")
        axes[1, col].imshow(img[..., 0], cmap="gray", vmin=0, vmax=1)
        axes[1, col].imshow(heatmap, cmap="jet", alpha=0.5)
        axes[1, col].axis("off")
    fig.suptitle("Tiny (2x2 px) marker: still 100% reliance, Grad-CAM at conv3", fontsize=11)
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig6_gradcam_tiny_marker.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    summary = {"tiny_marker_test_accuracy": acc, "tiny_marker_probe_reliance": probe_results,
               "tiny_marker_box": TINY_MARKER_BOX}
    with open(os.path.join(FIG_DIR, "..", "results_summary_tiny_marker.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
