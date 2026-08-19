"""
Synthetic shapes dataset for the Grad-CAM shortcut-learning demo.

Why synthetic instead of a downloaded dataset:
We need to know, with certainty, what the "true" discriminative signal is,
so we can later check whether Grad-CAM actually points at it. That's only
possible if we generate the images ourselves.

Task: binary classification, circle vs. square, rendered at random
position, size, and rotation on a noisy 64x64 canvas.

Two dataset variants are produced from the same generator:
  - "clean":     the only signal correlated with the label is shape identity.
  - "shortcut":  a small bright marker is stamped in a fixed corner region,
                 present on ~97% of squares and ~3% of circles. The marker
                 is a trivial shortcut a CNN can learn instead of shape.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

IMG_SIZE = 64
MARKER_CORNER = (2, 2, 8, 8)  # top-left box where the shortcut marker lives
MARKER_CORRELATION = 0.995     # P(marker | square) = P(no marker | circle) at train time

# The shape signal is deliberately made noisy/subtle (small, low-contrast,
# blurred) so it is *harder* to fit than the marker. This mirrors real
# shortcut learning: models gravitate to whichever predictive signal is
# easiest to fit, not necessarily the intended one. A crisp, deterministic
# 6x6 corner marker is trivially easier for a CNN to key off than a small
# blurry, low-contrast shape.


def _draw_shape(draw, label, cx, cy, r, rot_noise):
    """label: 0 = circle, 1 = square"""
    if label == 0:
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
    else:
        # slight rotation via polygon so squares aren't perfectly axis-aligned
        pts = np.array([[-r, -r], [r, -r], [r, r], [-r, r]], dtype=float)
        theta = rot_noise
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
        pts = pts @ rot.T
        pts[:, 0] += cx
        pts[:, 1] += cy
        draw.polygon([tuple(p) for p in pts], fill=255)


def _make_image(label, rng, inject_marker, force_marker=None, marker_box=None):
    img = Image.new("L", (IMG_SIZE, IMG_SIZE), color=0)
    draw = ImageDraw.Draw(img)

    cx = rng.integers(20, IMG_SIZE - 20)
    cy = rng.integers(20, IMG_SIZE - 20)
    r = rng.integers(7, 10)          # smaller shapes -> subtler signal
    rot = rng.uniform(-0.3, 0.3) if label == 1 else 0.0

    # low, randomized contrast: shape fill is well above background but
    # noisy, unlike the marker which is always pure white
    fill_val = rng.uniform(110, 170)
    draw = ImageDraw.Draw(img)
    if label == 0:
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=int(fill_val))
    else:
        pts = np.array([[-r, -r], [r, -r], [r, r], [-r, r]], dtype=float)
        theta = rot
        rotm = np.array([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta), np.cos(theta)]])
        pts = pts @ rotm.T
        pts[:, 0] += cx
        pts[:, 1] += cy
        draw.polygon([tuple(p) for p in pts], fill=int(fill_val))

    arr = np.array(img, dtype=np.float32)

    # heavier background speckle noise + blur, independent of label,
    # to make the shape genuinely harder to key off than the marker
    noise = rng.normal(0, 35, size=arr.shape)
    arr = np.clip(arr + noise, 0, 255)
    img_blur = Image.fromarray(arr.astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(radius=1.0)
    )
    arr = np.array(img_blur, dtype=np.float32)

    has_marker = False
    if inject_marker:
        if force_marker is not None:
            has_marker = force_marker
        else:
            p = MARKER_CORRELATION if label == 1 else (1 - MARKER_CORRELATION)
            has_marker = rng.random() < p
        if has_marker:
            x0, y0, x1, y1 = marker_box if marker_box is not None else MARKER_CORNER
            arr[y0:y1, x0:x1] = 255.0

    return arr / 255.0, has_marker


def make_dataset(n_per_class, seed, inject_marker, marker_box=None):
    rng = np.random.default_rng(seed)
    images, labels, markers = [], [], []
    for label in (0, 1):
        for _ in range(n_per_class):
            img, has_marker = _make_image(label, rng, inject_marker, marker_box=marker_box)
            images.append(img)
            labels.append(label)
            markers.append(has_marker)
    images = np.array(images, dtype=np.float32)[..., None]  # add channel dim
    labels = np.array(labels, dtype=np.int32)
    markers = np.array(markers, dtype=bool)

    idx = rng.permutation(len(images))
    return images[idx], labels[idx], markers[idx]


def make_probe_set(n_per_cell, seed, marker_box=None):
    """
    Diagnostic set that decouples shape from marker:
    circle+marker, circle+no-marker, square+marker, square+no-marker,
    in equal numbers. Used to prove the shortcut model relies on the
    marker rather than shape (or vice versa), independent of Grad-CAM.
    """
    rng = np.random.default_rng(seed)
    images, labels, markers = [], [], []
    for label in (0, 1):
        for forced in (False, True):
            for _ in range(n_per_cell):
                img, has_marker = _make_image(label, rng, inject_marker=True,
                                               force_marker=forced, marker_box=marker_box)
                images.append(img)
                labels.append(label)
                markers.append(has_marker)
    images = np.array(images, dtype=np.float32)[..., None]
    labels = np.array(labels, dtype=np.int32)
    markers = np.array(markers, dtype=bool)
    return images, labels, markers


if __name__ == "__main__":
    x, y, m = make_dataset(5, seed=0, inject_marker=True)
    print(x.shape, y.shape, m.sum(), "markers out of", len(m))
