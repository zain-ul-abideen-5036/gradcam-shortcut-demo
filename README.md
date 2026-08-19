# gradcam-shortcut-demo

![banner](figures/banner.png)

Does Grad-CAM actually catch shortcut learning, or does it just look like it does?

This repo trains three small CNNs on a synthetic circle-vs-square task:
a clean model, a model with a spatially localized shortcut (a corner
marker correlated with the label), and a model with a diffuse shortcut
(a global background-brightness shift correlated with the label). It
runs a from-scratch Grad-CAM implementation on all three to see which
kinds of shortcuts the method can and can't detect.

Full write-up: [`article.md`](article.md)

## Headline result

All three models score within a few points of each other on test accuracy
(94.8% to 99.5%). A decoupled probe set (shape and shortcut varied
independently) shows two of the three models rely **100%** on their
shortcut and **0%** on actual shape, despite high accuracy.

Grad-CAM correctly flags the *localized* shortcut (even at 2×2 pixels).
It gives no warning at all for the *diffuse* shortcut: the heatmap looks
identical to a correctly reasoning model's.

| Model | Test accuracy | Actual signal used | Grad-CAM catches it? |
|---|---|---|---|
| Clean | 99.5% | Shape | ✅ correctly localizes shape |
| Localized shortcut (corner marker, 6×6px) | 99.2% | Marker only | ✅ correctly localizes marker |
| Localized shortcut (tiny marker, 2×2px) | 99.2% | Marker only | ✅ still catches it |
| Diffuse shortcut (global brightness) | 94.8% | Brightness only | ❌ heatmap misleadingly highlights shape |

## Why synthetic data

The dataset (`src/dataset.py`) generates circles and squares procedurally,
which means the ground truth discriminative signal is known exactly. That's
what makes it possible to say with certainty whether a heatmap is telling
the truth, rather than eyeballing plausibility on real images where the
"true" signal is itself uncertain.

## Structure

```
src/
  dataset.py                        synthetic image generator (clean / localized shortcut / probe sets)
  model.py                          shared small CNN architecture
  gradcam.py                        Grad-CAM implemented from scratch with tf.GradientTape
  run_experiment.py                 main experiment: clean vs. localized shortcut models + figures
  run_marker_size_experiment.py     follow-up: does a 2x2px marker escape detection? (no)
  run_diffuse_shortcut_experiment.py follow-up: global brightness shortcut with no localized region
  make_banner.py                    article banner image (auto-crops real heatmaps from fig1/fig7)
figures/                            all generated figures (committed for convenience)
results_summary*.json               numeric results from each experiment
article.md                          full write-up
```

## Reproducing

```bash
pip install -r requirements.txt
cd src
python run_experiment.py                    # clean model + localized shortcut model
python run_marker_size_experiment.py        # tiny marker follow-up
python run_diffuse_shortcut_experiment.py   # diffuse shortcut follow-up
```

Each script is self-contained, uses a fixed random seed, and writes its
figures to `../figures/` and a results JSON to the repo root.

## Method notes

- **Architecture**: 3 conv blocks (16, 32, 64 filters) with max pooling,
  global average pooling, one dense layer, sigmoid output. Identical
  architecture across all three experiments, so any behavioral difference
  comes from the data, not the model.
- **Grad-CAM**: implemented manually with `tf.GradientTape` (`src/gradcam.py`)
  rather than a library, targeting the last conv layer by default. A
  shallow layer (`conv1`) comparison is also included.
- **Probe sets**: for each shortcut variant, a held-out set with shape and
  shortcut presence assigned independently and orthogonally (not
  correlated as in training), used to measure true reliance without any
  saliency method.

## Limitations

This is a controlled, synthetic demonstration built to isolate one
variable at a time. It's meant to show *that* this failure mode exists
and characterize it precisely, not to quantify how often it occurs in any
particular real-world dataset. The specific numbers (2×2px marker still
detected, diffuse shift not detected) are properties of this task and
architecture, not universal thresholds.

## License

MIT
