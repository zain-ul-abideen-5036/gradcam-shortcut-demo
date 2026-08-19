"""
Grad-CAM implemented from scratch with tf.GradientTape (no tf-keras-vis
black box). This is deliberate: the point of the article is to understand
exactly what the method computes, so we build it ourselves.

Grad-CAM, in plain terms:
1. Run the image through the network, keeping the activations of a chosen
   convolutional layer (feature maps) and the final class score.
2. Take the gradient of the class score with respect to those feature maps.
   This gradient says: "if this feature map's activation went up a bit,
   how much would the class score go up?" -> importance of each channel.
3. Global-average-pool the gradient over space to get one importance
   weight per channel.
4. Form a weighted sum of the feature maps using those weights, then ReLU
   it (we only care about evidence FOR the class, not against it).
5. Upsample that low-resolution map back to the input image size.

The result is a coarse heatmap over the *last conv layer's* spatial grid
(here 16x16 for a 64x64 input), not a pixel-exact map. That coarseness is
exactly the limitation the article is about.
"""
import numpy as np
import tensorflow as tf


def compute_gradcam(model, image, layer_name="conv3"):
    """image: (H, W, 1) float32 in [0,1]. Returns heatmap resized to (H, W),
    normalized to [0, 1]."""
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output],
    )

    img_batch = image[None, ...]
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_batch)
        # binary sigmoid output -> class score for the predicted class
        class_score = predictions[:, 0]

    grads = tape.gradient(class_score, conv_output)  # (1, h, w, c)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (c,)

    conv_output = conv_output[0]  # (h, w, c)
    heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1)
    heatmap = tf.nn.relu(heatmap)

    heatmap = heatmap.numpy()
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    heatmap_img = tf.image.resize(
        heatmap[..., None], image.shape[:2], method="bilinear"
    ).numpy()[..., 0]
    return heatmap_img
