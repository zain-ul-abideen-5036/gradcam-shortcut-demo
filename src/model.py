"""Small CNN for the circle-vs-square task. Same architecture is reused
for both the clean model and the shortcut model, so any behavioral
difference between them comes from the data, not the architecture."""
import tensorflow as tf
from tensorflow.keras import layers, Model


def build_model(img_size=64):
    inputs = tf.keras.Input(shape=(img_size, img_size, 1))
    x = layers.Conv2D(16, 3, padding="same", activation="relu", name="conv1")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu", name="conv2")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu", name="conv3")(x)
    # this is the layer Grad-CAM will target: last conv layer before pooling
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model
