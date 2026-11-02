"""Siamese CNN: a shared convolutional encoder, an L1 distance head and a sigmoid output.

TensorFlow loads inside each function, so the rest of the package imports
without it.
"""

from __future__ import annotations


def build_encoder(input_shape: tuple[int, int, int], embedding_dim: int = 128):
    """Return the convolutional encoder that both branches share."""
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=input_shape)
    x = inputs
    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(embedding_dim, activation="relu")(x)
    return keras.Model(inputs, x, name="encoder")


def build_siamese(input_shape: tuple[int, int, int], learning_rate: float = 1e-3):
    """Build and compile the Siamese model. Output 1 means both images share a class."""
    from tensorflow import keras
    from tensorflow.keras import layers

    encoder = build_encoder(input_shape)
    in_a = keras.Input(shape=input_shape, name="img_a")
    in_b = keras.Input(shape=input_shape, name="img_b")
    emb_a, emb_b = encoder(in_a), encoder(in_b)

    # L1 distance |a - b| = relu(a - b) + relu(b - a). Built-in layers keep the saved model portable.
    l1 = layers.Add(name="l1_distance")([
        layers.ReLU()(layers.Subtract()([emb_a, emb_b])),
        layers.ReLU()(layers.Subtract()([emb_b, emb_a])),
    ])
    out = layers.Dense(1, activation="sigmoid", name="same_class")(l1)

    model = keras.Model([in_a, in_b], out, name="siamese_watermark")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_model(path):
    """Load a model saved by train.py."""
    from tensorflow import keras

    return keras.models.load_model(path)
