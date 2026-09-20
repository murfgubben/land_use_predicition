"""Transfer-learning model for satellite image classification."""

import tensorflow as tf

from .config import DATA_CONFIG, MODEL_CONFIG


def create_model(num_classes: int) -> tf.keras.Model:
    """Create a MobileNetV2 classifier with a frozen pretrained backbone."""
    if num_classes < 2:
        raise ValueError(f"Expected at least two classes, got {num_classes}.")

    inputs = tf.keras.Input(
        shape=(*DATA_CONFIG.image_size, DATA_CONFIG.input_channels),
        name="image",
    )
    scaled_inputs = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(*DATA_CONFIG.image_size, DATA_CONFIG.input_channels),
        include_top=False,
        weights=MODEL_CONFIG.backbone_weights,
    )
    backbone.trainable = False
    features = backbone(scaled_inputs, training=False)
    features = tf.keras.layers.GlobalAveragePooling2D()(features)
    features = tf.keras.layers.Dropout(MODEL_CONFIG.dropout_rate)(features)
    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="class_probabilities"
    )(features)
    model = tf.keras.Model(inputs, outputs, name="satellite_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=MODEL_CONFIG.initial_learning_rate
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def enable_fine_tuning(model: tf.keras.Model) -> None:
    """Unfreeze only the top backbone layers and compile at a low learning rate."""
    backbone = next(
        layer
        for layer in model.layers
        if isinstance(layer, tf.keras.Model)
        and layer.name.startswith("mobilenetv2")
    )
    backbone.trainable = True
    frozen_layers = max(0, len(backbone.layers) - MODEL_CONFIG.fine_tune_layers)
    for layer in backbone.layers[:frozen_layers]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=MODEL_CONFIG.fine_tuning_learning_rate
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
