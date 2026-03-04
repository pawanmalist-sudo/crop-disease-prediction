import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import json
import os

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_DIR = r"P:\crop-disease-prediction\dataset"

if __name__ == "__main__":
    print("Loading dataset...")

    # Load train dataset and get class names BEFORE any mapping
    train_ds_raw = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "train"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int"
    )
    val_ds_raw = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "valid"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int"
    )

    # Get class names before mapping
    class_names = train_ds_raw.class_names
    print(f"Found {len(class_names)} classes")

    # Save class names
    os.makedirs(r"P:\crop-disease-prediction\model", exist_ok=True)
    with open(r"P:\crop-disease-prediction\class_names.json", "w") as f:
        json.dump(class_names, f)
    print("Class names saved!")

    # Normalize
    norm = tf.keras.layers.Rescaling(1.0 / 255)
    train_ds = train_ds_raw.map(lambda x, y: (norm(x), y))
    val_ds = val_ds_raw.map(lambda x, y: (norm(x), y))
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    # Build model
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
    base_model.trainable = False
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(len(class_names), activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            r"P:\crop-disease-prediction\model\best_model.h5",
            save_best_only=True,
            monitor="val_accuracy",
            verbose=1
        )
    ]

    print("Starting training...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=callbacks)
    print("Training complete!")
    print(f"Best Val Accuracy: {max(history.history['val_accuracy']):.4f}")
