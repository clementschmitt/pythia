import sys

import pandas as pa
from PIL import Image
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from log_results import log_result

from config import ROOT, DATA_DIR, THUMBNAILS_DIR

df = pa.read_json(DATA_DIR / "dataset_fps_fr_labeled.json")
le = LabelEncoder()
y = le.fit_transform(df["label"])

images = []
labels = []

for i, row in df.iterrows():
    try:
        image = Image.open(THUMBNAILS_DIR / f"{row['video_id']}.jpg")
        image_resized = image.resize((224,224))

        arr = preprocess_input(np.array(image_resized).astype(np.float32))
        images.append(arr)
        labels.append(y[i])
    except Exception as e:
        print(f"Error loading image for video {row['video_id']}: {e}", file=sys.stderr)
        continue

X = np.array(images)
y = np.array(labels)

# Split 70% train / 15% validation / 15% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

# Define our base model : We use MobileNetV2, a pre-trained convolutional neural network that has been trained 
# on the ImageNet dataset. We set include_top=False to exclude the final classification layer, 
# and weights="imagenet" to load the pre-trained weights. 
# We also freeze the base model's layers to prevent them from being updated during training.
base_model = keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False, weights="imagenet")
base_model.trainable = True
for layer in base_model.layers[:-10]:
    layer.trainable = False

x = base_model.output
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(64, activation="relu")(x)
x = keras.layers.Dropout(0.5)(x)
predictions = keras.layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=keras.optimizers.Adam(1e-5), loss="binary_crossentropy", metrics=["accuracy"])

class_weights = {
    0: len(y_train) / (2 * (y_train == 0).sum()),
    1: len(y_train) / (2 * (y_train == 1).sum())
}

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss", 
    patience=5,
    restore_best_weights=True
)

history = model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val), class_weight=class_weights, callbacks=[early_stopping])

loss, accuracy = model.evaluate(X_test, y_test)

print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.2%}")

log_result(
    params={"model": "MobileNetV2", "activation": "sigmoid", "loss_fn": "binary_crossentropy", "learning_rate": 1e-5, "patience": 5},
    val_history=history.history["val_loss"],
    val_acc_history=history.history["val_accuracy"],
    test_accuracy=accuracy,
    test_loss=loss
)