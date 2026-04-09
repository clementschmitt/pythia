import sys

import pandas as pa
from PIL import Image
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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
        arr = np.array(image_resized) / 255.0
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

# Define the CNN model : We use a simple architecture with 2 convolutional layers, followed by max pooling, 
# flattening, and dense layers. The final layer has 2 units (one for each class) and uses sigmoid activation 
# for binary classification.
model_cnn = keras.Sequential([keras.layers.Input(shape=(224,224,3)),
                keras.layers.RandomFlip("horizontal"),
                keras.layers.RandomRotation(0.1),
                keras.layers.RandomZoom(0.1),
                keras.layers.Conv2D(32, kernel_size=3, activation="relu"),
                keras.layers.MaxPooling2D(pool_size=2),
                keras.layers.Conv2D(64, kernel_size=3, activation="relu"),
                keras.layers.MaxPooling2D(pool_size=2),
                keras.layers.Flatten(),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(2, activation="softmax")
                ])

# Compile the model : We define how the model will learn : the optimizer (Adam), the loss function 
# (sparse_categorical_crossentropy) and the metrics (accuracy).
model_cnn.compile(optimizer = keras.optimizers.Adam(learning_rate=0.0001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Handle class imbalance : We calculate class weights to give more importance to the minority class during training.
# Here, the minority class is Viral (1), so we assign a higher weight to it compared to the Non-Viral class (0).
class_weights = {
    # 753 / (2 * 753) = 1.0 for class 0 (non-viral), and 753 / (2 * 247) ≈ 1.52 for class 1 (viral)
    0: len(y_train) / (2 * (y_train == 0).sum()),
    1: len(y_train) / (2 * (y_train == 1).sum())
}

# Train the model : We fit the model on the training data for 10 epochs, and we also provide the validation 
# data to monitor the performance of the model on unseen data during training.
model_cnn.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), class_weight=class_weights)

#Evaluate the model : Finally, we evaluate the performance of the trained model on the test set to see how well 
# it generalizes to new data.
loss, accuracy = model_cnn.evaluate(X_test, y_test)

print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.2%}")