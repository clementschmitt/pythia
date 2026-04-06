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

for i, row in df.iterrows():
    image = Image.open(THUMBNAILS_DIR / f"{row['video_id']}.jpg")
    image_resized = image.resize((224,224))

    arr = np.array(image_resized) / 255.0
    images.append(arr)

X = np.array(images)

# Split 70% train / 15% validation / 15% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

# Define our base model : We use MobileNetV2, a pre-trained convolutional neural network that has been trained 
# on the ImageNet dataset. We set include_top=False to exclude the final classification layer, 
# and weights="imagenet" to load the pre-trained weights. 
# We also freeze the base model's layers to prevent them from being updated during training.
base_model = keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False, weights="imagenet")
base_model.trainable = False

x = base_model.output
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(64, activation="relu")(x)
x = keras.layers.Dropout(0.5)(x)
predictions = keras.layers.Dense(4, activation="softmax")(x)

model = keras.Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

loss, accuracy = model.evaluate(X_test, y_test)

print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.2%}")