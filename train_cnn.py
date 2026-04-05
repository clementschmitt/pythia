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

