import pandas as pa
from PIL import Image
import numpy as np

from pathlib import Path
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"

df = pa.read_json(DATA_DIR / "dataset_fps_fr_labeled.json")

for i, row in df.head(5).iterrows():
    image = Image.open(ROOT / ".tmp" / "thumbnails" / f"{row['video_id']}.jpg")
    image_resized = image.resize((224,224))
    arr = np.array(image_resized) / 255.0
    print(f"Video ID : {row['video_id']}")
    print(f"Label : {row['label']}")