import requests #télécharger les images
from PIL import Image #traiter les images
import pandas as pa #lire le dataset

from pathlib import Path
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"

df = pa.read_json(DATA_DIR / "dataset_fps_fr_labeled.json")

for i, row in df.iterrows():
    url = row["thumbnail_url"]
    response = requests.get(url)
    
    chemin = ROOT / ".tmp" / "thumbnails" / f"{row['video_id']}.jpg"

    with open(chemin, "wb") as f:
        f.write(response.content)

    print(f"Vidéo n° {i} / {df.shape[0]}")