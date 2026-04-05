import requests #télécharger les images
import pandas as pa #lire le dataset

from config import ROOT, DATA_DIR, THUMBNAILS_DIR

df = pa.read_json(DATA_DIR / "dataset_fps_fr_labeled.json")

for i, row in df.iterrows():
    url = row["thumbnail_url"]
    response = requests.get(url)
    
    chemin = THUMBNAILS_DIR / f"{row['video_id']}.jpg"

    with open(chemin, "wb") as f:
        f.write(response.content)

    print(f"Vidéo n° {i} / {df.shape[0]}")