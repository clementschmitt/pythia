import requests #télécharger les images
import pandas as pa #lire le dataset
import sys

from config import ROOT, DATA_DIR, THUMBNAILS_DIR

df = pa.read_json(DATA_DIR / "dataset_fps_fr_labeled.json")

# Creating thumbnails directory if it doesn't exist
if not THUMBNAILS_DIR.exists():
    THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)

#Downloading thumbnails for each video
for i, row in df.iterrows():
    url = row["thumbnail_url"]
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Check if the request was successful
        chemin = THUMBNAILS_DIR / f"{row['video_id']}.jpg"

        with open(chemin, "wb") as f:
            f.write(response.content)

        print(f"Video n° {i} / {df.shape[0]}")

    except requests.RequestException as e:
        print(f"Error downloading thumbnail for video {row['video_id']}: {e}", file=sys.stderr)
        continue