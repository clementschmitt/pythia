import pandas as pa
import re

from pathlib import Path
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"

def duration_to_seconds(d):
    match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", d)
    if not match:
        return 0
    h = int(match.group(1) or 0)
    m = int(match.group(2) or 0)
    s = int(match.group(3) or 0)
    return h*3600+m*60+s


df = pa.read_json(DATA_DIR / "dataset_fps_fr_v2.json")

# On supprime les vidéos avec des informations NaN dans la colonnes video_id en attendant de les récupére à nouveau.
# Premières vidéos récupérées et sauvegardées sans les nouvelles colonnes du coup 
df = df.dropna(subset=["video_id"])

print(f"Doublons : {df.duplicated(subset='video_id').sum()}")
df = df.drop_duplicates(subset="video_id")

df["view_count"] = df["view_count"].astype(int)
df["like_count"] = df["like_count"].astype(int)
df["comment_count"] = df["comment_count"].astype(int)
df["subscriber_count"] = df["subscriber_count"].astype(int)
df["duration_seconds"] = df["duration"].apply(duration_to_seconds)

df["published_at"] = pa.to_datetime(df["published_at"])
df["day_of_week"] = df["published_at"].dt.dayofweek
df["hour"] = df["published_at"].dt.hour

print(f"Jour le plus fréquent : {df['day_of_week'].mode()[0]}")
print(f"Heure la plus fréquente : {df['hour'].mode()[0]}")

print(f"Durée min : {df["duration_seconds"].min()}s")
print(f"Durée max : {df["duration_seconds"].max()}s")
print(f"Durée médiane : {df["duration_seconds"].median()}s")


df = df[df['duration_seconds'] >= 180]
df = df[df['duration_seconds'] <= 2400]
print(f"Vidéos après filtrage durée (3min-40min) : {df.shape[0]}")

#calculer les features de titre
df["title_length"] = df["title"].apply(len)
df["has_number"] = df["title"].apply(lambda t : any (c.isdigit() for c in t))
df["has_punctuation"] = df["title"].apply(lambda t : "?" in t or "!" in t)
df["upper_word_count"] = df["title"].apply(lambda t : len([w for w in t.split() if w.isupper() and len(w) > 1]))

#Calcul de l'engagement rate et du subscriber ratio
df["engagement_rate"] = (df["like_count"] + df["comment_count"]) / df["view_count"] * 100
df["subscriber_ratio"] = df["view_count"] / df["subscriber_count"]

df.to_json(DATA_DIR / "dataset_fps_fr_clean.json", orient="records", indent=2, force_ascii=False)
print(f"\nDataset nettoyé sauvergardé : {df.shape[0]} vidéos, {df.shape[1]} colonnes")