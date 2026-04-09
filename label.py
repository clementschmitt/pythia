import pandas as pa
from config import ROOT, DATA_DIR

df = pa.read_json(DATA_DIR / "dataset_fps_fr_clean.json")

print(f"=== DISTRIBUTION VIEW ===")
print(f"P75 : {df['view_count'].quantile(0.75)}")

seuil = df['view_count'].quantile(0.75)

def labelliser(views):
    if views > seuil:
        return "viral"
    else:
        return "not_viral"

df["label"] = df["view_count"].apply(labelliser)
print(df["label"].value_counts())

df.to_json(DATA_DIR / "dataset_fps_fr_labeled.json", orient="records", indent=2, force_ascii=False)
print(f"\nDataset labellisé sauvegardé : {df.shape[0]} vidéos")