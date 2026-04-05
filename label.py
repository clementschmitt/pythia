import pandas as pa
from config import ROOT, DATA_DIR

df = pa.read_json(DATA_DIR / "dataset_fps_fr_clean.json")

print(f"=== DISTRIBUTION VIEW ===")
print(f"P20 : {df['view_count'].quantile(0.20)}")
print(f"P75 : {df['view_count'].quantile(0.75)}")
print(f"P90 : {df['view_count'].quantile(0.90)}")

def labelliser(views):
    if views > 234421:
        return "viral"
    elif views > 12213:
        return "strong"
    elif views > 1704:
        return "moderate"
    else:
        return "weak"

df["label"] = df["view_count"].apply(labelliser)
print(df["label"].value_counts())

df.to_json(DATA_DIR / "dataset_fps_fr_labeled.json", orient="records", indent=2, force_ascii=False)
print(f"\nDataset labellisé sauvegardé : {df.shape[0]} vidéos")