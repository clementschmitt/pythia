# PYTHIA — YouTube Virality Prediction using Machine Learning

> 🇬🇧 [English](#english) · 🇫🇷 [Français](#français)

---

## English

A YouTube virality prediction tool based on a full ML pipeline.
Built on a dataset of French-language FPS/TPS gaming videos.

### Context

The goal is to predict whether a YouTube video will go viral before it's published,
using data known before publication (title, duration, publishing time, thumbnail).

Virality is defined as 2 classes based on quantiles:
- **Viral** — top 25% by view count
- **Not Viral** — below that threshold

### Dataset

- ~6291 videos collected via YouTube Data API v3
- ~3557 videos after cleaning (duplicates, Shorts, VODs, outliers removed)
- Niche: French FPS/TPS gaming (Valorant, CS2, Apex, CoD, Black Ops 6, Marvel Rivals, Arc Raiders, Helldivers 2...)
- Features: views, likes, title, thumbnail URL

### Pipeline

```
collect.py             → YouTube API scraping
clean.py               → Cleaning (remove Shorts, VODs, duplicates)
label.py               → Labeling by quantile (Viral / Not Viral)
train_ml.py            → Classical ML training
train_cnn.py           → CNN training on thumbnails (with data augmentation)
train_cnn_mobilenet.py → Transfer learning MobileNetV2
log_results.py         → Log automatique des résultats d'entraînement
```

### Models

#### Classical ML (3557 videos, pre-publication features only)
| Model | Accuracy | Viral Recall | Viral F1 |
|-------|----------|--------------|----------|
| Random Forest (`class_weight=balanced`) | 73.91% | 0.14 | 0.22 |
| XGBoost (`scale_pos_weight`) | 75.00% | 0.43 | 0.45 |

#### Deep Learning — CNN on thumbnails
| Model | Best Val Accuracy | Test Accuracy | Notes |
|-------|------------------|---------------|-------|
| CNN from scratch | ~53% | - | Unstable, insufficient data |
| MobileNetV2 Softmax (patience=3) | 72.80% | 69.10% | Baseline |
| MobileNetV2 Sigmoid (patience=3) | 74.86% | 71.72% | Better generalization |
| MobileNetV2 Sigmoid (patience=5) | **77.30%** | **71.35%** | Best config — EarlyStopping optimal |
| MobileNetV2 Sigmoid (patience=7) | 74.86% | 70.60% | Diminishing returns |

> Best config: Sigmoid + `binary_crossentropy` + `patience=5` + `learning_rate=1e-5`

#### Deep Learning — coming soon
- LSTM on video titles
- CamemBERT for French semantic analysis

### Installation

```bash
git clone https://github.com/clementschmitt/pythia
cd pythia
pip install -r requirements.txt
```

### Usage

```bash
python collect.py                   # 1. Collect data
python clean.py                     # 2. Clean dataset
python label.py                     # 3. Label
python train_ml.py                  # 4. Train classical ML
py -3.12 train_cnn.py               # 5. Train CNN (thumbnails)
py -3.12 train_cnn_mobilenet.py     # 6. Transfer learning MobileNetV2
```

### Configuration

Add your YouTube Data API v3 key to `.env` (create the file first).

### Stack

Python · Scikit-learn · XGBoost · TensorFlow · CamemBERT · YouTube Data API v3

### Author

Clément Schmitt — [LinkedIn](https://www.linkedin.com/in/clement-schmitt/)

---

## Français

Outil de prédiction de viralité de vidéos YouTube basé sur un pipeline ML complet.
Développé sur un dataset de vidéos gaming FPS/TPS FR.

### Contexte

L'objectif est de prédire si une vidéo YouTube sera virale avant sa publication,
en s'appuyant sur des données connues avant publication (titre, durée, heure de publication, miniature).

La viralité est définie en 2 classes calculées par quantile sur le dataset :
- **Viral** — vidéos dans le top 25% des vues
- **Not Viral** — vidéos sous ce seuil

### Dataset

- ~6291 vidéos collectées via l'API YouTube Data v3
- ~3557 vidéos après nettoyage (suppression doublons, Shorts, VODs, valeurs aberrantes)
- Niche : gaming FPS/TPS FR (Valorant, CS2, Apex, CoD, Black Ops 6, Marvel Rivals, Arc Raiders, Helldivers 2...)
- Features : vues, likes, titre, URL miniature

### Pipeline

```
collect.py             → Scraping API YouTube
clean.py               → Nettoyage (suppression Shorts, VODs, doublons)
label.py               → Labellisation par quantile (Viral / Not Viral)
train_ml.py            → Entraînement ML classique
train_cnn.py           → Entraînement CNN sur miniatures (avec data augmentation)
train_cnn_mobilenet.py → Transfer learning MobileNetV2
log_results.py         → Log automatique des résultats d'entraînement
```

### Modèles

#### ML Classique (3557 vidéos, features pré-publication uniquement)
| Modèle | Accuracy | Recall viral | F1 viral |
|--------|----------|--------------|----------|
| Random Forest (`class_weight=balanced`) | 73.91% | 0.14 | 0.22 |
| XGBoost (`scale_pos_weight`) | 75.00% | 0.43 | 0.45 |

#### Deep Learning — CNN sur miniatures
| Modèle | Meilleure Val Accuracy | Test Accuracy | Notes |
|--------|----------------------|---------------|-------|
| CNN from scratch | ~53% | - | Instable, pas assez de données |
| MobileNetV2 Softmax (patience=3) | 72.80% | 69.10% | Baseline |
| MobileNetV2 Sigmoid (patience=3) | 74.86% | 71.72% | Meilleure généralisation |
| MobileNetV2 Sigmoid (patience=5) | **77.30%** | **71.35%** | Meilleure config — EarlyStopping optimal |
| MobileNetV2 Sigmoid (patience=7) | 74.86% | 70.60% | Rendements décroissants |

> Meilleure config : Sigmoid + `binary_crossentropy` + `patience=5` + `learning_rate=1e-5`

#### Deep Learning — à venir
- LSTM sur les titres
- CamemBERT pour l'analyse sémantique en français

### Installation

```bash
git clone https://github.com/clementschmitt/pythia
cd pythia
pip install -r requirements.txt
```

### Utilisation

```bash
python collect.py                   # 1. Collecter les données
python clean.py                     # 2. Nettoyer le dataset
python label.py                     # 3. Labelliser
python train_ml.py                  # 4. Entraîner ML classique
py -3.12 train_cnn.py               # 5. Entraîner CNN (miniatures)
py -3.12 train_cnn_mobilenet.py     # 6. Transfer learning MobileNetV2
```

### Configuration

Renseigner votre clé API YouTube Data v3 dans `.env` et pensez à créer le fichier au préalable.

### Stack

Python · Scikit-learn · XGBoost · TensorFlow · CamemBERT · YouTube Data API v3

### Auteur

Clément Schmitt — [LinkedIn](https://www.linkedin.com/in/clement-schmitt/)
