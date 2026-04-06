# PYTHIA — Prédiction de viralité YouTube par Machine Learning

Outil de prédiction de viralité de vidéos YouTube basé sur un pipeline ML complet.
Développé sur un dataset de vidéos gaming FPS FR.

## Contexte

L'objectif est de prédire si une vidéo YouTube sera virale avant sa publication,
en s'appuyant sur des données historiques (vues, likes, titre, miniature).

La viralité est définie en 4 niveaux calculés par quantiles sur le dataset :
- **Viral** — vidéos dans le top 10%
- **Strong** — vidéos dans le top 10% à 25% de vues
- **Moderate** — vidéos dans le 25% à 80%
- **Weak** — vidéos sous le top 80%

## Dataset

- ~4000 vidéos collectées via l'API YouTube Data v3
- ~1076 vidéos après nettoyage (suppression doublons, Shorts, VODs, valeurs aberrantes)
- Niche : gaming FPS FR (Valorant, CS2, Apex, CoD...)
- Features : vues, likes, titre, URL miniature

## Pipeline

collect.py    → Scraping API YouTube
clean.py      → Nettoyage (suppression Shorts, VODs, doublons)
label.py      → Labellisation par quantiles (Viral / Strong / Moderate / Weak)
train_ml.py   → Entraînement ML classique
train_cnn.py          → Entraînement CNN sur miniatures (avec data augmentation)
train_cnn_mobilenet.py → Transfer learning MobileNetV2



## Modèles

### ML Classique
| Modèle | Accuracy |
|--------|----------|
| Random Forest | 58% |
| XGBoost | 60% |

### Deep Learning — CNN sur miniatures

| Modèle | Test Accuracy | Notes |
|--------|--------------|-------|
| CNN from scratch | 45.68% | Overfitting massif (train 91% / val 55%) |
| CNN + data augmentation | 52.47% | Overfitting réduit, généralisation améliorée |
| MobileNetV2 (transfer learning) | 48.77% | Transfer learning — poids gelés, Dense 64 + Dropout |

### Deep Learning — à venir
- LSTM sur les titres
- CamemBERT pour l'analyse sémantique en français

## Installation

```bash
git clone https://github.com/clementschmitt/pythia
cd pythia
pip install -r requirements.txt
```

## Utilisation

```bash
python collect.py       # 1. Collecter les données
python clean.py         # 2. Nettoyer le dataset
python label.py         # 3. Labelliser
python train_ml.py      # 4. Entraîner ML classique
python train_cnn.py            # 5. Entraîner CNN (miniatures)
py -3.12 train_cnn.py          # 5b. Avec TensorFlow (Python 3.12)
py -3.12 train_cnn_mobilenet.py # 6. Transfer learning MobileNetV2
```

## Configuration

Renseigner votre clé API YouTube Data v3 dans `.env` et pensez à créer le fichier au préalable.

## Stack

Python · Scikit-learn · XGBoost · TensorFlow · CamemBERT · YouTube Data API v3

## Auteur

Clément Schmitt — [LinkedIn](https://www.linkedin.com/in/clement-schmitt/)