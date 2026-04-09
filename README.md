# PYTHIA — Prédiction de viralité YouTube par Machine Learning

Outil de prédiction de viralité de vidéos YouTube basé sur un pipeline ML complet.
Développé sur un dataset de vidéos gaming FPS FR.

## Contexte

L'objectif est de prédire si une vidéo YouTube sera virale avant sa publication,
en s'appuyant sur des données historiques (vues, likes, titre, miniature).

La viralité est définie en 2 classes calculées par quantile sur le dataset :
- **Viral** — vidéos dans le top 25% des vues
- **Not Viral** — vidéos sous ce seuil

## Dataset

- ~6291 vidéos collectées via l'API YouTube Data v3
- ~3557 vidéos après nettoyage (suppression doublons, Shorts, VODs, valeurs aberrantes)
- Niche : gaming FPS/TPS FR (Valorant, CS2, Apex, CoD, Black Ops 6, Marvel Rivals, Arc Raiders, Helldivers 2...)
- Features : vues, likes, titre, URL miniature

## Pipeline

collect.py    → Scraping API YouTube
clean.py      → Nettoyage (suppression Shorts, VODs, doublons)
label.py      → Labellisation par quantiles (Viral / Strong / Moderate / Weak)
train_ml.py   → Entraînement ML classique
train_cnn.py          → Entraînement CNN sur miniatures (avec data augmentation)
train_cnn_mobilenet.py → Transfer learning MobileNetV2



## Modèles

### ML Classique (3557 vidéos, features pré-publication uniquement)
| Modèle | Accuracy | Recall viral | F1 viral |
|--------|----------|--------------|----------|
| Random Forest (`class_weight=balanced`) | 73.91% | 0.14 | 0.22 |
| XGBoost (`scale_pos_weight`) | 75.00% | 0.43 | 0.45 |

### Deep Learning — CNN sur miniatures

| Modèle | Val Accuracy | Notes |
|--------|--------------|-------|
| CNN from scratch | ~53% | Instable, pas assez de données |
| MobileNetV2 (transfer learning) | 74.86% (epoch 10) | En progression, entraînement non terminé |

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