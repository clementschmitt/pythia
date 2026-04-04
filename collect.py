"""
PYTHIA — Collecte de données YouTube FPS Gaming FR
Scrape l'API YouTube Data pour construire le dataset de vidéos FPS gaming francophone.
Usage : python pythia/collect.py
"""
import requests
import os
import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
from pathlib import Path

# Chemins
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
DATASET_FILE = DATA_DIR / "dataset_fps_fr_v2.json"

load_dotenv(ROOT / ".env")
cle = os.getenv("YOUTUBE_API_KEY")

# Requêtes ciblées par noms de jeux FPS
requetes = [
    "Valorant gameplay français",
    "CS2 gameplay FR",
    "Call of Duty gameplay français",
    "Overwatch 2 gameplay FR",
    "Rainbow Six Siege français",
    "Apex Legends gameplay FR",
    "Warzone gameplay FR",
    "Battlefield gameplay FR",
    "Escape from Tarkov gameplay FR",
    "The Finals gameplay FR",
    "Halo Infinite gameplay FR",
    "XDefiant gameplay FR",
    "Destiny 2 gameplay FR",
    "Ready or Not gameplay FR",
    "Hunt Showdown gameplay FR",
]

video_ids_vus = set()

# Charger le dataset existant
try:
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"{len(dataset)} vidéos déjà dans le dataset")
except FileNotFoundError:
    dataset = []
    print("Nouveau dataset")

for video in dataset:
    if "video_id" in video:
        video_ids_vus.add(video["video_id"])

for requete in requetes:

    next_page = None

    for page in range(5):

        params = {
            "part": "id",
            "q": requete,
            "type": "video",
            "maxResults": 50,
            "key": cle,
        }

        if next_page:
            params["pageToken"] = next_page

        reponse = requests.get("https://www.googleapis.com/youtube/v3/search", params=params)

        if reponse.status_code != 200:
            print(f"Erreur API search: {reponse.status_code} - {requete}")
            continue

        videos = reponse.json()["items"]

        # Extrait les IDs des vidéos
        video_ids = []
        for video in videos:
            if "videoId" in video["id"]:
                video_ids.append(video["id"]["videoId"])

        # Appel pour tous les détails
        details = requests.get("https://www.googleapis.com/youtube/v3/videos", params={
                "part": "snippet,statistics,contentDetails",
                "id": ",".join(video_ids),
                "key": cle,
            })

        if details.status_code != 200:
            print(f"Erreur API videos: {details.status_code} - {requete}")
            continue

        # Regroupe les IDs des chaînes
        channel_ids = []
        for video_detail in details.json()["items"]:
            channel_ids.append(video_detail["snippet"]["channelId"])

        # Appel pour récupérer les infos des chaînes
        channels = requests.get("https://www.googleapis.com/youtube/v3/channels", params={
                "part": "statistics",
                "id": ",".join(channel_ids),
                "key": cle,
            })

        if channels.status_code != 200:
            print(f"Erreur API channels: {channels.status_code} - {requete}")
            continue

        if "items" not in channels.json():
            print(f"Erreur API channels : pas de résultats - {requete}")
            continue

        # Mappage channelId → nombre d'abonnés
        subscribers = {}
        for channel in channels.json()["items"]:
            subscribers[channel["id"]] = channel["statistics"]["subscriberCount"]

        # Stockage des données pour chaque vidéo
        for video_detail in details.json()["items"]:

            video_id = video_detail["id"]
            if video_id in video_ids_vus:
                continue
            video_ids_vus.add(video_id)

            video_titre = video_detail["snippet"]["title"]
            video_viewCount = video_detail["statistics"]["viewCount"]
            video_likeCount = video_detail["statistics"].get("likeCount", "0")
            video_commentCount = video_detail["statistics"].get("commentCount", "0")
            video_subscribersCount = subscribers.get(video_detail["snippet"]["channelId"])
            video_publishedAt = video_detail["snippet"]["publishedAt"]
            video_language = video_detail["snippet"].get("defaultAudioLanguage", "unknown")
            video_duration = video_detail["contentDetails"]["duration"]
            video_tags = video_detail["snippet"].get("tags", [])
            video_thumbnailUrl = video_detail["snippet"]["thumbnails"]["high"]["url"]

            if not video_subscribersCount or int(video_subscribersCount) == 0:
                continue

            view = int(video_viewCount)
            if view == 0:
                continue

            video_data = {
                "video_id": video_id,
                "title": video_titre,
                "view_count": video_viewCount,
                "like_count": video_likeCount,
                "comment_count": video_commentCount,
                "published_at" : video_publishedAt,
                "language" : video_language,
                "duration" : video_duration,
                "tags" : video_tags,
                "thumbnail_url" : video_thumbnailUrl,
                "query_source": requete,
                "channel_name": video_detail["snippet"]["channelTitle"],
                "subscriber_count": video_subscribersCount
            }

            dataset.append(video_data)

        next_page = reponse.json().get("nextPageToken")
        if not next_page:
            break

    # Sauvegarde après chaque requête
    with open(DATASET_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

print(f"\nTotal : {len(dataset)} vidéos sauvegardées dans {DATASET_FILE}")
