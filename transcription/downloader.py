"""
downloader.py  –  Download 10 episodes per podcast from RSS feeds.

Directory layout after running:
    data/
        <podcast_name>/
            <episode_title>.mp3
            <episode_title>.json   ← metadata (title, date, duration, …)
"""

import csv
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
import xml.etree.ElementTree as ET
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────────────────────

PODCAST_LIST  = Path("podcasts.csv")
from google.colab import drive
drive.mount('/content/drive')

DATA_DIR = Path("/content/drive/MyDrive/podcast_data")   # two-column CSV: name, rss_url
EPISODES_EACH = 10                     # how many episodes to download per podcast

# ──────────────────────────────────────────────────────────────────────────────


def sanitize(name: str) -> str:
    """Turn any string into a safe filename."""
    name = re.sub(r'[<>:"/\\|?*{},.„""]', "", name)
    return name.replace(" ", "_").lower()[:200]


def load_podcasts() -> list[dict]:
    podcasts = []
    with open(PODCAST_LIST, encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[1].strip():
                podcasts.append({"name": row[0].strip(), "rss_url": row[1].strip()})
    return podcasts


def fetch_episodes(podcast: dict) -> list[dict]:
    """Fetch up to EPISODES_EACH episodes from an RSS feed."""
    itunes = "http://www.itunes.com/dtds/podcast-1.0.dtd"
    try:
        r = requests.get(podcast["rss_url"], timeout=30)
        r.raise_for_status()
        root = ET.fromstring(r.content)
    except Exception as e:
        print(f"  [!] Could not fetch RSS for {podcast['name']}: {e}")
        return []

    episodes = []
    for item in root.findall(".//item"):
        enclosure = item.find("enclosure")
        if enclosure is None:
            continue
        url = enclosure.get("url", "")
        if not url:
            continue

        title   = getattr(item.find("title"),   "text", "") or ""
        pubdate = getattr(item.find("pubDate"), "text", "") or ""

        duration_elem = item.find(f"{{{itunes}}}duration")
        duration_text = getattr(duration_elem, "text", None)
        if duration_text and ":" in duration_text:
            parts = list(map(int, duration_text.split(":")))
            duration = sum(x * 60**i for i, x in enumerate(reversed(parts)))
        elif duration_text and duration_text.isdigit():
            duration = int(duration_text)
        else:
            duration = 4500  # default 75 min

        file_size = int(enclosure.get("length", 0) or 0)

        episodes.append({
            "title":        title,
            "pub_date":     pubdate,
            "url":          url,
            "duration":     duration,
            "file_size":    file_size,
            "podcast_name": podcast["name"],
        })

        if len(episodes) == EPISODES_EACH:
            break

    return episodes


def already_downloaded(episode: dict) -> bool:
    podcast_dir = DATA_DIR / sanitize(episode["podcast_name"])
    meta_path   = podcast_dir / f"{sanitize(episode['title'])}.json"
    if not meta_path.exists():
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return meta.get("download_finished", False)


def download_episode(episode: dict) -> None:
    podcast_dir = DATA_DIR / sanitize(episode["podcast_name"])
    podcast_dir.mkdir(parents=True, exist_ok=True)

    base      = sanitize(episode["title"])
    ext       = os.path.splitext(urlparse(episode["url"]).path)[1] or ".mp3"
    audio_out = podcast_dir / f"{base}{ext}"
    part_out  = audio_out.with_suffix(audio_out.suffix + ".part")
    meta_out  = podcast_dir / f"{base}.json"

    # Resume support
    downloaded = part_out.stat().st_size if part_out.exists() else 0
    headers    = {"Range": f"bytes={downloaded}-"} if downloaded else {}

    t0 = time.time()
    with requests.get(episode["url"], stream=True, timeout=60, headers=headers) as r:
        r.raise_for_status()
        mode = "ab" if (downloaded and r.status_code == 206) else "wb"
        total = episode["file_size"] or None

        with open(part_out, mode) as f, tqdm(
            total=total, initial=downloaded if mode == "ab" else 0,
            unit="B", unit_scale=True, leave=False,
            desc=f"  {episode['title'][:50]}"
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))

    part_out.replace(audio_out)

    meta = {
        "title":             episode["title"],
        "pub_date":          episode["pub_date"],
        "podcast_name":      episode["podcast_name"],
        "audio_url":         episode["url"],
        "audio_duration":    episode["duration"],
        "audio_file_size":   audio_out.stat().st_size,
        "download_date":     datetime.now(timezone.utc).isoformat(),
        "download_time":     round(time.time() - t0, 1),
        "download_finished": True,
        "transcription_finished": False,
    }
    meta_out.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    DATA_DIR.mkdir(exist_ok=True)
    podcasts = load_podcasts()
    print(f"Loaded {len(podcasts)} podcasts, downloading up to {EPISODES_EACH} episodes each.\n")

    for podcast in podcasts:
        print(f"── {podcast['name']}")
        episodes = fetch_episodes(podcast)
        if not episodes:
            continue

        for ep in episodes:
            if already_downloaded(ep):
                print(f"  [skip] {ep['title'][:60]}")
                continue
            print(f"  [dl]   {ep['title'][:60]}")
            try:
                download_episode(ep)
            except Exception as e:
                print(f"  [!]   Failed: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()