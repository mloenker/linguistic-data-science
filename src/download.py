import csv
import requests
import xml.etree.ElementTree as ET
import os
import time
import json
from urllib.parse import urlparse
import re
from pathlib import Path
from datetime import datetime, timezone
import enlighten
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import random
from tqdm import tqdm

class PodcastDownloader:
    def __init__(self):
        config_path = Path(__file__).parent.parent / 'config.json'
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.data_dir = Path(config['DATA_DIR'])
        self.csv_file = Path(__file__).parent.parent / config['PODCAST_LIST_FILE']
        self.data_dir.mkdir(exist_ok=True)
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        
    def load_podcasts_from_csv(self):
        podcasts = []
        with open(self.csv_file, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            
            for row in reader:
                if len(row) >= 2 and row[1].strip():
                    name = row[0].strip()
                    rss_feed = row[1].strip()
                    if rss_feed:
                        podcasts.append({'name': name, 'rss_url': rss_feed})
        return podcasts
    
    def sanitize_filename(self, filename):
        filename = re.sub(r'[<>:"/\\|?*{},.„""]', '', filename)
        filename = filename.replace(' ', '_')
        filename = filename.lower()
        return filename[:200]
    
    def fetch_rss_feed_data(self, rss_url, podcast_name):
        try:
            response = self.session.get(rss_url, timeout=30)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            episodes = []
            
            for item in root.findall('.//item'):
                episode_data = {}
                
                title_elem = item.find('title')
                episode_data['title'] = title_elem.text if title_elem is not None else ""

                pub_date_elem = item.find('pubDate')
                episode_data['pub_date'] = pub_date_elem.text if pub_date_elem is not None else ""

                author_elem = item.find("{http://www.itunes.com/dtds/podcast-1.0.dtd}author")
                episode_data['author'] = author_elem.text if author_elem is not None else ""

                url_elem = item.find('enclosure').get('url')
                episode_data['url'] = url_elem if url_elem is not None else ""

                keywords_elem = item.find("{http://www.itunes.com/dtds/podcast-1.0.dtd}keywords")
                episode_data['keywords'] = keywords_elem.text if keywords_elem is not None else ""

                duration_elem = item.find("{http://www.itunes.com/dtds/podcast-1.0.dtd}duration")
                duration_text = duration_elem.text if duration_elem is not None else None

                if ":" in (duration_text):
                    episode_data['duration'] = sum(x * 60 ** i for i, x in enumerate(reversed(list(map(int, duration_text.split(':'))))))
                elif duration_text is not None:
                    episode_data['duration'] = int(duration_text)
                else:
                    episode_data['duration'] = 4500 # assume a default of 75 minutes

                file_size = item.find('enclosure').get('length')
                episode_data['file_size'] = int(file_size) if file_size and file_size.isdigit() else 0

                episode_data['podcast_name'] = podcast_name

                episodes.append(episode_data)
            
            return episodes
            
        except Exception as e:
            print(f"Error fetching RSS feed {rss_url}: {str(e)}")
            return []
    

    def download_all(self, max_threads=3):
        podcasts = self.load_podcasts_from_csv()
        total_episodes = 0
        total_size = 0
        total_duration = 0
        all_episodes = []

        for podcast in podcasts:
            episodes = self.fetch_rss_feed_data(podcast['rss_url'], podcast['name'])
            total_episodes += len(episodes)
            total_size += sum(ep['file_size'] for ep in episodes)
            total_duration += sum(int(ep['duration']) for ep in episodes)
            all_episodes.extend(episodes)

        completed_downloads = {f.stem: metadata.get('download_finished', False) for f in self.data_dir.rglob("*.metadata") if (metadata := json.loads(f.read_text(encoding='utf-8')))}
        
        remaining_episodes = [
            ep for ep in all_episodes
            if not completed_downloads.get(
                f"{self.sanitize_filename(ep['podcast_name'])}_{self.sanitize_filename(ep['title'])}", False
            )
        ]

        remaining_file_size = sum(ep['file_size'] for ep in remaining_episodes)
        remaining_duration = sum(int(ep['duration']) for ep in remaining_episodes)

        print(f"Total remaining file size: {remaining_file_size / (1024**3):.2f} GB")

        lock = threading.Lock()

        # add tqdm
        with tqdm(total=total_episodes, desc="Downloading Episodes", unit="episode", initial=total_episodes - len(remaining_episodes), colour="magenta", position=0) as self.top_pbar:

            def worker():
                while True:
                    with lock:
                        if not remaining_episodes:
                            return
                        idx = random.randrange(len(remaining_episodes))
                        ep = remaining_episodes.pop(idx)
                    try:
                        self.download_episode(ep)
                        self.top_pbar.update(1)
                    except Exception as e:
                        print(f"Download failed: {ep.get('title','<no title>')}: {e}")

            with ThreadPoolExecutor(max_workers=max_threads) as ex:
                futures = [ex.submit(worker) for _ in range(max_threads)]
                for _ in as_completed(futures):
                    pass

    def download_episode(self, ep, retries=3, chunk_size=1 << 20):
        url = ep.get("url") or ""
        if not url:
            raise ValueError(f"No audio URL for: {ep.get('title','<no title>')}")

        podcast_dir = self.data_dir / self.sanitize_filename(ep['podcast_name'])
        podcast_dir.mkdir(exist_ok=True)

        base = f"{self.sanitize_filename(ep['title'])}"
        ext = os.path.splitext(urlparse(url).path)[1] or ".mp3"
        audio_path = podcast_dir / f"{base}{ext}"
        part_path = audio_path.with_suffix(audio_path.suffix + ".part")
        meta_path = podcast_dir / f"{base}.metadata"  # write inside subfolder

        def now_utc_iso():
            return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        start_ts = time.time()
        downloaded = part_path.stat().st_size if part_path.exists() else 0
        headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}

        attempt = 0
        try:
            with tqdm(total=ep.get('file_size', 0), unit='B', unit_scale=True, desc=f"{ep.get('podcast_name')[0:18]}: {ep.get('title','<no title>')[0:14]}", leave=False) as pbar:
                while attempt < retries:
                    attempt += 1
                    with self.session.get(url, stream=True, timeout=60, headers=headers) as r:
                        r.raise_for_status()
                        mode = "ab" if (downloaded and r.status_code == 206) else "wb"
                        if mode == "wb" and part_path.exists():
                            part_path.unlink(missing_ok=True)

                        with open(part_path, mode) as f:
                            for chunk in r.iter_content(chunk_size=chunk_size):
                                if not chunk:
                                    continue
                                f.write(chunk)
                                downloaded += len(chunk)
                                pbar.update(len(chunk))
                                self.top_pbar.refresh()
                    break
        except Exception as e:
            if attempt >= retries:
                print(f"Failed to download {ep.get('title','<no title>')} after {retries} attempts.")
            raise

        part_path.replace(audio_path)
        elapsed = max(0.0, time.time() - start_ts)
        final_size = audio_path.stat().st_size

        metadata = {
            "title": ep.get("title", ""),
            "author": ep.get("author", ""),
            "keywords": ep.get("keywords", ""),
            "pub_date": ep.get("pub_date", ""),
            "podcast_name": ep.get("podcast_name", ""),
            "download_date": now_utc_iso(),
            "audio_file_size": final_size,
            "audio_url": url,
            "audio_duration": int(ep.get("duration", 0)),
            "download_finished": True,
            "download_time": elapsed,
            "transcription_date": "",
            "transcription_time": 0,
            "transcription_finished": False,
        }
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")



if __name__ == "__main__":
    downloader = PodcastDownloader()
    downloader.download_all()
