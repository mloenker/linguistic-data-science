import csv
import requests
import xml.etree.ElementTree as ET
import os
import time
from urllib.parse import urlparse, urljoin
import re
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class PodcastDownloader:
    def __init__(self, csv_file, download_dir="data", rate_limit=1.0):
        self.csv_file = csv_file
        self.download_dir = Path(download_dir)
        self.rate_limit = rate_limit  # seconds between requests
        self.download_dir.mkdir(exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.progress_lock = threading.Lock()
        self.overall_progress = None
        
    def load_podcasts_from_csv(self):
        """Load podcast data from CSV file"""
        podcasts = []
        with open(self.csv_file, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # Skip first empty line
            headers = next(reader)  # Read headers
            
            for row in reader:
                if len(row) >= 3 and row[2].strip():  # Check if RSS feed exists
                    name = row[0].strip()
                    rss_feed = row[2].strip()
                    if rss_feed and not rss_feed.startswith('https://open.spotify.com'):
                        podcasts.append({
                            'name': name,
                            'rss_url': rss_feed
                        })
        return podcasts
    
    def sanitize_filename(self, filename):
        """Sanitize filename for safe saving"""
        # Remove invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)
        # Replace spaces with underscores
        filename = filename.replace(' ', '_')
        # Limit length
        return filename[:200]
    
    def fetch_rss_feed(self, rss_url):
        """Fetch and parse RSS feed"""
        try:
            response = self.session.get(rss_url, timeout=30)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            episodes = []
            
            # Find all item elements (episodes)
            for item in root.findall('.//item'):
                episode_data = {}
                
                # Extract basic episode info
                title_elem = item.find('title')
                episode_data['title'] = title_elem.text if title_elem is not None else 'Unknown'
                
                description_elem = item.find('description')
                episode_data['description'] = description_elem.text if description_elem is not None else ''
                
                pub_date_elem = item.find('pubDate')
                episode_data['pub_date'] = pub_date_elem.text if pub_date_elem is not None else ''
                
                # Find enclosure (audio file)
                enclosure = item.find('enclosure')
                if enclosure is not None:
                    episode_data['audio_url'] = enclosure.get('url')
                    episode_data['audio_type'] = enclosure.get('type', 'audio/mpeg')
                    episode_data['audio_length'] = enclosure.get('length', '0')
                else:
                    # Try alternative methods to find audio URL
                    for elem in item:
                        if 'url' in elem.tag.lower() and any(ext in str(elem.text).lower() for ext in ['.mp3', '.m4a', '.wav']):
                            episode_data['audio_url'] = elem.text
                            break
                
                if 'audio_url' in episode_data:
                    episodes.append(episode_data)
            
            return episodes
            
        except Exception as e:
            print(f"Error fetching RSS feed {rss_url}: {str(e)}")
            return []
    
    def count_total_episodes(self, podcasts):
        """Count total episodes across all podcasts for progress tracking"""
        print("Counting total episodes across all podcasts...")
        total_episodes = 0
        podcast_episode_counts = {}
        
        for podcast in tqdm(podcasts, desc="Scanning RSS feeds"):
            episodes = self.fetch_rss_feed(podcast['rss_url'])
            episode_count = len(episodes)
            podcast_episode_counts[podcast['name']] = episode_count
            total_episodes += episode_count
            time.sleep(0.5)  # Brief pause between RSS requests
        
        print(f"\nTotal episodes found: {total_episodes}")
        print("\nEpisodes per podcast:")
        for name, count in podcast_episode_counts.items():
            print(f"  {name}: {count} episodes")
        
        return total_episodes, podcast_episode_counts
    
    def download_episode(self, episode, podcast_name, progress_bar=None):
        """Download a single episode"""
        try:
            audio_url = episode['audio_url']
            if not audio_url:
                return False
            
            # Create podcast directory
            podcast_dir = self.download_dir / self.sanitize_filename(podcast_name)
            podcast_dir.mkdir(exist_ok=True)
            
            # Determine file extension
            parsed_url = urlparse(audio_url)
            file_ext = os.path.splitext(parsed_url.path)[1] or '.mp3'
            
            # Create filename
            title = self.sanitize_filename(episode['title'])
            filename = f"{title}{file_ext}"
            file_path = podcast_dir / filename
            
            # Skip if already downloaded (no rate limit needed)
            if file_path.exists():
                if progress_bar:
                    progress_bar.set_description(f"Skipping: {filename[:50]}...")
                return True
            
            # Download the file
            if progress_bar:
                progress_bar.set_description(f"Downloading: {filename[:50]}...")
            
            response = self.session.get(audio_url, stream=True, timeout=60)
            response.raise_for_status()
            
            # Save the file
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                    f.write(chunk)
            
            # Save episode metadata
            metadata = {
                'title': episode['title'],
                'description': episode['description'],
                'pub_date': episode['pub_date'],
                'audio_url': audio_url,
                'download_date': datetime.now().isoformat(),
                'file_path': str(file_path),
                'podcast_name': podcast_name,
                'file_size_bytes': file_path.stat().st_size
            }
            
            metadata_path = podcast_dir / f"{title}_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            return True
            
        except Exception as e:
            if progress_bar:
                progress_bar.set_description(f"Error: {str(e)[:50]}...")
            print(f"Error downloading episode {episode.get('title', 'Unknown')}: {str(e)}")
            return False
    
    def download_podcast_episodes(self, podcast_data):
        """Download all episodes for a single podcast"""
        podcast, expected_episodes = podcast_data
        podcast_name = podcast['name']
        
        # Create individual progress bar for this podcast
        podcast_progress = tqdm(total=expected_episodes, desc=f"{podcast_name[:30]}", 
                              unit="ep", position=None, leave=True)
        
        # Fetch episodes from RSS
        episodes = self.fetch_rss_feed(podcast['rss_url'])
        
        if not episodes:
            podcast_progress.close()
            return 0, expected_episodes
        
        # Update the total if it differs from expected
        if len(episodes) != expected_episodes:
            podcast_progress.total = len(episodes)
            podcast_progress.refresh()
        
        # Download episodes for this podcast
        podcast_downloaded = 0
        for episode in episodes:
            success = self.download_episode(episode, podcast_name)
            if success:
                podcast_downloaded += 1
            
            # Update both progress bars
            podcast_progress.update(1)
            if self.overall_progress:
                with self.progress_lock:
                    self.overall_progress.update(1)
            
            # Only apply rate limit when actually downloading (not for skipped files)
            file_path = self.download_dir / self.sanitize_filename(podcast_name) / f"{self.sanitize_filename(episode['title'])}.mp3"
            if not file_path.exists():  # Only rate limit for actual downloads
                time.sleep(self.rate_limit)
        
        podcast_progress.close()
        return podcast_downloaded, len(episodes)

    def download_all_podcasts(self, start_from_podcast=0, max_workers=3):
        """Download episodes from all podcasts with parallel processing"""
        podcasts = self.load_podcasts_from_csv()
        print(f"Found {len(podcasts)} podcasts with RSS feeds")
        
        # Count total episodes first
        total_episodes, podcast_episode_counts = self.count_total_episodes(podcasts)
        
        # Show size estimation and proceed automatically
        estimated_size_gb = total_episodes * 50 / 1024  # Rough estimate: 50MB per episode
        print(f"\nEstimated total download size: ~{estimated_size_gb:.1f} GB")
        print(f"\nProceeding with download using {max_workers} parallel threads...")
        
        print("\nStarting downloads...")
        total_downloaded = 0
        
        # Create overall progress bar
        self.overall_progress = tqdm(total=total_episodes, desc="Overall Progress", 
                                   unit="ep", position=0, leave=True)
        
        # Prepare podcast data for parallel processing
        podcast_data_list = [(podcast, podcast_episode_counts.get(podcast['name'], 0)) 
                            for podcast in podcasts[start_from_podcast:]]
        
        # Use ThreadPoolExecutor for parallel podcast downloads
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all podcast download tasks
            future_to_podcast = {
                executor.submit(self.download_podcast_episodes, podcast_data): podcast_data[0]['name']
                for podcast_data in podcast_data_list
            }
            
            # Process completed tasks
            for future in as_completed(future_to_podcast):
                podcast_name = future_to_podcast[future]
                try:
                    podcast_downloaded, total_episodes_in_podcast = future.result()
                    total_downloaded += podcast_downloaded
                    print(f"Completed {podcast_name}: {podcast_downloaded}/{total_episodes_in_podcast} episodes")
                except Exception as e:
                    print(f"Error processing {podcast_name}: {str(e)}")
        
        self.overall_progress.close()
        
        print(f"\n{'='*60}")
        print(f"Download complete! Total episodes downloaded: {total_downloaded}/{total_episodes}")
        print(f"Files saved to: {self.download_dir.absolute()}")
        
        # Generate summary report
        self.generate_download_report(total_downloaded, total_episodes)
    
    def generate_download_report(self, downloaded, total):
        """Generate a summary report of the download session"""
        report = {
            'download_date': datetime.now().isoformat(),
            'total_episodes_found': total,
            'total_episodes_downloaded': downloaded,
            'success_rate': f"{(downloaded/total*100):.1f}%" if total > 0 else "0%",
            'download_directory': str(self.download_dir.absolute())
        }
        
        report_path = self.download_dir / 'download_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"Download report saved to: {report_path}")

if __name__ == "__main__":
    # Configuration
    csv_file = "top_german_podcasts_full_updated.csv"
    download_dir = "data"
    rate_limit = 0.3  # seconds between requests
    
    # Create downloader
    downloader = PodcastDownloader(csv_file, download_dir, rate_limit)
    
    # Start download
    downloader.download_all_podcasts()