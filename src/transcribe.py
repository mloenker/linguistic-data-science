import json
import time
import subprocess
import re
from pathlib import Path
from datetime import datetime, timezone
from tqdm import tqdm
from faster_whisper import WhisperModel, BatchedInferencePipeline
import torch


class PodcastTranscriber:
    def __init__(self):
        config_path = Path(__file__).parent.parent / 'config.json'
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.data_dir = Path(config['DATA_DIR'])
        self.model_name = config['MODEL']
        self.model = None
        self.batched_model = None
        self.supported_formats = {'.mp3', '.m4a', '.wav', '.flac', '.ogg'}
    
    def sanitize_filename(self, filename):
        filename = re.sub(r'[<>:"/\\|?*{},]', '', filename)
        filename = filename.replace(' ', '_')
        filename = filename.lower()
        return filename[:200]
    
    def load_model(self):
        if self.model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if torch.cuda.is_available() else "int8"
            
            self.model = WhisperModel(
                self.model_name, 
                device=device, 
                compute_type=compute_type
            )
            self.batched_model = BatchedInferencePipeline(model=self.model)
    
    def get_audio_duration(self, audio_file):
        try:
            result = subprocess.run([
                'ffprobe', '-i', str(audio_file),
                '-show_entries', 'format=duration',
                '-v', 'quiet', '-of', 'csv=p=0'
            ], capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except:
            return 4500
    
    
    def transcribe_episode(self, episode, episode_pbar, overall_pbar):
        audio_file = episode['audio_file']
        metadata_file = episode['metadata_file']
        metadata = episode['metadata'].copy()
        
        start_time = time.time()
        audio_duration = self.get_audio_duration(audio_file)
        
        segments, info = self.batched_model.transcribe(
            str(audio_file),
            batch_size=8,
            beam_size=5,
            language="de",
            task="transcribe"
        )
        
        text_segments = []
        estimated_duration = getattr(info, 'duration', audio_duration)
        
        episode_pbar.reset(total=round(estimated_duration / 60))
        podcast_name = episode['metadata'].get('podcast_name', '')[:20]
        episode_title = episode['metadata'].get('title', audio_file.stem)[:16]
        episode_pbar.set_description(f"{podcast_name}: {episode_title}")
        
        last_progress_minutes = 0
        
        for segment in segments:
            text_segments.append(segment.text)
            
            current_minutes = round(segment.end / 60)
            if current_minutes > last_progress_minutes:
                update_amount = current_minutes - last_progress_minutes
                episode_pbar.update(update_amount)
                overall_pbar.refresh()
                last_progress_minutes = current_minutes
        
        remaining_minutes = round(estimated_duration / 60) - last_progress_minutes
        if remaining_minutes > 0:
            episode_pbar.update(remaining_minutes)
        
        text = "".join(segment.strip() + " " for segment in text_segments).strip()
        while text and text[0] in ',.;:!?-':
            text = text[1:].strip()
        if text:
            text = text[0].upper() + text[1:]
        
        transcript_file = audio_file.with_suffix('.txt')
        with open(transcript_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        transcription_time = time.time() - start_time
        
        metadata.update({
            'transcription_date': datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            'transcription_time': transcription_time,
            'transcription_finished': True,
            'transcription_model': self.model_name
        })
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        audio_file.unlink()
    
    def transcribe_all(self):
        all_episodes = []
        for metadata_file in self.data_dir.rglob('*.metadata'):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                audio_file = metadata_file.with_suffix('.mp3')
                for ext in self.supported_formats:
                    potential_audio = metadata_file.with_suffix(ext)
                    if potential_audio.exists():
                        audio_file = potential_audio
                        break
                
                if audio_file.exists() or metadata.get('transcription_finished', False):
                    all_episodes.append({
                        'audio_file': audio_file,
                        'metadata_file': metadata_file,
                        'metadata': metadata
                    })
            except:
                continue
        
        episodes = [ep for ep in all_episodes if not ep['metadata'].get('transcription_finished', False) and ep['audio_file'].exists()]
        
        if not episodes:
            print("No episodes to transcribe!")
            return
        
        total_remaining_duration = sum(ep['metadata'].get('audio_duration', 0) for ep in episodes)
        estimated_transcription_seconds = total_remaining_duration / 120
        
        hours = int(estimated_transcription_seconds // 3600)
        minutes = int((estimated_transcription_seconds % 3600) // 60)
        seconds = int(estimated_transcription_seconds % 60)
        
        print(f"Estimated remaining transcription time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        
        self.load_model()
        
        with tqdm(total=len(all_episodes), desc="Transcribing Episodes", unit="episode", initial=len(all_episodes) - len(episodes), position=0, colour="magenta") as overall_pbar:
            with tqdm(unit='min', position=1, leave=False) as episode_pbar:
                
                for episode in episodes:
                    try:
                        self.transcribe_episode(episode, episode_pbar, overall_pbar)
                        overall_pbar.update(1)
                    except Exception as e:
                        overall_pbar.write(f"Failed to transcribe {episode['audio_file'].name}: {e}")
                        raise


if __name__ == "__main__":
    transcriber = PodcastTranscriber()
    transcriber.transcribe_all()