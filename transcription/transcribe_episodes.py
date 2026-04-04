import os
import whisper
import torch
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import time

class PodcastTranscriber:
    def __init__(self, data_dir="data", model_size="base"):
        self.data_dir = Path(data_dir)
        self.model_size = model_size
        self.model = None
        self.supported_formats = {'.mp3', '.m4a', '.wav', '.flac', '.ogg'}
        
    def load_model(self):
        """Load the Whisper model"""
        if self.model is None:
            print(f"Loading Whisper model: {self.model_size}")
            print("This may take a few minutes on first run...")
            
            # Check if CUDA is available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")
            
            if device == "cpu":
                print("Warning: Running on CPU. This will be very slow.")
                print("For better performance, use a GPU-enabled environment.")
            
            self.model = whisper.load_model(self.model_size, device=device)
            print(f"Model loaded successfully!")
        
        return self.model
    
    def find_all_episodes(self):
        """Find all audio files in the data directory"""
        episodes = []
        
        if not self.data_dir.exists():
            print(f"Data directory {self.data_dir} does not exist!")
            return episodes
        
        print(f"Scanning for episodes in {self.data_dir}...")
        
        for podcast_dir in self.data_dir.iterdir():
            if podcast_dir.is_dir():
                for file_path in podcast_dir.iterdir():
                    if file_path.suffix.lower() in self.supported_formats:
                        transcript_path = file_path.with_suffix('.txt')
                        episodes.append({
                            'audio_file': file_path,
                            'transcript_file': transcript_path,
                            'podcast_name': podcast_dir.name,
                            'episode_name': file_path.stem,
                            'already_transcribed': transcript_path.exists()
                        })
        
        return episodes
    
    def transcribe_episode(self, episode):
        """Transcribe a single episode"""
        audio_file = episode['audio_file']
        transcript_file = episode['transcript_file']
        
        try:
            print(f"Transcribing: {episode['episode_name'][:60]}...")
            
            # Transcribe using Whisper
            result = self.model.transcribe(
                str(audio_file),
                language="de",  # German
                task="transcribe",
                verbose=False
            )
            
            # Save transcript
            with open(transcript_file, 'w', encoding='utf-8') as f:
                f.write(result["text"])
            
            # Save detailed results (with timestamps) as JSON
            json_file = transcript_file.with_suffix('.json')
            transcript_data = {
                'text': result["text"],
                'segments': result.get("segments", []),
                'language': result.get("language", "de"),
                'transcription_date': datetime.now().isoformat(),
                'model_used': self.model_size,
                'audio_file': str(audio_file),
                'audio_duration': result.get("duration", 0)
            }
            
            try:
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(transcript_data, f, ensure_ascii=False, indent=2)
            except UnicodeEncodeError:
                # Fallback for encoding issues
                with open(json_file, 'w', encoding='utf-8', errors='replace') as f:
                    json.dump(transcript_data, f, ensure_ascii=False, indent=2)
            
            return True, len(result["text"])
            
        except Exception as e:
            error_msg = f"Error transcribing {audio_file}: {str(e)}"
            print(error_msg)
            
            # Save error info
            error_file = transcript_file.with_suffix('.error')
            with open(error_file, 'w', encoding='utf-8') as f:
                f.write(f"Transcription failed: {error_msg}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            
            return False, 0
    
    def estimate_time(self, episodes_to_process):
        """Estimate transcription time"""
        if not episodes_to_process:
            return 0
        
        # Rough estimates based on typical performance
        if torch.cuda.is_available():
            # GPU: roughly real-time to 2x real-time
            minutes_per_hour = 45  # Conservative estimate
        else:
            # CPU: very slow, roughly 10-20x real-time
            minutes_per_hour = 600  # Very conservative
        
        # Assume average episode is 45 minutes
        total_audio_hours = len(episodes_to_process) * 0.75  # 45 minutes
        estimated_minutes = total_audio_hours * minutes_per_hour
        
        return estimated_minutes
    
    def transcribe_all(self, resume=True):
        """Transcribe all episodes"""
        episodes = self.find_all_episodes()
        
        if not episodes:
            print("No episodes found!")
            return
        
        print(f"Found {len(episodes)} episodes total")
        
        # Filter episodes that need transcription
        if resume:
            episodes_to_process = [ep for ep in episodes if not ep['already_transcribed']]
            already_done = len(episodes) - len(episodes_to_process)
            print(f"Already transcribed: {already_done}")
            print(f"Remaining to transcribe: {len(episodes_to_process)}")
        else:
            episodes_to_process = episodes
            print("Transcribing all episodes (overwriting existing)")
        
        if not episodes_to_process:
            print("All episodes already transcribed!")
            return
        
        # Estimate time
        estimated_minutes = self.estimate_time(episodes_to_process)
        if estimated_minutes > 60:
            print(f"Estimated transcription time: {estimated_minutes/60:.1f} hours")
        else:
            print(f"Estimated transcription time: {estimated_minutes:.0f} minutes")
        
        # Ask for confirmation
        device_type = "GPU" if torch.cuda.is_available() else "CPU"
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"\nUsing {device_type}: {gpu_name}")
        else:
            print(f"\nUsing {device_type} for transcription")
        
        print(f"\nProceeding with transcribing {len(episodes_to_process)} episodes automatically...")
        print("This will run continuously using the base model for optimal speed.")
        
        # Load model
        self.load_model()
        
        # Process episodes
        successful = 0
        failed = 0
        total_chars = 0
        
        progress_bar = tqdm(episodes_to_process, desc="Transcribing", unit="episodes")
        
        for episode in progress_bar:
            # Update progress bar with current episode
            episode_name = episode['episode_name'][:40]
            progress_bar.set_description(f"Transcribing: {episode_name}")
            
            success, char_count = self.transcribe_episode(episode)
            
            if success:
                successful += 1
                total_chars += char_count
            else:
                failed += 1
            
            # Update progress bar with stats
            progress_bar.set_postfix({
                'Success': successful,
                'Failed': failed,
                'Chars': f"{total_chars:,}"
            })
        
        progress_bar.close()
        
        # Final report
        print(f"\n{'='*60}")
        print(f"Transcription Complete!")
        print(f"Successfully transcribed: {successful}")
        print(f"Failed: {failed}")
        print(f"Total characters transcribed: {total_chars:,}")
        print(f"Transcripts saved to: {self.data_dir.absolute()}")
        
        # Generate summary report
        self.generate_transcription_report(successful, failed, total_chars)
    
    def generate_transcription_report(self, successful, failed, total_chars):
        """Generate a summary report"""
        report = {
            'transcription_date': datetime.now().isoformat(),
            'model_used': self.model_size,
            'device_used': "GPU" if torch.cuda.is_available() else "CPU",
            'successful_transcriptions': successful,
            'failed_transcriptions': failed,
            'total_characters': total_chars,
            'data_directory': str(self.data_dir.absolute())
        }
        
        report_path = self.data_dir / 'transcription_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"Transcription report saved to: {report_path}")
    
    def get_stats(self):
        """Get statistics about transcriptions"""
        episodes = self.find_all_episodes()
        
        if not episodes:
            print("No episodes found!")
            return
        
        total = len(episodes)
        transcribed = sum(1 for ep in episodes if ep['already_transcribed'])
        remaining = total - transcribed
        
        print(f"Transcription Statistics:")
        print(f"Total episodes: {total}")
        print(f"Already transcribed: {transcribed} ({transcribed/total*100:.1f}%)")
        print(f"Remaining: {remaining} ({remaining/total*100:.1f}%)")
        
        # Group by podcast
        by_podcast = {}
        for ep in episodes:
            podcast = ep['podcast_name']
            if podcast not in by_podcast:
                by_podcast[podcast] = {'total': 0, 'transcribed': 0}
            by_podcast[podcast]['total'] += 1
            if ep['already_transcribed']:
                by_podcast[podcast]['transcribed'] += 1
        
        print(f"\nBy Podcast:")
        for podcast, stats in by_podcast.items():
            pct = stats['transcribed']/stats['total']*100
            print(f"  {podcast}: {stats['transcribed']}/{stats['total']} ({pct:.1f}%)")

if __name__ == "__main__":
    # Configuration
    data_dir = "data"
    model_size = "base"  # Using base model for optimal speed
    
    transcriber = PodcastTranscriber(data_dir, model_size)
    
    # Show current stats
    transcriber.get_stats()
    
    # Start transcription
    transcriber.transcribe_all(resume=True)