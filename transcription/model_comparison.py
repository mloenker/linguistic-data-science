#!/usr/bin/env python3
"""
Model Comparison Tool for Transcription Quality Testing

This script tests different transcription models against ground truth text
and provides detailed accuracy metrics.
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
import difflib
import re

# Import existing transcription modules
from transcribe_episodes import PodcastTranscriber

# For additional metrics
try:
    from jiwer import wer, cer, mer, wil
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False
    print("Warning: jiwer not installed. Install with: pip install jiwer")
    print("Basic metrics will still be calculated.")

# For plotting
try:
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib")
    print("Plotting will be disabled.")

class TranscriptionModelTester:
    """Test different transcription models against ground truth"""
    
    def __init__(self, ground_truth_dir: str = "groundtruthcheck"):
        self.ground_truth_dir = Path(ground_truth_dir)
        self.audio_file = self.ground_truth_dir / "groundtruthaudio.mp3"
        self.truth_file = self.ground_truth_dir / "groundtruthtext.txt"
        self.results_dir = Path("model_comparison_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Load ground truth
        self.ground_truth = self._load_ground_truth()
        
    def _load_ground_truth(self) -> str:
        """Load and normalize ground truth text"""
        if not self.truth_file.exists():
            raise FileNotFoundError(f"Ground truth file not found: {self.truth_file}")
        
        with open(self.truth_file, 'r', encoding='utf-8') as f:
            truth = f.read().strip()
        
        return self._normalize_text(truth)
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        # Remove line numbers if present
        text = re.sub(r'^\s*\d+→', '', text, flags=re.MULTILINE)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove extra quotes and normalize punctuation
        text = text.replace('„', '"').replace('"', '"')
        text = text.replace('–', '-')
        
        return text.strip().lower()
    
    def test_model(self, model_name: str, model_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Test a single model and return metrics"""
        print(f"\nTesting model: {model_name}")
        print("=" * 50)
        
        if not self.audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {self.audio_file}")
        
        # Configure model
        config = model_config or {}
        transcriber = PodcastTranscriber(
            data_dir=str(self.ground_truth_dir.parent),
            model_size=config.get('model_size', model_name)
        )
        
        # Load model
        start_time = time.time()
        transcriber.load_model()
        load_time = time.time() - start_time
        
        # Sanitize model name for file paths (replace problematic characters)
        safe_model_name = model_name.replace('/', '_').replace('\\', '_').replace(':', '_')
        
        # Create episode-like structure for existing transcriber
        episode = {
            'audio_file': self.audio_file,
            'transcript_file': self.results_dir / f"{safe_model_name}_transcript.txt",
            'podcast_name': 'test',
            'episode_name': f'groundtruth_{safe_model_name}',
            'already_transcribed': False
        }
        
        # Transcribe
        print(f"Transcribing with {model_name}...")
        transcribe_start = time.time()
        success, char_count = transcriber.transcribe_episode(episode)
        transcribe_time = time.time() - transcribe_start
        
        if not success:
            return {
                'model_name': model_name,
                'success': False,
                'error': 'Transcription failed',
                'load_time': load_time,
                'transcribe_time': transcribe_time
            }
        
        # Load transcribed text
        with open(episode['transcript_file'], 'r', encoding='utf-8') as f:
            transcribed_text = f.read().strip()
        
        normalized_transcription = self._normalize_text(transcribed_text)
        
        # Calculate metrics
        metrics = self._calculate_metrics(self.ground_truth, normalized_transcription)
        
        result = {
            'model_name': model_name,
            'success': True,
            'load_time': load_time,
            'transcribe_time': transcribe_time,
            'character_count': len(transcribed_text),
            'ground_truth_length': len(self.ground_truth),
            'transcription_length': len(normalized_transcription),
            **metrics
        }
        
        # Save detailed results
        self._save_detailed_results(model_name, result, transcribed_text)
        
        return result
    
    def _calculate_metrics(self, truth: str, transcription: str) -> Dict[str, float]:
        """Calculate various accuracy metrics"""
        metrics = {}
        
        # Basic character-level accuracy
        char_accuracy = self._character_accuracy(truth, transcription)
        metrics['character_accuracy'] = char_accuracy
        
        # Word-level accuracy using difflib
        truth_words = truth.split()
        trans_words = transcription.split()
        
        # Calculate word accuracy using sequence matching
        matcher = difflib.SequenceMatcher(None, truth_words, trans_words)
        word_accuracy = matcher.ratio()
        metrics['word_accuracy'] = word_accuracy
        
        # If jiwer is available, calculate additional metrics
        if JIWER_AVAILABLE:
            try:
                # Word Error Rate (WER)
                wer_score = wer(truth, transcription)
                metrics['wer'] = wer_score
                metrics['word_accuracy_jiwer'] = 1 - wer_score
                
                # Character Error Rate (CER)
                cer_score = cer(truth, transcription)
                metrics['cer'] = cer_score
                metrics['character_accuracy_jiwer'] = 1 - cer_score
                
                # Match Error Rate (MER)
                mer_score = mer(truth, transcription)
                metrics['mer'] = mer_score
                
                # Word Information Lost (WIL)
                wil_score = wil(truth, transcription)
                metrics['wil'] = wil_score
                
            except Exception as e:
                print(f"Warning: Error calculating jiwer metrics: {e}")
        
        return metrics
    
    def _character_accuracy(self, truth: str, transcription: str) -> float:
        """Calculate character-level accuracy using edit distance"""
        # Simple character accuracy using difflib
        matcher = difflib.SequenceMatcher(None, truth, transcription)
        return matcher.ratio()
    
    def _save_detailed_results(self, model_name: str, metrics: Dict, transcription: str):
        """Save detailed results including the actual transcription"""
        detailed_result = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'metrics': metrics,
            'transcription': transcription,
            'ground_truth': self.ground_truth,
            'config': {
                'audio_file': str(self.audio_file),
                'ground_truth_file': str(self.truth_file)
            }
        }
        
        # Sanitize model name for file paths
        safe_model_name = model_name.replace('/', '_').replace('\\', '_').replace(':', '_')
        result_file = self.results_dir / f"{safe_model_name}_detailed_results.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_result, f, ensure_ascii=False, indent=2)
    
    def compare_models(self, models: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare multiple models and generate report"""
        print("Starting model comparison...")
        print(f"Ground truth audio: {self.audio_file}")
        print(f"Ground truth text length: {len(self.ground_truth)} characters")
        
        results = []
        for model_config in models:
            model_name = model_config.get('name', model_config.get('model_size', 'unknown'))
            try:
                result = self.test_model(model_name, model_config)
                results.append(result)
                print(f"✓ Completed: {model_name}")
            except Exception as e:
                error_result = {
                    'model_name': model_name,
                    'success': False,
                    'error': str(e)
                }
                results.append(error_result)
                print(f"✗ Failed: {model_name} - {e}")
        
        # Generate comparison report
        comparison = self._generate_comparison_report(results)
        
        # Save comparison report
        report_file = self.results_dir / f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
        
        return comparison
    
    def _generate_comparison_report(self, results: List[Dict]) -> Dict[str, Any]:
        """Generate a comprehensive comparison report"""
        successful_results = [r for r in results if r.get('success', False)]
        
        if not successful_results:
            return {
                'timestamp': datetime.now().isoformat(),
                'total_models': len(results),
                'successful_models': 0,
                'results': results,
                'error': 'No models completed successfully'
            }
        
        # Find best performing models
        best_char_accuracy = max(successful_results, key=lambda x: x.get('character_accuracy', 0))
        best_word_accuracy = max(successful_results, key=lambda x: x.get('word_accuracy', 0))
        fastest_transcription = min(successful_results, key=lambda x: x.get('transcribe_time', float('inf')))
        
        comparison_report = {
            'timestamp': datetime.now().isoformat(),
            'ground_truth_stats': {
                'character_count': len(self.ground_truth),
                'word_count': len(self.ground_truth.split()),
                'file_path': str(self.truth_file)
            },
            'total_models': len(results),
            'successful_models': len(successful_results),
            'failed_models': len(results) - len(successful_results),
            'best_performers': {
                'character_accuracy': {
                    'model': best_char_accuracy['model_name'],
                    'score': best_char_accuracy.get('character_accuracy', 0)
                },
                'word_accuracy': {
                    'model': best_word_accuracy['model_name'],
                    'score': best_word_accuracy.get('word_accuracy', 0)
                },
                'fastest_transcription': {
                    'model': fastest_transcription['model_name'],
                    'time_seconds': fastest_transcription.get('transcribe_time', 0)
                }
            },
            'detailed_results': results
        }
        
        return comparison_report
    
    def print_summary(self, comparison_report: Dict[str, Any]):
        """Print a readable summary of the comparison"""
        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY")
        print("="*60)
        
        print(f"Tested: {comparison_report['total_models']} models")
        print(f"Successful: {comparison_report['successful_models']} models")
        print(f"Failed: {comparison_report['failed_models']} models")
        
        if comparison_report['successful_models'] > 0:
            print(f"\nGround Truth: {comparison_report['ground_truth_stats']['word_count']} words")
            
            print("\nBest Performers:")
            best = comparison_report['best_performers']
            print(f"  Character Accuracy: {best['character_accuracy']['model']} ({best['character_accuracy']['score']:.3f})")
            print(f"  Word Accuracy: {best['word_accuracy']['model']} ({best['word_accuracy']['score']:.3f})")
            print(f"  Fastest: {best['fastest_transcription']['model']} ({best['fastest_transcription']['time_seconds']:.1f}s)")
            
            print("\nDetailed Results:")
            for result in comparison_report['detailed_results']:
                if result.get('success'):
                    char_acc = result.get('character_accuracy', 0)
                    word_acc = result.get('word_accuracy', 0)
                    time_taken = result.get('transcribe_time', 0)
                    print(f"  {result['model_name']:15} | Char: {char_acc:.3f} | Word: {word_acc:.3f} | Time: {time_taken:.1f}s")
                else:
                    print(f"  {result['model_name']:15} | FAILED: {result.get('error', 'Unknown error')}")
        
        print(f"\nResults saved to: {self.results_dir}")
    
    def plot_comparison_chart(self, comparison_report: Dict[str, Any]):
        """Create a comprehensive comparison chart with multiple metrics"""
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available. Skipping plot generation.")
            return
        
        successful_results = [r for r in comparison_report['detailed_results'] if r.get('success', False)]
        
        if not successful_results:
            print("No successful results to plot.")
            return
        
        # Extract data for plotting
        model_names = [r['model_name'] for r in successful_results]
        # Truncate long model names for better display
        display_names = [name.split('/')[-1] if '/' in name else name for name in model_names]
        
        # Metrics to plot - use only jiwer metrics if available, otherwise fallback to difflib
        if JIWER_AVAILABLE and any('wer' in r for r in successful_results):
            metrics = {
                'Word Accuracy': [r.get('word_accuracy_jiwer', 0) for r in successful_results],
                'Character Accuracy': [r.get('character_accuracy_jiwer', 0) for r in successful_results],
                'Word Error Rate (WER)': [r.get('wer', 0) for r in successful_results],
                'Character Error Rate (CER)': [r.get('cer', 0) for r in successful_results],
                'Transcription Time (s)': [r.get('transcribe_time', 0) for r in successful_results],
            }
        else:
            # Fallback to difflib metrics if jiwer not available
            metrics = {
                'Word Accuracy (difflib)': [r.get('word_accuracy', 0) for r in successful_results],
                'Character Accuracy (difflib)': [r.get('character_accuracy', 0) for r in successful_results],
                'Transcription Time (s)': [r.get('transcribe_time', 0) for r in successful_results],
            }
        
        # Create subplots - 2x2 layout for up to 4 metrics, or adjust as needed
        n_metrics = len(metrics)
        n_cols = 2
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
        fig.suptitle('Transcription Model Comparison', fontsize=16, fontweight='bold')
        
        # Flatten axes array for easier indexing
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        # Color palette
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
        
        for i, (metric_name, values) in enumerate(metrics.items()):
            ax = axes[i]
            
            # Special handling for different metric types
            if 'time' in metric_name.lower():
                bars = ax.bar(display_names, values, color=colors)
                ax.set_ylabel('Time (seconds)')
                ax.set_title(f'{metric_name} (Lower is Better)')
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                           f'{value:.1f}s', ha='center', va='bottom', fontsize=9)
            
            elif 'error rate' in metric_name.lower():  # Error rates (lower is better)
                bars = ax.bar(display_names, values, color=colors)
                ax.set_ylabel('Error Rate')
                ax.set_ylim(0, max(max(values) * 1.1, 0.1))  # Dynamic y-limit
                ax.set_title(f'{metric_name} (Lower is Better)')
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=9)
            
            else:  # Accuracy metrics (higher is better)
                bars = ax.bar(display_names, values, color=colors)
                ax.set_ylabel('Accuracy')
                ax.set_ylim(0, 1)
                ax.set_title(f'{metric_name} (Higher is Better)')
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=9)
            
            # Rotate x-axis labels for better readability
            ax.set_xticklabels(display_names, rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(metrics), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_file = self.results_dir / f"model_comparison_chart_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"\nComparison chart saved to: {plot_file}")
        
        # Also save as PDF for better quality
        pdf_file = self.results_dir / f"model_comparison_chart_{timestamp}.pdf"
        plt.savefig(pdf_file, bbox_inches='tight')
        print(f"High-quality PDF saved to: {pdf_file}")
        
        plt.show()
    
    def create_performance_summary_plot(self, comparison_report: Dict[str, Any]):
        """Create a summary plot showing accuracy vs speed tradeoff"""
        if not MATPLOTLIB_AVAILABLE:
            return
            
        successful_results = [r for r in comparison_report['detailed_results'] if r.get('success', False)]
        
        if not successful_results:
            return
            
        # Extract data - use jiwer metrics if available
        model_names = [r['model_name'] for r in successful_results]
        display_names = [name.split('/')[-1] if '/' in name else name for name in model_names]
        accuracies = [r.get('character_accuracy_jiwer', r.get('character_accuracy', 0)) for r in successful_results]
        times = [r.get('transcribe_time', 0) for r in successful_results]
        
        # Create scatter plot
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(times, accuracies, s=100, alpha=0.7, c=range(len(model_names)), cmap='viridis')
        
        # Add model name labels
        for i, (name, x, y) in enumerate(zip(display_names, times, accuracies)):
            plt.annotate(name, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        plt.xlabel('Transcription Time (seconds)')
        plt.ylabel('Character Accuracy')
        plt.title('Model Performance: Accuracy vs Speed Tradeoff')
        plt.grid(True, alpha=0.3)
        
        # Add quadrant labels
        plt.axhline(y=np.mean(accuracies), color='gray', linestyle='--', alpha=0.5)
        plt.axvline(x=np.mean(times), color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_file = self.results_dir / f"accuracy_vs_speed_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Accuracy vs Speed plot saved to: {plot_file}")
        
        plt.show()


def main():
    """Main function to run model comparison"""
    
    # Define models to test
    models_to_test = [
        {'name': 'tiny', 'model_size': 'tiny'},
        {'name': 'base', 'model_size': 'base'},
        {'name': 'small', 'model_size': 'small'},
        {'name': 'medium', 'model_size': 'medium'},
        {'name': 'large-v3', 'model_size': 'large-v3'},
        {'name': 'toby-v3turbo-de', 'model_size': 'TheTobyB/whisper-large-v3-turbo-german-ct2'},
        {'name': 'chola-v3turbo-de', 'model_size': 'TheChola/whisper-large-v3-turbo-german-faster-whisper'}
    ]
    
    # Initialize tester
    tester = TranscriptionModelTester()
    
    # Run comparison
    try:
        comparison_report = tester.compare_models(models_to_test)
        tester.print_summary(comparison_report)
        
        # Generate plots if matplotlib is available
        if MATPLOTLIB_AVAILABLE:
            print("\nGenerating comparison charts...")
            tester.plot_comparison_chart(comparison_report)
            tester.create_performance_summary_plot(comparison_report)
        
        print(f"\nComparison complete! Check {tester.results_dir} for detailed results.")
        
    except Exception as e:
        print(f"Error during comparison: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())