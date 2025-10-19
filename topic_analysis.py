"""
BERTopic Analysis for Podcast Transcripts
Analyzes topics across all podcast episodes using BERTopic with German-optimized embeddings.
"""

import os
import pickle
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import nltk


# Configuration
DATA_DIR = Path("../data/")
OUTPUT_DIR = Path("./outputs/")
OUTPUT_DIR.mkdir(exist_ok=True)

# Model configuration - fast German-optimized model
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def get_german_stopwords():
    """
    Download and retrieve German stopwords from NLTK.

    Returns:
        List of German stopwords
    """
    try:
        from nltk.corpus import stopwords
        # Try to use stopwords, download if not available
        try:
            german_stopwords = stopwords.words('german')
        except LookupError:
            print("Downloading NLTK stopwords...")
            nltk.download('stopwords', quiet=True)
            german_stopwords = stopwords.words('german')

        print(f"Loaded {len(german_stopwords)} German stopwords")
        return german_stopwords
    except Exception as e:
        print(f"Warning: Could not load German stopwords: {e}")
        return []


def load_transcripts() -> Tuple[List[str], pd.DataFrame]:
    """
    Load all podcast transcripts from the data directory.

    Returns:
        Tuple of (documents, metadata_df) where documents is a list of transcript texts
        and metadata_df contains podcast name and episode file information.
    """
    print("Loading podcast transcripts...")

    documents = []
    metadata = []

    # Get all podcast folders
    podcast_folders = sorted([f for f in DATA_DIR.iterdir() if f.is_dir()])

    # Progress bar for podcasts
    for podcast_folder in tqdm(podcast_folders, desc="Scanning podcasts", unit="podcast"):
        podcast_name = podcast_folder.name

        # Get all .txt files (transcripts) in this podcast folder
        txt_files = sorted(podcast_folder.glob("*.txt"))

        # Progress bar for episodes within each podcast (nested)
        for txt_file in tqdm(txt_files, desc=f"  {podcast_name}", unit="episode", leave=False):
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()

                    # Only include non-empty transcripts
                    if content:
                        documents.append(content)
                        metadata.append({
                            'podcast': podcast_name,
                            'episode_file': txt_file.name,
                            'word_count': len(content.split())
                        })
            except Exception as e:
                print(f"\nError reading {txt_file}: {e}")
                continue

    metadata_df = pd.DataFrame(metadata)

    print(f"\nLoaded {len(documents):,} transcripts from {len(podcast_folders)} podcasts")
    print(f"Total words: {metadata_df['word_count'].sum():,}")
    print(f"Average words per episode: {metadata_df['word_count'].mean():,.0f}")

    return documents, metadata_df


def perform_topic_modeling(documents: List[str]) -> BERTopic:
    """
    Perform BERTopic modeling on the documents.

    Args:
        documents: List of transcript texts

    Returns:
        Fitted BERTopic model
    """
    print(f"\n{'='*60}")
    print("Starting BERTopic Analysis")
    print(f"{'='*60}\n")

    # Detect best available device
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    print(f"Using device: {device}")

    # Get German stopwords
    print("\nLoading German stopwords...")
    german_stopwords = get_german_stopwords()

    # Initialize embedding model
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=device)

    # Generate embeddings with progress bar
    print("\nGenerating embeddings for all documents...")
    embeddings = embedding_model.encode(
        documents,
        show_progress_bar=True,
        batch_size=32
    )

    # Configure CountVectorizer with German stopwords
    print("\nConfiguring text vectorization with stopword filtering...")
    vectorizer_model = CountVectorizer(
        stop_words=german_stopwords,
        ngram_range=(1, 2),  # Use unigrams and bigrams
        min_df=5,            # Ignore words that appear in fewer than 5 documents
        max_df=0.7,          # Ignore words that appear in more than 70% of documents
    )

    # Initialize BERTopic model with optimized parameters
    print("Initializing BERTopic model...")
    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        verbose=True,
        calculate_probabilities=True,
        min_topic_size=10,  # Minimum documents per topic
        nr_topics="auto"     # Automatically determine optimal number of topics
    )

    # Fit the model
    print("\nFitting BERTopic model (this may take a while)...")
    topics, probabilities = topic_model.fit_transform(documents, embeddings)

    print("\nTopic modeling complete!")

    return topic_model, topics, probabilities


def save_model_and_results(
    topic_model: BERTopic,
    topics: List[int],
    probabilities: np.ndarray,
    metadata_df: pd.DataFrame
):
    """
    Save the trained model and results to disk.

    Args:
        topic_model: Fitted BERTopic model
        topics: Topic assignments for each document
        probabilities: Topic probabilities for each document
        metadata_df: Metadata for each document
    """
    print(f"\n{'='*60}")
    print("Saving Results")
    print(f"{'='*60}\n")

    # Save the trained model
    model_path = OUTPUT_DIR / "bertopic_model.pkl"
    print(f"Saving trained model to {model_path}...")
    with open(model_path, 'wb') as f:
        pickle.dump(topic_model, f)
    print(f"✓ Model saved ({model_path.stat().st_size / 1e6:.1f} MB)")

    # Save topic information
    topic_info = topic_model.get_topic_info()
    topic_info_path = OUTPUT_DIR / "topic_info.csv"
    print(f"\nSaving topic information to {topic_info_path}...")
    topic_info.to_csv(topic_info_path, index=False)
    print(f"✓ Topic info saved ({len(topic_info)} topics)")

    # Save document-topic assignments with metadata
    results_df = metadata_df.copy()
    results_df['topic'] = topics
    results_df['topic_probability'] = probabilities.max(axis=1) if len(probabilities.shape) > 1 else probabilities

    results_path = OUTPUT_DIR / "document_topics.csv"
    print(f"\nSaving document-topic assignments to {results_path}...")
    results_df.to_csv(results_path, index=False)
    print(f"✓ Document assignments saved ({len(results_df):,} documents)")

    print(f"\nAll results saved to: {OUTPUT_DIR.absolute()}")


def print_statistics(
    topic_model: BERTopic,
    topics: List[int],
    metadata_df: pd.DataFrame
):
    """
    Print comprehensive statistics about the topic modeling results.

    Args:
        topic_model: Fitted BERTopic model
        topics: Topic assignments for each document
        metadata_df: Metadata for each document
    """
    print(f"\n{'='*60}")
    print("TOPIC MODELING STATISTICS")
    print(f"{'='*60}\n")

    # Basic statistics
    topic_info = topic_model.get_topic_info()
    n_topics = len(topic_info) - 1  # Exclude outlier topic (-1)
    n_outliers = (np.array(topics) == -1).sum()

    print(f"Total Documents Analyzed: {len(topics):,}")
    print(f"Number of Topics Discovered: {n_topics}")
    print(f"Outliers (Topic -1): {n_outliers:,} ({n_outliers/len(topics)*100:.1f}%)")

    # Topic size distribution
    print(f"\n{'─'*60}")
    print("Topic Size Distribution:")
    print(f"{'─'*60}")

    topic_counts = topic_info[topic_info['Topic'] != -1]['Count']
    print(f"  Mean topic size: {topic_counts.mean():.0f} documents")
    print(f"  Median topic size: {topic_counts.median():.0f} documents")
    print(f"  Min topic size: {topic_counts.min()} documents")
    print(f"  Max topic size: {topic_counts.max():,} documents")

    # Top 15 topics
    print(f"\n{'─'*60}")
    print("Top 15 Topics by Size:")
    print(f"{'─'*60}\n")

    top_topics = topic_info[topic_info['Topic'] != -1].head(15)
    for idx, row in top_topics.iterrows():
        topic_id = row['Topic']
        count = row['Count']

        # Get representative words for this topic
        topic_words = topic_model.get_topic(topic_id)
        if topic_words:
            keywords = ", ".join([word for word, score in topic_words[:5]])
        else:
            keywords = "N/A"

        print(f"Topic {topic_id:3d}: {count:5,} docs | {keywords}")

    # Per-podcast topic distribution
    print(f"\n{'─'*60}")
    print("Topics per Podcast:")
    print(f"{'─'*60}\n")

    results_df = metadata_df.copy()
    results_df['topic'] = topics

    podcast_topic_stats = results_df.groupby('podcast')['topic'].agg([
        ('episodes', 'count'),
        ('unique_topics', lambda x: len(set(x) - {-1})),
        ('outlier_pct', lambda x: (x == -1).sum() / len(x) * 100)
    ]).sort_values('unique_topics', ascending=False)

    print(f"{'Podcast':<30} {'Episodes':>10} {'Unique Topics':>15} {'Outliers %':>12}")
    print(f"{'-'*30} {'-'*10} {'-'*15} {'-'*12}")

    for podcast, row in podcast_topic_stats.head(10).iterrows():
        print(f"{podcast:<30} {row['episodes']:>10,} {row['unique_topics']:>15} {row['outlier_pct']:>11.1f}%")

    print(f"\n{len(podcast_topic_stats)} podcasts analyzed (showing top 10)")

    print(f"\n{'='*60}")


def main():
    """Main execution function."""
    print(f"\n{'#'*60}")
    print("BERTOPIC PODCAST ANALYSIS")
    print(f"{'#'*60}\n")

    # Load transcripts
    documents, metadata_df = load_transcripts()

    if len(documents) == 0:
        print("Error: No documents loaded. Please check the data directory.")
        return

    # Perform topic modeling
    topic_model, topics, probabilities = perform_topic_modeling(documents)

    # Save results
    save_model_and_results(topic_model, topics, probabilities, metadata_df)

    # Print statistics
    print_statistics(topic_model, topics, metadata_df)

    print(f"\n{'#'*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
