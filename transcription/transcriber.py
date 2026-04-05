"""
transcriber.py  –  Transcribe all downloaded episodes using faster-whisper.

Reads every .json metadata file under data/, transcribes the matching audio
file if not yet done, writes a .txt transcript.

Model choice:
    large-v3  →  best quality,  needs a GPU  (recommended)
    medium    →  good quality,  works on CPU (slower)
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from faster_whisper import WhisperModel
from tqdm import tqdm
from google.colab import drive

# ── Configuration ──────────────────────────────────────────────────────────────

drive.mount('/content/drive')

DATA_DIR   = Path("/content/drive/MyDrive/podcast_data")
MODEL_NAME = "large-v3"          # change to "medium" if you only have CPU
LANGUAGE   = "de"                # German

# ──────────────────────────────────────────────────────────────────────────────

SUPPORTED_AUDIO = {".mp3", ".m4a", ".wav", ".flac", ".ogg"}


def load_model() -> WhisperModel:
    device       = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"Loading {MODEL_NAME} on {device} ({compute_type}) …")
    return WhisperModel(MODEL_NAME, device=device, compute_type=compute_type)


def find_audio_file(meta_path: Path):
    """Return the audio file that belongs to this metadata file, or None."""
    for ext in SUPPORTED_AUDIO:
        candidate = meta_path.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def collect_pending() -> list[dict]:
    """Return all episodes that still need transcription."""
    pending = []
    for meta_path in sorted(DATA_DIR.rglob("*.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if meta.get("transcription_finished"):
            continue
        audio = find_audio_file(meta_path)
        if audio:
            pending.append({"audio": audio, "meta_path": meta_path, "meta": meta})
    return pending


def transcribe_episode(episode: dict, model: WhisperModel) -> None:
    audio     = episode["audio"]
    meta_path = episode["meta_path"]
    meta      = episode["meta"].copy()

    t0 = time.time()

    segments, _ = model.transcribe(
        str(audio),
        language=LANGUAGE,
        beam_size=5,
        task="transcribe",
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()

    # Basic cleanup: remove leading punctuation, capitalise first letter
    while text and text[0] in ",.;:!?-":
        text = text[1:].strip()
    if text:
        text = text[0].upper() + text[1:]

    # Write transcript
    transcript_path = audio.with_suffix(".txt")
    transcript_path.write_text(text, encoding="utf-8")

    # Update metadata
    meta.update({
        "transcription_finished": True,
        "transcription_model":    MODEL_NAME,
        "transcription_date":     datetime.now(timezone.utc).isoformat(),
        "transcription_time":     round(time.time() - t0, 1),
    })
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    pending = collect_pending()
    if not pending:
        print("Nothing to transcribe – all episodes are already done.")
        return

    print(f"Episodes to transcribe: {len(pending)}")
    model = load_model()

    with tqdm(pending, desc="Transcribing", unit="ep") as bar:
        for episode in bar:
            title = episode["meta"].get("title", episode["audio"].stem)[:55]
            bar.set_description(f"{title}")
            try:
                transcribe_episode(episode, model)
            except Exception as e:
                tqdm.write(f"[!] Failed – {episode['audio'].name}: {e}")

    print("\nDone. Transcripts are saved as .txt files next to each .json metadata file.")


if __name__ == "__main__":
    main()