"""Transcribe a video locally with mlx-whisper (Apple Silicon) — no API cost.

Drop-in replacement for transcribe.py (ElevenLabs Scribe): extracts mono
16kHz audio via ffmpeg, transcribes with word-level timestamps, and writes
Scribe-compatible JSON to <edit_dir>/transcripts/<video_stem>.json so the
rest of the pipeline (pack_transcripts.py, render.py --build-subtitles,
identify_speaker.py) works unchanged.

Speaker IDs are left null — run identify_speaker.py afterwards to label them.

Cached: if the output file already exists, transcription is skipped.
The extracted 16kHz WAV is cached at <edit_dir>/cache/<stem>_16k.wav and
reused by identify_speaker.py / audio_peaks.py.

Usage:
    .venv/bin/python helpers/transcribe_local.py <video_path>
    .venv/bin/python helpers/transcribe_local.py <video> --edit-dir /custom/edit
    .venv/bin/python helpers/transcribe_local.py <video> --language fr
    .venv/bin/python helpers/transcribe_local.py <video> --model mlx-community/whisper-tiny
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"


def cached_wav_path(video: Path, edit_dir: Path) -> Path:
    return edit_dir / "cache" / f"{video.stem}_16k.wav"


def extract_audio(video_path: Path, dest: Path) -> None:
    """Extract mono 16kHz PCM WAV. Cached: skipped if dest already exists."""
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp.wav")
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(tmp),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    tmp.rename(dest)


def whisper_to_scribe(result: dict) -> dict:
    """Convert an mlx-whisper result (word_timestamps=True) into the
    ElevenLabs Scribe JSON shape consumed by the rest of the pipeline:

    {"language_code": ..., "text": ..., "words": [
        {"text": "...", "start": s, "end": e, "type": "word", "speaker_id": null},
        {"text": " ", "start": s, "end": e, "type": "spacing"},   # gaps
        ...
    ]}
    """
    words_out: list[dict] = []
    prev_end: float | None = None

    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            text = (w.get("word") or "").strip()
            if not text:
                continue
            start = float(w["start"])
            end = float(w["end"])
            if end < start:
                end = start
            # whisper splits on apostrophes ("aujourd" + "'hui") — merge the
            # fragment into the previous word so subtitles stay readable
            if (text[0] in "'’" and words_out and words_out[-1]["type"] == "word"
                    and start - words_out[-1]["end"] < 0.15):
                words_out[-1]["text"] += text
                words_out[-1]["end"] = round(end, 3)
                prev_end = end
                continue
            if prev_end is not None and start > prev_end:
                words_out.append({
                    "text": " ",
                    "start": round(prev_end, 3),
                    "end": round(start, 3),
                    "type": "spacing",
                })
            words_out.append({
                "text": text,
                "start": round(start, 3),
                "end": round(end, 3),
                "type": "word",
                "speaker_id": None,
            })
            prev_end = end

    return {
        "language_code": result.get("language"),
        "text": (result.get("text") or "").strip(),
        "words": words_out,
    }


def transcribe_one(
    video: Path,
    edit_dir: Path,
    model: str = DEFAULT_MODEL,
    language: str | None = None,
    verbose: bool = True,
) -> Path:
    """Transcribe a single video. Returns path to transcript JSON.

    Cached: returns existing path immediately if the transcript already exists.
    """
    transcripts_dir = edit_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    out_path = transcripts_dir / f"{video.stem}.json"

    if out_path.exists():
        if verbose:
            print(f"cached: {out_path.name}")
        return out_path

    wav = cached_wav_path(video, edit_dir)
    if verbose and not wav.exists():
        print(f"  extracting audio from {video.name}", flush=True)
    extract_audio(video, wav)

    import mlx_whisper  # deferred: slow import

    if verbose:
        size_mb = wav.stat().st_size / (1024 * 1024)
        print(f"  transcribing {wav.name} ({size_mb:.1f} MB) with {model}", flush=True)

    t0 = time.time()
    decode_options: dict = {
        "word_timestamps": True,
        # VODs contain music/silence stretches; both options limit
        # hallucination loops there.
        "condition_on_previous_text": False,
        "hallucination_silence_threshold": 2.0,
        "verbose": False,  # tqdm progress bar
    }
    if language:
        decode_options["language"] = language

    try:
        result = mlx_whisper.transcribe(str(wav), path_or_hf_repo=model, **decode_options)
    except TypeError:
        # older mlx-whisper without hallucination_silence_threshold
        decode_options.pop("hallucination_silence_threshold", None)
        result = mlx_whisper.transcribe(str(wav), path_or_hf_repo=model, **decode_options)

    payload = whisper_to_scribe(result)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    dt = time.time() - t0

    if verbose:
        n_words = sum(1 for w in payload["words"] if w["type"] == "word")
        kb = out_path.stat().st_size / 1024
        print(f"  saved: {out_path.name} ({kb:.1f} KB) in {dt:.1f}s")
        print(f"    words: {n_words}, language: {payload.get('language_code')}")

    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Transcribe a video locally with mlx-whisper")
    ap.add_argument("video", type=Path, help="Path to video file")
    ap.add_argument(
        "--edit-dir",
        type=Path,
        default=None,
        help="Edit output directory (default: <video_parent>/edit)",
    )
    ap.add_argument(
        "--language",
        type=str,
        default=None,
        help="Optional ISO language code (e.g., 'fr'). Omit to auto-detect.",
    )
    ap.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"HF repo of the MLX whisper model (default: {DEFAULT_MODEL})",
    )
    args = ap.parse_args()

    video = args.video.resolve()
    if not video.exists():
        sys.exit(f"video not found: {video}")

    edit_dir = (args.edit_dir or (video.parent / "edit")).resolve()
    transcribe_one(video=video, edit_dir=edit_dir, model=args.model, language=args.language)


if __name__ == "__main__":
    main()
