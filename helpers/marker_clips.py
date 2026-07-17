"""Marker-based clip extraction — the "pré-mâché" mode for streamers.

Takes a VOD + a JSON of stream markers (created live via /marker or the Helix
API, fetched by the caller with the user:read:broadcast scope) and produces
one clip per marker with a fixed window around it, plus a machine-readable
manifest for a host application (web app, script...).

Design decisions (validated 2026-07-15):
  - window = [position - 30s, position + 30s] (the streamer drops the marker
    during or just after the moment)
  - overlapping windows are MERGED into a single clip (multiple markers listed)
  - transcription runs ONLY inside each window (mlx-whisper on the audio
    slice) — no full-VOD transcription, keeps processing to minutes
  - clips are delivered as-is (source aspect/quality, frame-accurate cuts);
    no diarization, no subtitles, no LLM — zero tokens, no HF gate needed

Markers JSON format (list, seconds relative to VOD start):
[
  {"position_seconds": 3721, "description": "clutch 1v3", "id": "hx-123"},
  ...
]
`id` optional. Extra fields ignored (a raw Helix marker object works).

Output: <edit>/clips_markers/
  clip_{NN}_{slug}.mp4        one per (merged) marker window
  manifest.json               [{file, start, end, duration_s, markers:[...],
                               title, transcript}]

Usage:
    .venv/bin/python helpers/marker_clips.py <vod.mp4> --markers markers.json
        [--before 30] [--after 30] [--language fr] [--no-transcript]
        [--out-dir clips_markers] [--edit-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transcribe_local import DEFAULT_MODEL, whisper_to_scribe  # noqa: E402
from whisper_backends import BACKENDS, resolve_backend, transcribe_slice  # noqa: E402


def probe_duration(video: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def slug(text: str, max_len: int = 40) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return text[:max_len] or "marker"


def merge_windows(markers: list[dict], before: float, after: float,
                  vod_duration: float) -> list[dict]:
    """One window per marker, overlapping windows merged (sorted by time)."""
    markers = sorted(markers, key=lambda m: m["position_seconds"])
    clips: list[dict] = []
    for m in markers:
        pos = float(m["position_seconds"])
        start = max(0.0, pos - before)
        end = min(vod_duration, pos + after)
        entry = {"id": m.get("id"), "position_seconds": pos,
                 "description": (m.get("description") or "").strip()}
        if clips and start <= clips[-1]["end"]:
            clips[-1]["end"] = max(clips[-1]["end"], end)
            clips[-1]["markers"].append(entry)
        else:
            clips.append({"start": start, "end": end, "markers": [entry]})
    return clips


def transcribe_window(video: Path, start: float, end: float,
                      model: str, language: str | None,
                      backend: str) -> str:
    """Windowed transcription: extract the audio slice, run whisper on it."""
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "slice.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{start:.2f}", "-i", str(video),
             "-t", f"{end - start:.2f}", "-vn", "-ac", "1", "-ar", "16000",
             "-c:a", "pcm_s16le", str(wav)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return transcribe_slice(str(wav), model, language, backend)


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract clips around stream markers")
    ap.add_argument("video", type=Path)
    ap.add_argument("--markers", type=Path, required=True,
                    help="JSON list of {position_seconds, description, id}")
    ap.add_argument("--before", type=float, default=30.0,
                    help="Seconds kept before each marker (default 30)")
    ap.add_argument("--after", type=float, default=30.0,
                    help="Seconds kept after each marker (default 30)")
    ap.add_argument("--language", type=str, default=None,
                    help="ISO code for windowed transcription (default: auto)")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--backend", type=str, default="auto", choices=BACKENDS,
                    help="Transcription backend: mlx (Apple Silicon), "
                         "faster-whisper (portable), auto = mlx if available")
    ap.add_argument("--no-transcript", action="store_true",
                    help="Skip windowed transcription entirely")
    ap.add_argument("--edit-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=str, default="clips_markers")
    ap.add_argument("--crf", type=int, default=20)
    args = ap.parse_args()

    video = args.video.resolve()
    if not video.exists():
        sys.exit(f"video not found: {video}")
    markers = json.loads(args.markers.read_text())
    if not markers:
        sys.exit("markers JSON is empty")

    edit_dir = (args.edit_dir or (video.parent / "edit")).resolve()
    out_dir = edit_dir / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    backend = resolve_backend(args.backend) if not args.no_transcript else "none"
    if backend != "none":
        print(f"  transcription backend: {backend}")

    vod_duration = probe_duration(video)
    clips = merge_windows(markers, args.before, args.after, vod_duration)
    merged = len(markers) - len(clips)
    print(f"  {len(markers)} markers → {len(clips)} clips"
          + (f" ({merged} merged)" if merged else ""))

    manifest: list[dict] = []
    for i, c in enumerate(clips, 1):
        title = c["markers"][0]["description"] or f"marker {i}"
        name = f"clip_{i:02d}_{slug(title)}.mp4"
        dest = out_dir / name
        dur = c["end"] - c["start"]
        if not dest.exists():
            subprocess.run(
                ["ffmpeg", "-y", "-ss", f"{c['start']:.2f}", "-i", str(video),
                 "-t", f"{dur:.2f}", "-c:v", "libx264", "-preset", "veryfast",
                 "-crf", str(args.crf), "-c:a", "aac", "-b:a", "192k",
                 "-movflags", "+faststart", str(dest)],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        transcript = ""
        if not args.no_transcript:
            transcript = transcribe_window(video, c["start"], c["end"],
                                           args.model, args.language, backend)
        manifest.append({
            "file": name,
            "start": round(c["start"], 2),
            "end": round(c["end"], 2),
            "duration_s": round(dur, 2),
            "markers": c["markers"],
            "title": title,
            "transcript": transcript,
        })
        quote = (transcript[:60] + "…") if len(transcript) > 60 else transcript
        print(f"  {name} ({dur:.0f}s) — {quote or 'no transcript'}")

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\nmanifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
