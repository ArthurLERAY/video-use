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
    no diarization, no burned-in subtitles, no LLM — zero tokens, no HF gate
    needed. A sidecar .srt (clip-relative cues, from the same whisper pass) is
    written alongside each clip for the host app to toggle/burn as it wishes.

Markers JSON format (list, seconds relative to VOD start):
[
  {"position_seconds": 3721, "description": "clutch 1v3", "id": "hx-123"},
  ...
]
`id` optional. Extra fields ignored (a raw Helix marker object works).

Output: <edit>/clips_markers/
  clip_{NN}_{slug}.mp4        one per (merged) marker window
  clip_{NN}_{slug}.srt        SRT for that clip — timestamps relative to the
                              clip start (clip = 0), one cue per whisper
                              segment; omitted with --no-transcript / when empty
  manifest.json               [{file, start, end, duration_s, markers:[...],
                               title, transcript, srt?}]   (srt is additive)

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
from whisper_backends import BACKENDS, resolve_backend, transcribe_slice_segments  # noqa: E402


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


def _srt_timestamp(seconds: float) -> str:
    """SRT time code HH:MM:SS,mmm (mirrors render._srt_timestamp)."""
    total_ms = max(0, int(round(seconds * 1000)))
    h, rem = divmod(total_ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def segments_to_srt(segments: list[dict]) -> str:
    """Standard SRT from whisper segments (times already clip-relative).

    One cue per non-empty segment — no merge/split — blocks separated by a
    blank line, cue index recomputed so skipped empty segments leave no gap.
    Returns "" when there is nothing to write.
    """
    blocks: list[str] = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start = _srt_timestamp(seg.get("start", 0.0))
        end = _srt_timestamp(seg.get("end", 0.0))
        blocks.append(f"{len(blocks) + 1}\n{start} --> {end}\n{text}")
    return "\n\n".join(blocks) + "\n" if blocks else ""


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
                      backend: str) -> list[dict]:
    """Windowed transcription: extract the audio slice, run whisper on it.

    Returns the whisper segments (times clip-relative — the slice begins at 0),
    from which both the flat transcript and the sidecar SRT are built.
    """
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "slice.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{start:.2f}", "-i", str(video),
             "-t", f"{end - start:.2f}", "-vn", "-ac", "1", "-ar", "16000",
             "-c:a", "pcm_s16le", str(wav)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return transcribe_slice_segments(str(wav), model, language, backend)


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
        segments: list[dict] = []
        if not args.no_transcript:
            segments = transcribe_window(video, c["start"], c["end"],
                                         args.model, args.language, backend)
        transcript = " ".join(s["text"] for s in segments if s.get("text")).strip()

        entry = {
            "file": name,
            "start": round(c["start"], 2),
            "end": round(c["end"], 2),
            "duration_s": round(dur, 2),
            "markers": c["markers"],
            "title": title,
            "transcript": transcript,
        }

        # Sidecar SRT next to the clip — clip-relative cues, one per whisper
        # segment. Additive manifest field; manifests without it stay valid.
        srt_text = segments_to_srt(segments)
        if srt_text:
            srt_name = Path(name).with_suffix(".srt").name
            (out_dir / srt_name).write_text(srt_text, encoding="utf-8")
            entry["srt"] = srt_name

        manifest.append(entry)
        quote = (transcript[:60] + "…") if len(transcript) > 60 else transcript
        print(f"  {name} ({dur:.0f}s) — {quote or 'no transcript'}")

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\nmanifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
