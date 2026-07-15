"""Extract highlight clips from a VOD based on a candidates JSON.

Reads <edit>/clips_candidates.json (produced by the editorial selection pass,
see PIPELINE.md step 5), extracts each clip with frame-accurate re-encode and
fixed padding, computes the speaker share of each clip from the diarization
sidecar (if present), and writes <edit>/clips/clips.md as a human recap.

Candidates JSON format (list, sorted by descending score):
[
  {"start": 123.4, "end": 152.0, "quote": "...", "why": "...",
   "peak": "+33dB à 00:26:19" | null, "score": 1-10},
  ...
]

Output naming: clip_{rank:02d}_s{score}_{HHMMSS}.mp4  (HHMMSS = position in VOD)

Usage:
    .venv/bin/python helpers/extract_clips.py <video>
    .venv/bin/python helpers/extract_clips.py <video> --candidates my.json --pad-after 1.0
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def fmt_compact(t: float) -> str:
    s = int(t)
    return f"{s // 3600:02d}{s % 3600 // 60:02d}{s % 60:02d}"


def fmt_h(t: float) -> str:
    s = int(t)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def speaker_share(diar_segments: list[dict], start: float, end: float) -> str:
    tot: dict[str, float] = {}
    for seg in diar_segments:
        o = min(end, seg["end"]) - max(start, seg["start"])
        if o > 0:
            tot[seg["speaker"]] = tot.get(seg["speaker"], 0.0) + o
    total = sum(tot.values())
    if not total:
        return "aucune parole détectée"
    return ", ".join(f"{k} {v / total * 100:.0f}%"
                     for k, v in sorted(tot.items(), key=lambda x: -x[1]))


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract highlight clips from candidates JSON")
    ap.add_argument("video", type=Path)
    ap.add_argument("--edit-dir", type=Path, default=None)
    ap.add_argument("--candidates", type=Path, default=None,
                    help="Candidates JSON (default: <edit>/clips_candidates.json)")
    ap.add_argument("--pad-before", type=float, default=0.25,
                    help="Seconds added before each clip start (default 0.25)")
    ap.add_argument("--pad-after", type=float, default=0.75,
                    help="Seconds added after each clip end (default 0.75)")
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--out-dir", type=str, default="clips",
                    help="Output subdir inside <edit> (default: clips)")
    args = ap.parse_args()

    video = args.video.resolve()
    if not video.exists():
        sys.exit(f"video not found: {video}")
    edit_dir = (args.edit_dir or (video.parent / "edit")).resolve()
    cand_path = (args.candidates or (edit_dir / "clips_candidates.json")).resolve()
    if not cand_path.exists():
        sys.exit(f"candidates JSON not found: {cand_path}")

    candidates = json.loads(cand_path.read_text())
    diar_path = edit_dir / "diarization" / f"{video.stem}.json"
    diar_segments = (json.loads(diar_path.read_text())["segments"]
                     if diar_path.exists() else [])

    out_dir = edit_dir / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# Clips candidats — {video.name}", "",
             f"Padding: -{args.pad_before}s / +{args.pad_after}s autour des bornes.", ""]

    for i, c in enumerate(candidates, 1):
        start = max(0.0, c["start"] - args.pad_before)
        dur = (c["end"] + args.pad_after) - start
        name = f"clip_{i:02d}_s{c['score']}_{fmt_compact(c['start'])}.mp4"
        dest = out_dir / name
        if not dest.exists():
            subprocess.run([
                "ffmpeg", "-y", "-ss", f"{start:.2f}", "-i", str(video),
                "-t", f"{dur:.2f}", "-c:v", "libx264", "-preset", "veryfast",
                "-crf", str(args.crf), "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart", str(dest),
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        who = speaker_share(diar_segments, start, start + dur)
        print(f"  {name} ({dur:.0f}s) — {who}")
        lines += [f"## {i:02d}. [{fmt_h(c['start'])}] score {c['score']}/10 — `{name}` ({dur:.0f}s)",
                  f"> « {c['quote']} »", "",
                  f"- {c['why']}",
                  f"- Pic audio : {c.get('peak') or 'aucun'}",
                  f"- Voix : {who}", ""]

    (out_dir / "clips.md").write_text("\n".join(lines) + "\n")
    print(f"\nrecap: {out_dir / 'clips.md'}")


if __name__ == "__main__":
    main()
