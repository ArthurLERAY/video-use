"""Detect the STREAMER's loudest moments (screams) → tight-clip candidates JSON.

Crosses the diarization sidecar (segments labeled "streamer" by
identify_speaker.py) with a short-window loudness-lift profile: for each
streamer speech segment, measure the max lift above the local 5-min baseline;
keep the strongest, well-separated moments. Zero LLM tokens.

Clip bounds are snapped to natural speech gaps from the word-level transcript
(last silence >=0.4s before the scream, first silence >=0.5s after), targeting
5-12s tight clips (context -> scream -> short reaction).

Output: <edit>/clips_screams_candidates.json in the extract_clips.py
candidates format, sorted by lift. Then run:

    .venv/bin/python helpers/extract_clips.py <video> \
        --candidates <edit>/clips_screams_candidates.json \
        --out-dir clips_screams --pad-before 0.15 --pad-after 0.5

Usage:
    .venv/bin/python helpers/scream_clips.py <video> [--top 12] [--min-lift 15]
        [--min-gap 30]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transcribe_local import cached_wav_path, extract_audio  # noqa: E402
from audio_peaks import compute_rms_db, fmt_ts, HOP_S  # noqa: E402

SMOOTH_S = 1.0       # short smoothing: screams are brief (vs 4s in audio_peaks)
BASELINE_S = 300.0


def lift_profile(wav: Path) -> np.ndarray:
    from scipy.ndimage import median_filter, uniform_filter1d
    db = compute_rms_db(wav)
    smooth = uniform_filter1d(db, size=max(1, int(SMOOTH_S / HOP_S)))
    baseline = median_filter(smooth, size=max(3, int(BASELINE_S / HOP_S)), mode="nearest")
    return smooth - baseline


def snap_bounds(words: list[dict], scream_t: float) -> tuple[float, float]:
    """Natural clip bounds around a scream: last gap >=0.4s within 6s before
    (scream_t - 1.5), first gap >=0.5s within 5s after (scream_t + 1.5)."""
    start = scream_t - 4.0
    end = scream_t + 3.0
    prev_end = None
    for w in words:
        if w.get("type") != "word":
            continue
        ws, we = w["start"], w["end"]
        if prev_end is not None:
            gap_lo, gap_hi = prev_end, ws
            gap = gap_hi - gap_lo
            if gap >= 0.4 and scream_t - 7.5 <= gap_hi <= scream_t - 1.5:
                start = ws  # start at the first word after the gap
            if gap >= 0.5 and scream_t + 1.5 <= gap_lo <= scream_t + 6.5 and end == scream_t + 3.0:
                end = gap_lo  # end at the last word before the gap
        prev_end = max(prev_end or 0.0, we)
    if end - start < 3.0:
        start, end = scream_t - 3.0, scream_t + 3.0
    return max(0.0, start), end


def quote_in(words: list[dict], start: float, end: float, max_words: int = 18) -> str:
    sel = [w["text"] for w in words
           if w.get("type") == "word" and start <= w["start"] <= end]
    txt = " ".join(sel[:max_words]) + ("…" if len(sel) > max_words else "")
    return txt.replace(" '", "'")


def main() -> None:
    ap = argparse.ArgumentParser(description="Streamer scream detector → candidates JSON")
    ap.add_argument("video", type=Path)
    ap.add_argument("--edit-dir", type=Path, default=None)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--min-lift", type=float, default=15.0,
                    help="Min dB lift above local baseline (default 15)")
    ap.add_argument("--min-gap", type=float, default=30.0,
                    help="Min seconds between two selected screams (default 30)")
    ap.add_argument("-o", "--output", type=Path, default=None)
    args = ap.parse_args()

    video = args.video.resolve()
    if not video.exists():
        sys.exit(f"video not found: {video}")
    edit_dir = (args.edit_dir or (video.parent / "edit")).resolve()

    diar_path = edit_dir / "diarization" / f"{video.stem}.json"
    if not diar_path.exists():
        sys.exit(f"no diarization sidecar at {diar_path} — run identify_speaker.py label first")
    segments = [s for s in json.loads(diar_path.read_text())["segments"]
                if s["speaker"] == "streamer"]
    if not segments:
        sys.exit("no 'streamer' segments in the diarization sidecar")

    tr_path = edit_dir / "transcripts" / f"{video.stem}.json"
    words = json.loads(tr_path.read_text()).get("words", []) if tr_path.exists() else []

    wav = cached_wav_path(video, edit_dir)
    extract_audio(video, wav)
    print("  computing loudness lift profile...", flush=True)
    lift = lift_profile(wav)

    # max lift inside each streamer segment
    moments: list[tuple[float, float]] = []  # (time, lift_db)
    for s in segments:
        i0, i1 = int(s["start"] / HOP_S), min(int(s["end"] / HOP_S) + 1, len(lift))
        if i1 <= i0:
            continue
        j = i0 + int(np.argmax(lift[i0:i1]))
        moments.append((j * HOP_S, float(lift[j])))

    # greedy top-N with min separation
    moments.sort(key=lambda m: -m[1])
    picked: list[tuple[float, float]] = []
    for t, lv in moments:
        if lv < args.min_lift or len(picked) >= args.top:
            break
        if all(abs(t - pt) >= args.min_gap for pt, _ in picked):
            picked.append((t, lv))

    lifts = [lv for _, lv in picked] or [args.min_lift]
    candidates = []
    for t, lv in picked:
        start, end = snap_bounds(words, t)
        score = int(round(np.interp(lv, [min(lifts), max(lifts) + 1e-6], [6, 10])))
        candidates.append({
            "start": round(start, 2), "end": round(end, 2),
            "quote": quote_in(words, start, end),
            "why": f"cri de la streameuse à {fmt_ts(t)}",
            "peak": f"+{lv:.1f}dB à {fmt_ts(t)}",
            "score": score,
        })
    candidates.sort(key=lambda c: -c["score"])

    out = args.output or (edit_dir / "clips_screams_candidates.json")
    out.write_text(json.dumps(candidates, indent=2, ensure_ascii=False))
    print(f"  {len(candidates)} scream moments → {out}")
    for c in candidates:
        print(f"    [{c['peak']}] {c['end']-c['start']:.1f}s — « {c['quote'][:70]} »")


if __name__ == "__main__":
    main()
