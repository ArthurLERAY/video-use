"""Detect sustained loudness peaks in a VOD — free highlight-candidate signal.

Computes short-window RMS loudness over the whole audio, subtracts a rolling
median baseline (so music-on vs music-off sections compare fairly), and
picks the top sustained peaks with a minimum gap between them. Laughter,
shouting, and hype moments show up as strong positive lift.

Output: <edit>/peaks/<stem>.json (machine) and <stem>.md (human/LLM view,
HH:MM:SS timestamps). Feed the .md to the highlight-selection pass so the
LLM knows where to read the transcript closely.

Usage:
    .venv/bin/python helpers/audio_peaks.py <video> [--top 30] [--min-gap 45]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transcribe_local import cached_wav_path, extract_audio  # noqa: E402

SAMPLE_RATE = 16000
HOP_S = 0.5          # one RMS value per 0.5s
SMOOTH_S = 4.0       # smooth RMS over 4s (sustained peaks, not clicks)
BASELINE_S = 300.0   # rolling median baseline over 5 min


def fmt_ts(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def compute_rms_db(wav_path: Path) -> np.ndarray:
    """Chunked read → one RMS dB value per HOP_S seconds."""
    import soundfile as sf
    hop = int(HOP_S * SAMPLE_RATE)
    rms_values: list[np.ndarray] = []
    with sf.SoundFile(str(wav_path)) as f:
        if f.samplerate != SAMPLE_RATE:
            sys.exit(f"expected {SAMPLE_RATE}Hz wav, got {f.samplerate}Hz")
        while True:
            block = f.read(hop * 240, dtype="float32")  # 2-min blocks
            if len(block) == 0:
                break
            if block.ndim > 1:
                block = block.mean(axis=1)
            n = len(block) // hop
            if n == 0:
                break
            frames = block[: n * hop].reshape(n, hop)
            rms_values.append(np.sqrt((frames ** 2).mean(axis=1)))
    rms = np.concatenate(rms_values) if rms_values else np.array([])
    return 20 * np.log10(rms + 1e-8)


def find_peaks(db: np.ndarray, top: int, min_gap_s: float) -> list[dict]:
    from scipy.ndimage import median_filter, uniform_filter1d

    smooth = uniform_filter1d(db, size=max(1, int(SMOOTH_S / HOP_S)))
    baseline = median_filter(smooth, size=max(3, int(BASELINE_S / HOP_S)), mode="nearest")
    lift = smooth - baseline

    peaks: list[dict] = []
    work = lift.copy()
    gap = int(min_gap_s / HOP_S)
    for _ in range(top):
        i = int(np.argmax(work))
        score = float(work[i])
        if score <= 0.5:  # nothing meaningful left
            break
        # measure how long the lift stays above half the peak value
        half = score / 2
        lo = i
        while lo > 0 and lift[lo - 1] >= half:
            lo -= 1
        hi = i
        while hi < len(lift) - 1 and lift[hi + 1] >= half:
            hi += 1
        peaks.append({
            "time": round(i * HOP_S, 1),
            "timestamp": fmt_ts(i * HOP_S),
            "lift_db": round(score, 2),
            "sustain_s": round((hi - lo + 1) * HOP_S, 1),
        })
        work[max(0, i - gap): i + gap + 1] = -np.inf
    peaks.sort(key=lambda p: p["time"])
    return peaks


def main() -> None:
    ap = argparse.ArgumentParser(description="Detect loudness peaks in a VOD")
    ap.add_argument("video", type=Path)
    ap.add_argument("--edit-dir", type=Path, default=None)
    ap.add_argument("--top", type=int, default=30, help="Max peaks to report (default 30)")
    ap.add_argument("--min-gap", type=float, default=45.0,
                    help="Min seconds between two peaks (default 45)")
    args = ap.parse_args()

    video = args.video.resolve()
    if not video.exists():
        sys.exit(f"video not found: {video}")
    edit_dir = (args.edit_dir or (video.parent / "edit")).resolve()

    wav = cached_wav_path(video, edit_dir)
    if not wav.exists():
        print(f"  extracting audio from {video.name}", flush=True)
    extract_audio(video, wav)

    print("  computing loudness profile...", flush=True)
    db = compute_rms_db(wav)
    duration = len(db) * HOP_S
    peaks = find_peaks(db, args.top, args.min_gap)

    out_dir = edit_dir / "peaks"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{video.stem}.json").write_text(json.dumps({
        "duration_s": duration,
        "hop_s": HOP_S,
        "peaks": peaks,
    }, indent=2))

    lines = [f"# Loudness peaks — {video.name}",
             "",
             f"Duration {fmt_ts(duration)}. Lift = dB above the local 5-min baseline;",
             "sustain = how long it stays above half the peak. High lift + long",
             "sustain ≈ laughter / shouting / hype. Read the transcript around these.",
             ""]
    for rank, p in enumerate(sorted(peaks, key=lambda x: -x["lift_db"]), 1):
        lines.append(f"{rank:2d}. [{p['timestamp']}] lift {p['lift_db']:+.1f} dB, "
                     f"sustain {p['sustain_s']:.0f}s")
    (out_dir / f"{video.stem}.md").write_text("\n".join(lines) + "\n")

    print(f"  {len(peaks)} peaks → {out_dir / (video.stem + '.md')}")


if __name__ == "__main__":
    main()
