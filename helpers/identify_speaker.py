"""Identify the streamer's voice in a VOD by voice enrollment — not talk time.

Two subcommands:

  enroll  — compute a reference voice embedding (ECAPA-TDNN, SpeechBrain)
            from a clean sample of the target speaker and save it to .npy.
            The sample can be any audio/video file; crop with --start/--end.

  label   — diarize the VOD with pyannote (local, free, needs a Hugging Face
            token with pyannote model conditions accepted), match each
            detected speaker cluster to the reference by cosine similarity,
            and rewrite the transcript JSON speaker_ids: "streamer" for the
            matching cluster, "guest_1", "guest_2", ... for the others
            (ordered by talk time). Works regardless of how little the
            target speaker talks.

Usage:
    .venv/bin/python helpers/identify_speaker.py enroll ref_clip.mp4 \
        --start 12.0 --end 55.0 -o streamer_ref.npy

    .venv/bin/python helpers/identify_speaker.py label vod.mp4 \
        --reference streamer_ref.npy [--num-speakers 3] [--threshold 0.30]

HF token resolution: --hf-token flag, then HF_TOKEN / HUGGINGFACE_HUB_TOKEN
env vars, then HF_TOKEN=... line in .env at the repo root.

Prerequisites on huggingface.co (one-time, free):
  - accept conditions of pyannote/speaker-diarization-community-1
  - create an access token (read) at hf.co/settings/tokens
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from bisect import bisect_right
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transcribe_local import cached_wav_path, extract_audio  # noqa: E402

DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
EMBEDDING_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
SAMPLE_RATE = 16000


# ---------------------------------------------------------------- utilities

def load_hf_token(explicit: str | None = None) -> str | None:
    import os
    if explicit:
        return explicit
    for var in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        if os.environ.get(var):
            return os.environ[var]
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("HF_TOKEN") and "=" in line:
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def extract_to_temp_wav(source: Path, tmpdir: str) -> Path:
    """Extract any audio/video source to a mono 16kHz WAV in tmpdir."""
    dest = Path(tmpdir) / f"{source.stem}_16k.wav"
    cmd = [
        "ffmpeg", "-y", "-i", str(source),
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le",
        str(dest),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return dest


def read_wav(path: Path, start: float | None = None, end: float | None = None) -> np.ndarray:
    """Read a mono 16kHz WAV (optionally cropped) as float32 numpy array."""
    import soundfile as sf
    start_frame = int(start * SAMPLE_RATE) if start else 0
    stop_frame = int(end * SAMPLE_RATE) if end else None
    data, sr = sf.read(str(path), start=start_frame, stop=stop_frame, dtype="float32")
    if sr != SAMPLE_RATE:
        sys.exit(f"expected {SAMPLE_RATE}Hz wav, got {sr}Hz")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data


def get_embedding_model():
    from speechbrain.inference.speaker import EncoderClassifier
    savedir = Path.home() / ".cache" / "video-use" / "spkrec-ecapa-voxceleb"
    return EncoderClassifier.from_hparams(source=EMBEDDING_MODEL, savedir=str(savedir))


def embed(waveform: np.ndarray, classifier) -> np.ndarray:
    """L2-normalized ECAPA embedding of a mono 16kHz float32 waveform."""
    import torch
    with torch.no_grad():
        emb = classifier.encode_batch(torch.from_numpy(waveform).unsqueeze(0))
    vec = emb.squeeze().cpu().numpy().astype(np.float64)
    return vec / (np.linalg.norm(vec) + 1e-10)


# ------------------------------------------------------------------- enroll

def cmd_enroll(args: argparse.Namespace) -> None:
    source = args.source.resolve()
    if not source.exists():
        sys.exit(f"source not found: {source}")

    with tempfile.TemporaryDirectory() as tmp:
        wav = extract_to_temp_wav(source, tmp)
        audio = read_wav(wav, args.start, args.end)

    duration = len(audio) / SAMPLE_RATE
    if duration < 5:
        sys.exit(f"reference sample too short ({duration:.1f}s) — use at least 10s of clean speech")
    if duration < 15:
        print(f"warning: only {duration:.1f}s of reference audio; 30-60s is more robust")

    print(f"computing reference embedding from {duration:.1f}s of audio...")
    classifier = get_embedding_model()
    vec = embed(audio, classifier)

    out = args.output.resolve()
    np.save(out, vec)
    print(f"saved reference embedding: {out} (dim {vec.shape[0]})")


# -------------------------------------------------------------------- label

def run_diarization(wav_path: Path, token: str, num_speakers: int | None):
    """Run pyannote diarization; returns list of (start, end, cluster_label)."""
    import torch
    import soundfile as sf
    from pyannote.audio import Pipeline

    try:
        pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=token)
    except TypeError:
        pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, use_auth_token=token)
    if pipeline is None:
        sys.exit(
            f"could not load {DIARIZATION_MODEL} — make sure the HF token is valid and\n"
            "you accepted the conditions of\n"
            "pyannote/speaker-diarization-community-1 on huggingface.co"
        )

    # torchcodec audio decoding is broken in this env — preload in memory
    # (officially supported input format).
    data, sr = sf.read(str(wav_path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    audio_dict = {"waveform": torch.from_numpy(data).unsqueeze(0), "sample_rate": sr}

    kwargs = {"num_speakers": num_speakers} if num_speakers else {}

    def _run(device: str):
        pipeline.to(torch.device(device))
        return pipeline(audio_dict, **kwargs)

    t0 = time.time()
    try:
        if torch.backends.mps.is_available():
            print("  diarizing on MPS...", flush=True)
            annotation = _run("mps")
        else:
            raise RuntimeError("no MPS")
    except Exception as e:  # MPS op gaps → CPU fallback
        print(f"  MPS failed ({type(e).__name__}), falling back to CPU...", flush=True)
        annotation = _run("cpu")
    print(f"  diarization done in {time.time() - t0:.0f}s")

    # pyannote 4.x pipelines return a DiarizeOutput wrapper; the Annotation
    # lives in .speaker_diarization. 3.x returned the Annotation directly.
    if not hasattr(annotation, "itertracks"):
        annotation = annotation.speaker_diarization

    segments = [
        (float(seg.start), float(seg.end), str(label))
        for seg, _, label in annotation.itertracks(yield_label=True)
    ]
    segments.sort()
    return segments


def match_streamer(
    segments: list[tuple[float, float, str]],
    wav_path: Path,
    reference: np.ndarray,
    threshold: float,
) -> tuple[dict[str, str], dict[str, float]]:
    """Compare each diarization cluster to the reference embedding.

    Returns (cluster_label -> final name, cluster_label -> similarity).
    """
    classifier = get_embedding_model()

    by_cluster: dict[str, list[tuple[float, float]]] = {}
    for start, end, label in segments:
        by_cluster.setdefault(label, []).append((start, end))

    similarities: dict[str, float] = {}
    talk_time: dict[str, float] = {}
    for label, segs in by_cluster.items():
        talk_time[label] = sum(e - s for s, e in segs)
        # embed up to 60s from the longest segments (>=1s each)
        usable = sorted((s for s in segs if s[1] - s[0] >= 1.0),
                        key=lambda x: x[1] - x[0], reverse=True)
        embs, total = [], 0.0
        for s, e in usable:
            e = min(e, s + 20.0)  # cap one chunk at 20s
            audio = read_wav(wav_path, s, e)
            if len(audio) < SAMPLE_RATE:
                continue
            embs.append(embed(audio, classifier))
            total += e - s
            if total >= 60.0:
                break
        if not embs:
            similarities[label] = -1.0
            continue
        mean_emb = np.mean(embs, axis=0)
        mean_emb /= np.linalg.norm(mean_emb) + 1e-10
        similarities[label] = float(np.dot(mean_emb, reference))

    best = max(similarities, key=lambda k: similarities[k]) if similarities else None
    mapping: dict[str, str] = {}
    if best is not None and similarities[best] >= threshold:
        mapping[best] = "streamer"
    guests = sorted(
        (l for l in by_cluster if l not in mapping),
        key=lambda l: talk_time[l], reverse=True,
    )
    for i, label in enumerate(guests, 1):
        mapping[label] = f"guest_{i}"

    print("\n  cluster similarity report (threshold {:.2f}):".format(threshold))
    for label in sorted(by_cluster, key=lambda l: talk_time[l], reverse=True):
        marker = " ← STREAMER" if mapping.get(label) == "streamer" else ""
        print(f"    {label}: sim={similarities[label]:+.3f}  talk={talk_time[label]:.0f}s"
              f"  → {mapping[label]}{marker}")
    if "streamer" not in mapping.values():
        print("    WARNING: no cluster matched the reference above the threshold —"
              " all speakers labeled guest_N. Check the reference sample or lower"
              " --threshold.")
    return mapping, similarities


def relabel_transcript(transcript_path: Path,
                       segments: list[tuple[float, float, str]],
                       mapping: dict[str, str],
                       tolerance: float = 0.75) -> int:
    """Assign speaker_id to each word by its midpoint. Returns words labeled."""
    data = json.loads(transcript_path.read_text())
    starts = [s for s, _, _ in segments]

    def speaker_at(t: float) -> str | None:
        i = bisect_right(starts, t) - 1
        # check containing / nearest segments around i
        best, best_dist = None, tolerance
        for j in (i, i + 1, i - 1):
            if 0 <= j < len(segments):
                s, e, label = segments[j]
                if s <= t <= e:
                    return mapping.get(label)
                dist = min(abs(t - s), abs(t - e))
                if dist < best_dist:
                    best, best_dist = mapping.get(label), dist
        return best

    n = 0
    for w in data.get("words", []):
        if w.get("type") != "word":
            continue
        mid = (w["start"] + w["end"]) / 2
        spk = speaker_at(mid)
        w["speaker_id"] = spk
        if spk is not None:
            n += 1
    transcript_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return n


def cmd_label(args: argparse.Namespace) -> None:
    video = args.video.resolve()
    if not video.exists():
        sys.exit(f"video not found: {video}")
    reference_path = args.reference.resolve()
    if not reference_path.exists():
        sys.exit(f"reference embedding not found: {reference_path} — run enroll first")
    reference = np.load(reference_path)

    token = load_hf_token(args.hf_token)
    if not token:
        sys.exit(
            "no Hugging Face token found. Create one at hf.co/settings/tokens,\n"
            "accept the conditions of\n"
            "pyannote/speaker-diarization-community-1, then either export HF_TOKEN=...\n"
            "or add an HF_TOKEN=... line to .env at the repo root."
        )

    edit_dir = (args.edit_dir or (video.parent / "edit")).resolve()
    wav = cached_wav_path(video, edit_dir)
    if not wav.exists():
        print(f"  extracting audio from {video.name}", flush=True)
    extract_audio(video, wav)

    segments = run_diarization(wav, token, args.num_speakers)
    print(f"  {len(segments)} speech segments, "
          f"{len({l for _, _, l in segments})} speakers detected")

    mapping, similarities = match_streamer(segments, wav, reference, args.threshold)

    # sidecar with raw segments + report
    diar_dir = edit_dir / "diarization"
    diar_dir.mkdir(parents=True, exist_ok=True)
    sidecar = diar_dir / f"{video.stem}.json"
    sidecar.write_text(json.dumps({
        "model": DIARIZATION_MODEL,
        "similarities": similarities,
        "mapping": mapping,
        "segments": [
            {"start": s, "end": e, "speaker": mapping.get(l, l)}
            for s, e, l in segments
        ],
    }, indent=2))
    print(f"  saved diarization: {sidecar}")

    transcript = edit_dir / "transcripts" / f"{video.stem}.json"
    if transcript.exists():
        n = relabel_transcript(transcript, segments, mapping)
        print(f"  relabeled {n} words in {transcript.name}")
        print("  re-run pack_transcripts.py to refresh takes_packed.md")
    else:
        print(f"  note: no transcript at {transcript} — run transcribe_local.py,"
              " then re-run this command (diarization is cached in the sidecar)")


# --------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description="Enroll and identify the streamer's voice")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_e = sub.add_parser("enroll", help="Build a reference voice embedding")
    ap_e.add_argument("source", type=Path, help="Audio/video file with clean target speech")
    ap_e.add_argument("--start", type=float, default=None, help="Crop start (s)")
    ap_e.add_argument("--end", type=float, default=None, help="Crop end (s)")
    ap_e.add_argument("-o", "--output", type=Path, default=Path("streamer_ref.npy"))
    ap_e.set_defaults(func=cmd_enroll)

    ap_l = sub.add_parser("label", help="Diarize a VOD and label the streamer")
    ap_l.add_argument("video", type=Path)
    ap_l.add_argument("--reference", type=Path, required=True, help="Path to enroll output .npy")
    ap_l.add_argument("--edit-dir", type=Path, default=None)
    ap_l.add_argument("--num-speakers", type=int, default=None,
                      help="Number of speakers if known (improves diarization)")
    ap_l.add_argument("--threshold", type=float, default=0.30,
                      help="Min cosine similarity to accept the streamer match (default 0.30)")
    ap_l.add_argument("--hf-token", type=str, default=None)
    ap_l.set_defaults(func=cmd_label)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
