"""Compose a vertical (1080x1920) montage from a 16:9 gaming VOD.

Layout: facecam band on top (1080 x top_h), gameplay crop below
(1080 x 1920-top_h). The facecam position can differ per segment (streamers
move their cam overlay mid-stream) — each segment names a cam preset defined
in the spec. Implements the video-use hard rules:

  - per-segment extract -> lossless -c copy concat (rule 2)
  - 30ms audio fades at every segment boundary (rule 3)
  - master SRT on output-timeline offsets via render.build_master_srt (rule 5)
  - subtitles burned LAST (rule 1), punctuation-only cues stripped

Spec JSON format (see PIPELINE.md step 6 for how to measure the crops):
{
  "source": "/abs/path/vod.mp4",
  "output": "tiktok_v1.mp4",                  // relative to <edit>
  "top_h": 600,                               // facecam band height (px)
  "cam_crops": {                              // ffmpeg crop w:h:x:y strings,
    "left":  "464:258:0:87",                  // aspect MUST equal 1080/top_h
    "right": "468:260:1452:0"
  },
  "game_crop": "884:1080:518:0",              // aspect MUST equal 1080/(1920-top_h)
  "segments": [                               // word-boundary bounds + padding
    {"start": 7248.61, "end": 7260.74, "cam": "right"},
    {"start": 1167.23, "end": 1182.82, "cam": "left"}
  ]
}

Usage:
    .venv/bin/python helpers/compose_vertical.py <spec.json> [--edit-dir DIR]
        [--no-subtitles] [--crf 19]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render import build_master_srt, SUB_FORCE_STYLE  # noqa: E402

OUT_W, OUT_H = 1080, 1920


def run_ffmpeg(cmd: list[str]) -> None:
    subprocess.run(["ffmpeg", "-y", *cmd], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def check_aspect(name: str, crop: str, target_w: int, target_h: int) -> None:
    w, h = (int(x) for x in crop.split(":")[:2])
    got, want = w / h, target_w / target_h
    if abs(got - want) / want > 0.03:
        print(f"  warning: crop '{name}' aspect {got:.3f} differs from "
              f"target {want:.3f} (>3%) — image will be distorted")


def strip_punct_only_cues(srt_path: Path) -> int:
    """Whisper emits standalone punctuation tokens; drop cues with no word char."""
    blocks = srt_path.read_text().strip().split("\n\n")
    kept = [b.splitlines() for b in blocks
            if re.search(r"\w", " ".join(b.splitlines()[2:]))]
    srt_path.write_text("\n\n".join(
        "\n".join([str(i)] + lines[1:]) for i, lines in enumerate(kept, 1)) + "\n")
    return len(blocks) - len(kept)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compose vertical montage from spec JSON")
    ap.add_argument("spec", type=Path)
    ap.add_argument("--edit-dir", type=Path, default=None,
                    help="Edit dir (default: <source_parent>/edit)")
    ap.add_argument("--no-subtitles", action="store_true")
    ap.add_argument("--crf", type=int, default=19)
    args = ap.parse_args()

    spec = json.loads(args.spec.read_text())
    src = Path(spec["source"]).resolve()
    if not src.exists():
        sys.exit(f"source not found: {src}")
    edit_dir = (args.edit_dir or (src.parent / "edit")).resolve()
    top_h = int(spec.get("top_h", 600))
    game_h = OUT_H - top_h
    game_crop = spec["game_crop"]

    check_aspect("game_crop", game_crop, OUT_W, game_h)
    for name, crop in spec["cam_crops"].items():
        check_aspect(f"cam_crops.{name}", crop, OUT_W, top_h)

    out_name = spec.get("output", "montage_vertical.mp4")
    work = edit_dir / (Path(out_name).stem + "_work")
    work.mkdir(parents=True, exist_ok=True)

    # 1. per-segment extract with vstack composition + 30ms audio fades
    parts: list[Path] = []
    for i, s in enumerate(spec["segments"], 1):
        dur = s["end"] - s["start"]
        seg_out = work / f"seg_{i}.mp4"
        parts.append(seg_out)
        cam_crop = spec["cam_crops"][s["cam"]]
        fc = (f"[0:v]split=2[a][b];"
              f"[a]crop={cam_crop},scale={OUT_W}:{top_h}[cam];"
              f"[b]crop={game_crop},scale={OUT_W}:{game_h}[game];"
              f"[cam][game]vstack[v]")
        af = f"afade=t=in:st=0:d=0.03,afade=t=out:st={dur - 0.03:.3f}:d=0.03"
        run_ffmpeg([
            "-ss", f"{s['start']:.3f}", "-i", str(src), "-t", f"{dur:.3f}",
            "-filter_complex", fc, "-map", "[v]", "-map", "0:a", "-af", af,
            "-c:v", "libx264", "-preset", "medium", "-crf", str(args.crf),
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            str(seg_out)])
        print(f"  seg_{i} ({dur:.1f}s, cam {s['cam']}) done")

    # 2. lossless concat
    concat_list = work / "list.txt"
    concat_list.write_text("".join(f"file '{p}'\n" for p in parts))
    base = work / "base.mp4"
    run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(concat_list), "-c", "copy", str(base)])
    print("  concat done")

    final = edit_dir / out_name
    if args.no_subtitles:
        run_ffmpeg(["-i", str(base), "-c", "copy", "-movflags", "+faststart", str(final)])
    else:
        # 3. master SRT on output timeline, punct-only cues stripped
        edl = {"sources": {src.stem: str(src)},
               "ranges": [{"source": src.stem, "start": s["start"], "end": s["end"]}
                          for s in spec["segments"]]}
        srt = edit_dir / (Path(out_name).stem + ".srt")
        build_master_srt(edl, edit_dir, srt)
        dropped = strip_punct_only_cues(srt)
        if dropped:
            print(f"  stripped {dropped} punctuation-only cues")
        # 4. subtitles LAST
        run_ffmpeg(["-i", str(base),
                    "-vf", f"subtitles={srt}:force_style='{SUB_FORCE_STYLE}'",
                    "-c:v", "libx264", "-preset", "medium", "-crf", str(args.crf),
                    "-c:a", "copy", "-movflags", "+faststart", str(final)])

    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "csv=p=0", str(final)], capture_output=True, text=True)
    expected = sum(s["end"] - s["start"] for s in spec["segments"])
    print(f"final: {final} ({float(probe.stdout):.1f}s, expected {expected:.1f}s)")


if __name__ == "__main__":
    main()
