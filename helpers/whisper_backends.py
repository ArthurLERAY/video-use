"""Portable windowed-transcription backends for marker_clips.py.

mlx-whisper runs only on Apple Silicon (MLX framework). On Windows/Linux the
same slice-level transcription is served by faster-whisper (CTranslate2,
CPU int8 by default, CUDA when available). Backend "auto" picks mlx when the
module is importable, faster-whisper otherwise — callers never need to know
which platform they are on.

Model names are expressed in mlx repo form everywhere (the historical
default, e.g. "mlx-community/whisper-large-v3-turbo") and mapped to the
equivalent CTranslate2 model for faster-whisper ("large-v3-turbo", needs
faster-whisper >= 1.1).
"""

from __future__ import annotations

import importlib.util

BACKENDS = ("auto", "mlx", "faster-whisper")

_MLX_TO_CT2 = {
    "mlx-community/whisper-large-v3-turbo": "large-v3-turbo",
    "mlx-community/whisper-large-v3": "large-v3",
    "mlx-community/whisper-medium": "medium",
    "mlx-community/whisper-small": "small",
    "mlx-community/whisper-base": "base",
    "mlx-community/whisper-tiny": "tiny",
}


def resolve_backend(requested: str = "auto") -> str:
    if requested not in BACKENDS:
        raise ValueError(f"unknown backend: {requested!r} (expected one of {BACKENDS})")
    if requested != "auto":
        return requested
    return "mlx" if importlib.util.find_spec("mlx_whisper") else "faster-whisper"


def ct2_model_name(model: str) -> str:
    """Map an mlx-community repo name to its CTranslate2 equivalent."""
    if model in _MLX_TO_CT2:
        return _MLX_TO_CT2[model]
    if model.startswith("mlx-community/whisper-"):
        return model.split("whisper-", 1)[1]
    return model


# One loaded model per name for the process lifetime — loading dominates
# the cost of transcribing many short slices.
_fw_models: dict[tuple[str, bool], object] = {}
# Set once a CUDA attempt failed (e.g. GPU present but cublas64_*.dll absent):
# all further slices go straight to CPU instead of failing again.
_force_cpu = False


def _is_cuda_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(k in text for k in ("cublas", "cudnn", "cuda", "hip"))


def _faster_whisper_model(model: str, cpu: bool = False):
    from faster_whisper import WhisperModel

    name = ct2_model_name(model)
    key = (name, cpu)
    wm = _fw_models.get(key)
    if wm is None:
        if cpu:
            wm = WhisperModel(name, device="cpu", compute_type="int8")
        else:
            try:
                wm = WhisperModel(name, device="auto", compute_type="auto")
            except (ValueError, RuntimeError):
                # e.g. no float16 support, or CUDA runtime absent — int8 CPU
                # runs everywhere.
                return _faster_whisper_model(model, cpu=True)
        _fw_models[key] = wm
    return wm


def transcribe_slice(wav_path: str, model: str, language: str | None,
                     backend: str) -> str:
    """Transcribe one extracted 16 kHz mono WAV slice, return plain text."""
    if backend == "mlx":
        import mlx_whisper
        opts: dict = {"word_timestamps": False, "verbose": None,
                      "condition_on_previous_text": False}
        if language:
            opts["language"] = language
        result = mlx_whisper.transcribe(wav_path, path_or_hf_repo=model, **opts)
        return (result.get("text") or "").strip()

    if backend == "faster-whisper":
        global _force_cpu
        wm = _faster_whisper_model(model, cpu=_force_cpu)
        try:
            segments, _info = wm.transcribe(  # type: ignore[attr-defined]
                wav_path, language=language, condition_on_previous_text=False)
            # The generator is lazy: consume it here so CUDA failures surface.
            return " ".join(s.text.strip() for s in segments).strip()
        except RuntimeError as exc:
            # GPU visible mais runtime CUDA absent (ex. cublas64_12.dll
            # introuvable) : repli CPU définitif pour ce processus.
            if _force_cpu or not _is_cuda_error(exc):
                raise
            print("  CUDA indisponible (" + str(exc).splitlines()[0] + ") — repli CPU int8")
            _force_cpu = True
            wm = _faster_whisper_model(model, cpu=True)
            segments, _info = wm.transcribe(  # type: ignore[attr-defined]
                wav_path, language=language, condition_on_previous_text=False)
            return " ".join(s.text.strip() for s in segments).strip()

    raise ValueError(f"unknown backend: {backend!r}")
