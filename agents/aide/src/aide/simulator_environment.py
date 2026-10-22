"""Measure the execution environment in the env_v1 training prompt's format.

The probe list and rendering match scripts/datagen/capture_env_manifest.py.
Run inside the agent's Python environment, where real scripts execute; the
server's packages and the training manifest are not substitutes.
"""

import hashlib
import importlib.metadata
import importlib.util
import os
import platform
import socket
import sys
from functools import lru_cache
from pathlib import Path


PROBE = {
    "numpy": None, "pandas": None, "torch": None, "sklearn": "scikit-learn",
    "PIL": "pillow", "torchvision": None, "tqdm": None, "cv2": "opencv-python",
    "scipy": None, "xgboost": None, "lightgbm": None, "matplotlib": None,
    "timm": None, "tensorflow": None, "albumentations": None, "joblib": None,
    "nltk": None, "skimage": "scikit-image", "sentence_transformers": None,
    "tifffile": None, "seaborn": None, "imageio": None, "keras": None,
    "imblearn": "imbalanced-learn", "torchmetrics": None, "catboost": None,
    "transformers": None, "statsmodels": None, "optuna": None,
    "albucore": None, "polars": None, "fastai": None,
}


def _wrap(prefix, items, width=76):
    lines, current = [], prefix
    for i, item in enumerate(items):
        piece = item + (", " if i < len(items) - 1 else "")
        if len(current) + len(piece) > width and current != prefix:
            lines.append(current.rstrip())
            current = ""
        current += piece
    if current.strip():
        lines.append(current.rstrip())
    return lines


def render_manifest(installed, missing, network, hub_cache, py_version=None):
    py = py_version or ".".join(str(x) for x in sys.version_info[:2])
    lines = [f"python {py} · {network}"] + _wrap("", installed)
    if missing:
        lines += _wrap("not installed: ", missing)
    if hub_cache:
        lines += _wrap("torch hub cache: ", hub_cache)
    return "\n".join(lines)


@lru_cache(maxsize=1)
def capture_environment() -> str:
    """Probe once per agent process without importing heavy ML libraries."""
    installed, missing = [], []
    for module_name, distribution in PROBE.items():
        try:
            available = importlib.util.find_spec(module_name) is not None
        except (ImportError, ValueError):
            available = False
        if not available:
            missing.append(module_name)
            continue
        version = None
        for name in (distribution or module_name, module_name):
            try:
                version = importlib.metadata.version(name)
                break
            except importlib.metadata.PackageNotFoundError:
                pass
        installed.append(f"{module_name} {version}" if version else module_name)

    try:
        with socket.create_connection(("pypi.org", 443), timeout=3.0):
            network = "network"
    except OSError:
        network = "no network"
    cache = Path(os.environ.get("TORCH_HOME") or Path.home() / ".cache/torch")
    checkpoints = cache / "hub/checkpoints"
    hub_cache = (
        sorted({p.stem.split("-")[0] for p in checkpoints.iterdir() if p.is_file()})
        if checkpoints.is_dir() else []
    )
    return render_manifest(installed, missing, network, hub_cache)


# ---------------------------------------------------------------------------
# Training-corpus <environment> text.
#
# The world model (corpus world_final_low_9_3_exc_token_h100, checkpoint-6685) was
# trained with <environment> stamped from scripts/datagen/env_manifest.json, not
# from a live probe: x86_64 run groups carry the manifest's x86 entry (162,223
# rows), aarch64 run groups carry its "default" entry (334,833 rows). Both end
# with an "installed but import fails at startup: ..." line that
# render_manifest() never emits, and the aarch64 entry pins older imageio /
# albucore versions, so capture_environment() matches no training row (audit
# 2026-09-18, logs/eval/sim_audit_20260918). Deployment therefore sends the
# training text for the running arch unless AIDE_SIMULATOR_ENV_TEXT=live.
# Byte-exact copies; TRAINING_ENVIRONMENT_SHA256 is checked by the tests.
# ---------------------------------------------------------------------------

_TRAINING_ENVIRONMENT_LINES = {
    "x86_64": (
        'python 3.11 · network',
        'numpy 1.26.2, pandas 2.1.4, torch 2.11.0+cu128, sklearn 1.2.2, PIL 10.4.0,',
        'torchvision 0.26.0+cu128, tqdm 4.66.2, cv2 4.10.0.84, scipy 1.11.4,',
        'xgboost 2.1.1, lightgbm 4.5.0, matplotlib 3.9.2, timm 0.9.7,',
        'albumentations 1.4.14, joblib 1.5.3, nltk 3.9.1, skimage 0.24.0,',
        'sentence_transformers 3.0.1, tifffile 2026.3.3, seaborn 0.13.2,',
        'imageio 2.37.4, keras 3.5.0, imblearn 0.12.3, torchmetrics 1.3.1,',
        'catboost 1.2.5, transformers 4.44.2, optuna 4.0.0, albucore 0.2.13,',
        'fastai 2.7.17',
        'not installed: tensorflow, statsmodels, polars',
        "installed but import fails at startup: albumentations (ImportError: cannot import name 'preserve_channel_dim' from 'albucore.utils'); keras with the default TensorFlow backend (ModuleNotFoundError: No module named 'tensorflow').",
    ),
    "aarch64": (
        'python 3.11 · network',
        'numpy 1.26.2, pandas 2.1.4, torch 2.11.0+cu128, sklearn 1.2.2, PIL 10.4.0,',
        'torchvision 0.26.0+cu128, tqdm 4.66.2, cv2 4.10.0.84, scipy 1.11.4,',
        'xgboost 2.1.1, lightgbm 4.5.0, matplotlib 3.9.2, timm 0.9.7,',
        'albumentations 1.4.14, joblib 1.5.3, nltk 3.9.1, skimage 0.24.0,',
        'sentence_transformers 3.0.1, tifffile 2026.3.3, seaborn 0.13.2,',
        'imageio 2.37.3, keras 3.5.0, imblearn 0.12.3, torchmetrics 1.3.1,',
        'catboost 1.2.5, transformers 4.44.2, optuna 4.0.0, albucore 0.1.6,',
        'fastai 2.7.17',
        'not installed: tensorflow, statsmodels, polars',
        "installed but import fails at startup: albumentations (ImportError: cannot import name 'preserve_channel_dim' from 'albucore.utils'); keras with the default TensorFlow backend (ModuleNotFoundError: No module named 'tensorflow').",
    ),
}
TRAINING_ENVIRONMENT = {
    arch: "\n".join(lines) for arch, lines in _TRAINING_ENVIRONMENT_LINES.items()
}
TRAINING_ENVIRONMENT_SHA256 = {
    "x86_64": "f0c4616ac5484547e487b8ae8d9c49dafbbe49d8c8db1d27aa690738c714fe19",
    "aarch64": "b653e69b7c0038e303dafa24c66b3ff4846956ddf2fe712cb5b0dd133a8ecc73",
}

_ENV_TEXT_SOURCES = ("training", "live")


def training_environment(arch: str | None = None) -> str:
    """The <environment> block the corpus used for this CPU arch."""
    arch = arch or platform.machine()
    try:
        return TRAINING_ENVIRONMENT[arch]
    except KeyError:
        raise ValueError(
            f"no training <environment> text for arch {arch!r}; "
            f"expected one of {sorted(TRAINING_ENVIRONMENT)}"
        ) from None


def resolve_environment() -> tuple[str, dict]:
    """<environment> text for the simulator prompt plus its provenance.

    AIDE_SIMULATOR_ENV_TEXT selects "training" (default: the corpus text for this
    arch) or "live" (capture_environment()). Unknown values raise so a typo fails
    the launch instead of silently probing.
    """
    source = os.environ.get("AIDE_SIMULATOR_ENV_TEXT", "").strip() or "training"
    if source not in _ENV_TEXT_SOURCES:
        raise ValueError(
            f"Unknown AIDE_SIMULATOR_ENV_TEXT {source!r}; expected one of {_ENV_TEXT_SOURCES}"
        )
    arch = platform.machine()
    text = training_environment(arch) if source == "training" else capture_environment()
    return text, {
        "source": source,
        "arch": arch,
        "lines": text.count("\n") + 1,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
