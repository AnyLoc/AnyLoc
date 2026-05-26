"""Central configuration for the offline VLAD master-codebook exporter.

This package trains the per-extractor *master* VLAD codebook consumed by
``DL-VINS/src/loop_fusion`` as a warm-start seed (the runtime refits the
codebook at KF=``VLAD_WARMUP_KFS`` over the deployment's own descriptors).
Training pulls images from one or more held-out VPR datasets (see
``vpr/export_codebook.py`` ``--image-roots``); evaluation of the C++
loop-closure path happens against the DL-VINS dataset roots, which are
*never* fed into master training.

Layout:  <repo>/AnyLoc/vpr/config.py
  VPR_PKG_DIR  = <repo>/AnyLoc/vpr
  ANYLOC_ROOT  = <repo>/AnyLoc
  REPO_ROOT    = <repo>              (weights/, tmp/, LightGlue-ONNX-Jetson/)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# ── repo layout ───────────────────────────────────────────────────────────────
VPR_PKG_DIR: Path = Path(__file__).resolve().parent
ANYLOC_ROOT: Path = VPR_PKG_DIR.parent
REPO_ROOT: Path = ANYLOC_ROOT.parent

# make ``import vpr`` resolve no matter the cwd
if str(ANYLOC_ROOT) not in sys.path:
    sys.path.insert(0, str(ANYLOC_ROOT))

# ── cache / weights ───────────────────────────────────────────────────────────
WEIGHTS_DIR: Path = Path(os.environ.get("VPR_WEIGHTS_DIR", str(REPO_ROOT / "weights")))
CACHE_DIR: Path = Path(os.environ.get("VPR_CACHE_DIR", str(REPO_ROOT / "tmp" / "vpr")))
FEATURES_DIR: Path = CACHE_DIR / "features"      # cached local features per (extractor, WxH)

# third-party model code reused by extractors
LIGHTGLUE_DIR: Path = REPO_ROOT / "LightGlue-ONNX-Jetson"

# ── extractor config ──────────────────────────────────────────────────────────
MATCHER_KPTS: int = int(os.environ.get("VPR_MATCHER_KPTS", "1024"))
# Master-codebook training uses NATIVE per-image resolution by default — the
# warm-start refit at runtime adapts the codebook to whatever resolution the
# deployed TRT engine actually feeds the network. To force a specific
# resolution (matching a fixed-shape deployment), pass --target-wh W H to
# vpr.export_codebook. The TRT engine filenames in
# DL-VINS/src/dl_vins/weights/<host>/<role>/*_h<H>_w<W>_*.engine document the
# runtime input dimensions if you do want to pin training to them.
ALIKED_VARIANT: str = os.environ.get("VPR_ALIKED_VARIANT", "aliked-n16")


def torch_device() -> str:
    """Resolve the torch device lazily."""
    forced = os.environ.get("VPR_DEVICE", "").strip()
    if forced:
        return forced
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def ensure_dirs() -> None:
    for d in (CACHE_DIR, FEATURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
