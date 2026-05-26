"""Local-feature extractors (SuperPoint / ALIKED / XFeat) for master-codebook
training.

Each is a thin PyTorch wrapper around the model vendored in the
LightGlue-ONNX-Jetson submodule. Maps a sequence of images to fixed-K local
descriptors (zero-padded if the detector returns fewer keypoints than K).

Resolution policy
-----------------
By default we process every image at its **native resolution**, with H and W
cropped down to the nearest multiple of ``LocalExtractor.align`` (32) since the
networks need stride-aligned input. This is the right choice for training a
domain-general master codebook: clusters cover the descriptor distribution the
network actually produces on real-world imagery rather than one specific
camera's resized output.

If you want to match a specific deployment exactly (e.g. an EuRoC-built TRT
engine fixed at 480×752), construct the extractor with ``target_wh=(W,H)`` —
images are then resized before extraction and clusters are pinned to that
distribution.

Caching
-------
Cached features land at ``tmp/vpr/features/<extractor>_<key>/<seq>.npz`` where
``<key>`` is ``native`` or ``WxH`` depending on mode. So the two modes can
coexist on disk without trampling each other.
"""
from __future__ import annotations

import contextlib
import sys
import time

import cv2
import numpy as np
import torch

from vpr import config
from vpr.descriptors import base

# make the submodule's ``lightglue_dynamo`` package importable
if str(config.LIGHTGLUE_DIR) not in sys.path:
    sys.path.insert(0, str(config.LIGHTGLUE_DIR))


@contextlib.contextmanager
def _local_hub_weights(state_dict: dict):
    """Redirect ``torch.hub.load_state_dict_from_url`` to a preloaded dict."""
    import torch.hub as hub
    orig = hub.load_state_dict_from_url
    hub.load_state_dict_from_url = lambda *a, **kw: state_dict
    try:
        yield
    finally:
        hub.load_state_dict_from_url = orig


def _align_crop(gray: np.ndarray, align: int, min_side: int = 64) -> np.ndarray:
    """Crop H/W down to the nearest multiple of `align`. Returns None-equivalent
    (raises) if either side falls below `min_side` after alignment.
    """
    h, w = gray.shape[:2]
    h2 = (h // align) * align
    w2 = (w // align) * align
    if h2 < min_side or w2 < min_side:
        raise ValueError(f"image too small after /{align} align: {w2}x{h2}")
    if h2 == h and w2 == w:
        return gray
    # Center-crop so we throw away the same number of pixels on each edge.
    y0 = (h - h2) // 2
    x0 = (w - w2) // 2
    return gray[y0:y0 + h2, x0:x0 + w2]


class LocalExtractor:
    """Uniform extractor: image paths -> fixed-K descriptors per image."""

    name: str = "base"
    desc_dim: int = 0
    align: int = 32   # network stride alignment (all three are /8 or /16; /32 is safe)

    def __init__(self, target_wh: tuple[int, int] | None = None):
        """target_wh=None  -> native resolution (recommended for master codebook)
        target_wh=(W,H)    -> resize every image to (W,H) before extraction
                              (use when matching a fixed-shape TRT runtime engine)
        """
        self._target_wh = target_wh

    @property
    def mode_key(self) -> str:
        if self._target_wh is None:
            return "native"
        return f"{self._target_wh[0]}x{self._target_wh[1]}"

    def _prepare(self, path) -> np.ndarray:
        gray = base.load_gray(path)
        if self._target_wh is not None:
            w, h = self._target_wh
            gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
        return _align_crop(gray, self.align)

    def _forward_single(self, gray: np.ndarray) -> np.ndarray:
        """1 image -> (K, desc_dim) float32 descriptors. Override per network."""
        raise NotImplementedError

    def features(self, seq: str, image_paths, force: bool = False,
                 quiet: bool = False) -> dict:
        """Extract (or load cached) descriptors for a whole sequence.

        Returns ``{'desc': (N, K, D) float16, 'ms': (N,) float32}``. We only
        keep descriptors — keypoint coords aren't used for codebook training
        and would have to be variable-length in native mode.
        """
        out_dir = config.FEATURES_DIR / f"{self.name}_{self.mode_key}"
        out_dir.mkdir(parents=True, exist_ok=True)
        npz = out_dir / f"{seq}.npz"
        n = len(image_paths)
        if npz.exists() and not force:
            cached = np.load(npz)
            if cached["desc"].shape[0] == n:
                return {k: cached[k] for k in cached.files}

        k = config.MATCHER_KPTS
        desc = np.zeros((n, k, self.desc_dim), np.float16)
        ms = np.zeros(n, np.float32)
        skipped = 0
        for i, p in enumerate(image_paths):
            try:
                gray = self._prepare(p)
            except (ValueError, FileNotFoundError) as e:
                if not quiet:
                    print(f"    [skip] {p}: {e}")
                skipped += 1
                continue
            t0 = time.time()
            ds = self._forward_single(gray)            # (K, D)
            ms[i] = (time.time() - t0) * 1000.0
            desc[i] = ds.astype(np.float16)
        result = {"desc": desc, "ms": ms}
        np.savez(npz, **result)
        if not quiet:
            print(f"  [feat] {self.name}@{self.mode_key}/{seq}  {n} frames"
                  f"{f' ({skipped} skipped)' if skipped else ''}, "
                  f"{float(np.mean(ms[ms > 0])) if (ms > 0).any() else 0.0:.1f} ms/img")
        return result


class _SuperPointExtractor(LocalExtractor):
    name = "superpoint"
    desc_dim = 256

    def __init__(self, target_wh: tuple[int, int] | None = None):
        super().__init__(target_wh)
        from lightglue_dynamo.models import SuperPoint
        self._device = config.torch_device()
        sd = torch.load(str(config.WEIGHTS_DIR / "superpoint_v1.pth"),
                        map_location="cpu")
        with _local_hub_weights(sd):
            net = SuperPoint(num_keypoints=config.MATCHER_KPTS)
        self._net = net.to(self._device).eval()

    def _forward_single(self, gray: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(gray.astype(np.float32) / 255.0)[None, None]
        with torch.no_grad():
            _kp, _sc, ds = self._net(x.to(self._device))
        return ds[0].cpu().numpy().astype(np.float32)   # (K, 256)


class _XFeatExtractor(LocalExtractor):
    name = "xfeat"
    desc_dim = 64

    def __init__(self, target_wh: tuple[int, int] | None = None):
        super().__init__(target_wh)
        from lightglue_dynamo.models import XFeat
        self._device = config.torch_device()
        self._net = XFeat(weights=str(config.WEIGHTS_DIR / "xfeat.pt"),
                          top_k=config.MATCHER_KPTS).to(self._device).eval()

    def _forward_single(self, gray: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(gray.astype(np.float32) / 255.0)[None, None]
        with torch.no_grad():
            _kp, _sc, ds, _nv = self._net.forward_export(
                x.to(self._device), top_k=config.MATCHER_KPTS)
        return ds[0].cpu().numpy().astype(np.float32)   # (K, 64)


class _ALIKEDExtractor(LocalExtractor):
    name = "aliked"
    desc_dim = 128

    def __init__(self, target_wh: tuple[int, int] | None = None):
        super().__init__(target_wh)
        from lightglue_dynamo.models import ALIKED
        self._device = config.torch_device()
        ckpt = config.WEIGHTS_DIR / f"{config.ALIKED_VARIANT}.pth"
        self._net = ALIKED(model_name=config.ALIKED_VARIANT, device=self._device,
                           top_k=config.MATCHER_KPTS,
                           pretrained_path=str(ckpt)).to(self._device).eval()

    def _forward_single(self, gray: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        x = torch.from_numpy(rgb.astype(np.float32) / 255.0)
        x = x.permute(2, 0, 1)[None].to(self._device)
        with torch.no_grad():
            pred = self._net(x)
        dsc = pred["descriptors"][0].detach().cpu().numpy()  # (Ni, 128)
        k = config.MATCHER_KPTS
        out = np.zeros((k, self.desc_dim), np.float32)
        ni = min(len(dsc), k)
        out[:ni] = dsc[:ni]
        return out


_REGISTRY = {"superpoint": _SuperPointExtractor,
             "xfeat": _XFeatExtractor,
             "aliked": _ALIKEDExtractor}
_CACHE: dict[tuple[str, tuple[int, int] | None], LocalExtractor] = {}


def make_extractor(name: str,
                   target_wh: tuple[int, int] | None = None) -> LocalExtractor:
    """Return a (process-cached) extractor. target_wh=None defaults to native."""
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(f"unknown extractor: {name} (have {sorted(_REGISTRY)})")
    cache_key = (key, target_wh)
    if cache_key not in _CACHE:
        _CACHE[cache_key] = _REGISTRY[key](target_wh=target_wh)
    return _CACHE[cache_key]
