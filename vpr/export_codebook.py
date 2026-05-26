"""Export per-extractor master VLAD codebooks for ``loop_fusion``.

Builds a k-means VLAD codebook over a learned extractor's local-descriptor
space, pooled across one or more **held-out VPR benchmark image roots** (NOT
our DL-VINS evaluation datasets, to avoid train/eval leakage). Writes the
result as a binary that ``loop_fusion``'s ``VladCodebook`` reads at startup:

    int32 clusters | int32 dim | clusters*dim float32 centres (row-major)

At C++ runtime, this codebook is loaded as a **warm-start seed** for the
online refit that happens at KF=``VLAD_WARMUP_KFS`` (see
``DL-VINS/src/loop_fusion/src/loop_fusion_component.cpp``). It does NOT
need to be the final codebook — clusters refine to the deployment's
descriptor distribution at refit time.

Output -> ``DL-VINS/src/loop_fusion/support_files/vlad_codebook_master_<ext>.bin``.

Recommended training mix (domain coverage > sheer count):
  indoor      AnyLoc/custom_datasets/baidu_mall, gardens
  urban       AnyLoc/datasets_vg/pitts30k (train/database), tokyo247
  aerial      AnyLoc/custom_datasets/VPAir
  subt        AnyLoc/custom_datasets/hawkins
  seasonal    AnyLoc/datasets_vg/nordland

    uv run python -m vpr.export_codebook \\
        --extractors superpoint xfeat aliked \\
        --image-roots /data/baidu_mall/images /data/gardens \\
                      /data/pitts30k/images/train/database
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

from vpr import config, extractors
from vpr.descriptors import _vlad

_OUT_DIR = config.REPO_ROOT / "DL-VINS" / "src" / "loop_fusion" / "support_files"
_DEFAULT_EXTRACTORS = ("aliked", "superpoint", "xfeat")
_IMAGE_GLOBS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


def _list_images(root: Path) -> list[Path]:
    paths: list[Path] = []
    for pat in _IMAGE_GLOBS:
        paths.extend(root.rglob(pat))
    # Deduplicate (case-insensitive globs can double-hit on case-insensitive FS).
    seen = set()
    out: list[Path] = []
    for p in sorted(paths):
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def build_one(extractor: str, image_roots: list[Path], clusters: int,
              sample_per_image: int, sample_cap: int, seed: int,
              target_wh: tuple[int, int] | None = None) -> None:
    ex = extractors.make_extractor(extractor, target_wh=target_wh)
    rng = np.random.default_rng(seed)
    pool: list[np.ndarray] = []
    total_imgs = 0
    for root in image_roots:
        imgs = _list_images(root)
        if not imgs:
            print(f"  [skip] {extractor}: no images under {root}")
            continue
        # Tag the cache by the root's leaf folder name so each root caches
        # independently (extractors.features keys its npz by ``seq``).
        seq_tag = root.name or root.parent.name
        desc = ex.features(seq_tag, imgs)["desc"]       # (N, K, D), float16
        for frame in desc:
            # Discard zero-padding rows (where the detector returned fewer than K
            # keypoints or the image was skipped). L2-normalised descriptors
            # have norm ≈ 1, so we gate on norm > 0.5.
            norms = np.linalg.norm(frame.astype(np.float32), axis=1)
            valid = np.where(norms > 0.5)[0]
            if not len(valid):
                continue
            take = min(sample_per_image, len(valid))
            idx = rng.choice(valid, take, replace=False)
            pool.append(frame[idx].astype(np.float32))
        total_imgs += len(imgs)
        print(f"  [pool] {extractor}@{ex.mode_key}/{seq_tag}: {len(imgs)} imgs, "
              f"{sample_per_image * len(imgs)} sampled (target)")

    if not pool:
        print(f"  [{extractor}] no features — skipped")
        return

    sample = np.concatenate(pool, axis=0)
    if len(sample) > sample_cap:
        sample = sample[rng.choice(len(sample), sample_cap, replace=False)]
    km = _vlad.build_codebook(sample, clusters, seed)
    centers = np.ascontiguousarray(km.cluster_centers_, dtype=np.float32)
    k, d = centers.shape

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = _OUT_DIR / f"vlad_codebook_master_{extractor}.bin"
    with open(out, "wb") as fp:
        fp.write(struct.pack("<ii", int(k), int(d)))
        fp.write(centers.tobytes())
    print(f"  [{extractor}] master codebook {k}x{d} -> {out}  "
          f"({len(sample)} descriptors over {total_imgs} images @ {ex.mode_key})")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Export master VLAD codebooks for loop_fusion.")
    p.add_argument("--extractors", nargs="*", default=list(_DEFAULT_EXTRACTORS),
                   choices=list(_DEFAULT_EXTRACTORS))
    p.add_argument("--image-roots", nargs="+", required=True, type=Path,
                   help="Directories of held-out VPR images (recursively globbed). "
                        "Use AnyLoc/{custom_datasets,datasets_vg}-style folders, "
                        "NOT the DL-VINS evaluation datasets.")
    p.add_argument("--target-wh", nargs=2, type=int, default=None,
                   metavar=("W", "H"),
                   help="Optional: resize every image to W x H before extraction. "
                        "Default (omitted) uses each image's native resolution "
                        "(cropped to /32-aligned shape). Pin only when matching "
                        "a fixed-shape deployment TRT engine exactly.")
    p.add_argument("--clusters", type=int, default=32)
    p.add_argument("--sample-per-image", type=int, default=64)
    p.add_argument("--sample-cap", type=int, default=150_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    roots = [r.expanduser().resolve() for r in args.image_roots]
    missing = [r for r in roots if not r.is_dir()]
    if missing:
        print(f"image roots not found / not directories: {missing}", file=sys.stderr)
        return 1

    target_wh: tuple[int, int] | None = (
        tuple(args.target_wh) if args.target_wh else None)  # type: ignore[assignment]
    mode = "native" if target_wh is None else f"{target_wh[0]}x{target_wh[1]}"

    config.ensure_dirs()
    print(f"Exporting master codebooks ({args.clusters} clusters, mode={mode}) "
          f"over {len(roots)} image root(s):")
    for r in roots:
        print(f"  - {r}")
    for extractor in args.extractors:
        build_one(extractor, roots, args.clusters,
                  args.sample_per_image, args.sample_cap, args.seed,
                  target_wh=target_wh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
