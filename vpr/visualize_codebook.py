"""Visualize per-extractor local-descriptor distributions + master cluster
centres via 2-D UMAP projection.

For each extractor (SuperPoint / XFeat / ALIKED) we:
  1. Pool valid (non-zero-padded) descriptors from the training image roots,
     coloured by source dataset.
  2. Optionally overlay descriptors from a held-out "deployment" set in a
     distinct colour — useful to eyeball whether the master codebook's
     coverage spans where your DL-VINS evaluation data actually lands.
  3. Load the exported ``vlad_codebook_master_<ext>.bin`` and overlay its
     centres as black X markers.
  4. Fit a single UMAP on the union (so sample / deployment / centres share
     one 2-D coordinate system) and save one PNG per extractor under
     ``tmp/vpr/viz/``.

Usage
-----

    # Master training distribution only (what export_codebook saw)
    uv run python -m vpr.visualize_codebook \\
        --image-roots datasets_vg/datasets/baidu_datasets \\
                      datasets_vg/datasets/17places \\
                      datasets_vg/datasets/gardens \\
                      datasets_vg/datasets/VPAir \\
                      datasets_vg/datasets/hawkins_long_corridor \\
                      datasets_vg/datasets/laurel_caverns

    # Add a held-out deployment overlay (e.g. EuRoC cam0 PNGs you extracted)
    uv run python -m vpr.visualize_codebook \\
        --image-roots <master roots ...> \\
        --deployment-roots /tmp/euroc_cam0/V1_01_easy
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # never popup a window in headless / SSH usage
import matplotlib.pyplot as plt
import numpy as np

from vpr import config, extractors

_OUT_DIR = config.CACHE_DIR / "viz"
_BIN_DIR = config.REPO_ROOT / "DL-VINS" / "src" / "loop_fusion" / "support_files"
_IMAGE_GLOBS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
_DEFAULT_EXTRACTORS = ("superpoint", "xfeat", "aliked")


def _list_images(root: Path) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for pat in _IMAGE_GLOBS:
        for p in root.rglob(pat):
            if p not in seen:
                seen.add(p)
                out.append(p)
    return sorted(out)


def _load_master_centres(extractor: str) -> np.ndarray | None:
    path = _BIN_DIR / f"vlad_codebook_master_{extractor}.bin"
    if not path.is_file():
        print(f"  [{extractor}] no master codebook at {path.name} — skipping overlay")
        return None
    with open(path, "rb") as fp:
        header = fp.read(8)
        k, d = struct.unpack("<ii", header)
        centres = np.frombuffer(fp.read(k * d * 4), dtype=np.float32).reshape(k, d)
    return centres


def _pool_descs(extractor_obj, roots: list[Path],
                sample_per_image: int, cap_per_root: int,
                rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Returns {dataset_tag: (n, D) float32}. Skips zero-padding rows."""
    out: dict[str, np.ndarray] = {}
    for root in roots:
        imgs = _list_images(root)
        if not imgs:
            print(f"    [skip] no images under {root}")
            continue
        tag = root.name or root.parent.name
        feats = extractor_obj.features(tag, imgs)
        desc_all = feats["desc"]                              # (N, K, D) float16
        sampled: list[np.ndarray] = []
        for frame in desc_all:
            norms = np.linalg.norm(frame.astype(np.float32), axis=1)
            valid = np.where(norms > 0.5)[0]
            if not len(valid):
                continue
            take = min(sample_per_image, len(valid))
            idx = rng.choice(valid, take, replace=False)
            sampled.append(frame[idx].astype(np.float32))
        if not sampled:
            continue
        pool = np.concatenate(sampled, axis=0)
        if len(pool) > cap_per_root:
            pool = pool[rng.choice(len(pool), cap_per_root, replace=False)]
        out[tag] = pool
        print(f"    [{tag}] {len(pool)} descriptors")
    return out


def _build_pool(extractor: str,
                image_roots: list[Path],
                deployment_roots: list[Path] | None,
                target_wh: tuple[int, int] | None,
                sample_per_image: int, cap_per_root: int,
                seed: int):
    """Run extraction + concat once. Returns (ex, parts, samples, centres) where
    `parts` is a list of (tag, desc, group), `samples` is the concatenated
    (N, D) matrix, `centres` is the master codebook centres or None.
    """
    print(f"[{extractor}] pooling descriptors ...")
    ex = extractors.make_extractor(extractor, target_wh=target_wh)
    rng = np.random.default_rng(seed)
    train_pools = _pool_descs(ex, image_roots, sample_per_image, cap_per_root, rng)
    if not train_pools:
        print(f"  [{extractor}] empty training pool — skipping")
        return None
    deploy_pools = (_pool_descs(ex, deployment_roots, sample_per_image,
                                cap_per_root, rng)
                    if deployment_roots else {})

    centres = _load_master_centres(extractor)
    if centres is not None and centres.shape[1] != ex.desc_dim:
        print(f"  [{extractor}] master dim={centres.shape[1]} mismatches "
              f"extractor dim={ex.desc_dim} — ignoring centres")
        centres = None

    parts: list[tuple[str, np.ndarray, str]] = []
    for tag, pool in train_pools.items():
        parts.append((tag, pool, "train"))
    for tag, pool in deploy_pools.items():
        parts.append((tag, pool, "deploy"))
    samples = np.concatenate([p[1] for p in parts], axis=0)
    return ex, parts, samples, centres


def _umap_project(samples: np.ndarray, centres: np.ndarray | None,
                  n_neighbors: int, min_dist: float, seed: int):
    """Fit UMAP on (samples ++ centres) and return (proj_samples, proj_centres)."""
    try:
        import umap
    except ImportError:
        print("umap-learn not installed. Install with: uv sync --group viz",
              file=sys.stderr)
        sys.exit(1)
    joint = np.vstack([samples, centres]) if centres is not None else samples
    print(f"    UMAP fit: N={len(joint)} d={joint.shape[1]} "
          f"n_neighbors={n_neighbors} min_dist={min_dist}")
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                        min_dist=min_dist, random_state=seed,
                        metric="cosine")
    proj = reducer.fit_transform(joint)
    if centres is not None:
        return proj[:len(samples)], proj[len(samples):]
    return proj, None


def _plot_one_panel(ax, parts, proj_samples, proj_centres, centres,
                    *, big_legend: bool, train_size: int = 4,
                    deploy_size: int = 10, centre_size: int = 180,
                    centroid_size: int = 400, train_alpha: float = 0.35):
    """Render one scatter into a given matplotlib Axes.

    Per-deployment dataset we also add a large filled ★ at the **median** of
    the dataset's projected UMAP coordinates — gives a single anchor point
    summarising where that eval cloud lives vs the master `x` centres.
    Median is preferred over mean for robustness against outlier descriptors.
    """
    cmap_train = plt.cm.tab10
    cmap_deploy = plt.cm.Set2
    offset = 0
    train_i = deploy_i = 0
    # Defer centroid drawing until after all scatters so stars sit on top.
    deploy_centroids: list[tuple[str, np.ndarray, tuple]] = []
    for tag, pool, group in parts:
        n = len(pool)
        xy = proj_samples[offset:offset + n]
        offset += n
        if group == "train":
            color = cmap_train(train_i % 10)
            train_i += 1
            ax.scatter(xy[:, 0], xy[:, 1], s=train_size, alpha=train_alpha,
                       c=[color], label=f"train: {tag}", linewidths=0)
        else:
            color = cmap_deploy(deploy_i % 8)
            deploy_i += 1
            ax.scatter(xy[:, 0], xy[:, 1], s=deploy_size, alpha=0.7,
                       c=[color], marker="o", edgecolors="black",
                       linewidths=0.3, label=f"deploy: {tag}")
            deploy_centroids.append((tag, np.median(xy, axis=0), color))
    if proj_centres is not None:
        ax.scatter(proj_centres[:, 0], proj_centres[:, 1],
                   s=centre_size, c="black", marker="x", linewidths=2.5,
                   label=f"master ({centres.shape[0]} centres)")
    for tag, mid, color in deploy_centroids:
        ax.scatter([mid[0]], [mid[1]], s=centroid_size, c=[color],
                   marker="*", edgecolors="black", linewidths=1.8,
                   zorder=5, label=f"centroid: {tag}")
    ax.set_aspect("equal", adjustable="datalim")
    if big_legend:
        leg = ax.legend(markerscale=3, fontsize=8, loc="best", framealpha=0.85)
        # markerscale=3 keeps the small train/deploy dots readable but blows up
        # the large `x` / `★` markers — clamp those individually.
        for handle, text in zip(leg.legend_handles, leg.get_texts()):
            label = text.get_text()
            if label.startswith("master"):
                handle.set_sizes([60])
            elif label.startswith("centroid:"):
                handle.set_sizes([90])


def visualize_one(extractor: str, image_roots: list[Path],
                  deployment_roots: list[Path] | None,
                  target_wh: tuple[int, int] | None,
                  sample_per_image: int, cap_per_root: int,
                  n_neighbors: int, min_dist: float, seed: int) -> None:
    built = _build_pool(extractor, image_roots, deployment_roots, target_wh,
                        sample_per_image, cap_per_root, seed)
    if built is None:
        return
    ex, parts, samples, centres = built
    proj_samples, proj_centres = _umap_project(
        samples, centres, n_neighbors, min_dist, seed)

    fig, ax = plt.subplots(figsize=(11, 8))
    _plot_one_panel(ax, parts, proj_samples, proj_centres, centres,
                    big_legend=True)
    ax.set_title(f"{extractor} descriptor space "
                 f"@ {ex.mode_key} (UMAP-2D, cosine, "
                 f"n_neighbors={n_neighbors}, min_dist={min_dist})")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUT_DIR / f"viz_{extractor}_{ex.mode_key}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  [{extractor}] -> {out_path}")


def sweep_one(extractor: str, image_roots: list[Path],
              deployment_roots: list[Path] | None,
              target_wh: tuple[int, int] | None,
              sample_per_image: int, cap_per_root: int,
              nn_grid: list[int], md_grid: list[float], seed: int) -> None:
    """3×3 (or NxM) sweep over (n_neighbors, min_dist). Pool is built once and
    reused — only the UMAP fit changes per cell."""
    built = _build_pool(extractor, image_roots, deployment_roots, target_wh,
                        sample_per_image, cap_per_root, seed)
    if built is None:
        return
    ex, parts, samples, centres = built

    rows = len(nn_grid)
    cols = len(md_grid)
    fig, axes = plt.subplots(rows, cols,
                             figsize=(4.5 * cols, 4 * rows + 0.8),
                             squeeze=False)
    for i, nn in enumerate(nn_grid):
        for j, md in enumerate(md_grid):
            ax = axes[i, j]
            proj_samples, proj_centres = _umap_project(
                samples, centres, nn, md, seed)
            _plot_one_panel(ax, parts, proj_samples, proj_centres, centres,
                            big_legend=False)
            ax.set_title(f"n_neighbors={nn}, min_dist={md}", fontsize=10)
            if i == rows - 1:
                ax.set_xlabel("UMAP-1")
            if j == 0:
                ax.set_ylabel("UMAP-2")

    # Single shared legend at the top.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="upper center", ncol=min(8, len(labels)),
                     fontsize=8, markerscale=3, bbox_to_anchor=(0.5, 1.00),
                     framealpha=0.9)
    for handle, text in zip(leg.legend_handles, leg.get_texts()):
        label = text.get_text()
        if label.startswith("master"):
            handle.set_sizes([60])
        elif label.startswith("centroid:"):
            handle.set_sizes([90])

    fig.suptitle(f"{extractor} UMAP sweep @ {ex.mode_key} "
                 f"(cosine; {len(samples)} train + "
                 f"{centres.shape[0] if centres is not None else 0} centres)",
                 fontsize=12, y=0.995)
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUT_DIR / f"viz_sweep_{extractor}_{ex.mode_key}.png"
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  [{extractor}] -> {out_path}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--extractors", nargs="*", default=list(_DEFAULT_EXTRACTORS),
                   choices=list(_DEFAULT_EXTRACTORS))
    p.add_argument("--image-roots", nargs="+", required=True, type=Path,
                   help="Same image roots as vpr.export_codebook (master training set).")
    p.add_argument("--deployment-roots", nargs="*", type=Path, default=None,
                   help="Optional held-out roots to overlay (e.g. extracted EuRoC cam0).")
    p.add_argument("--target-wh", nargs=2, type=int, default=None,
                   metavar=("W", "H"),
                   help="Match vpr.export_codebook: omit for native, set to "
                        "pin extractor input resolution.")
    p.add_argument("--sample-per-image", type=int, default=64)
    p.add_argument("--cap-per-root", type=int, default=8000,
                   help="Cap descriptors per dataset (UMAP slows above ~50k total)")
    p.add_argument("--n-neighbors", type=int, default=30,
                   help="UMAP n_neighbors for the single-plot mode (ignored under --sweep).")
    p.add_argument("--min-dist", type=float, default=0.1,
                   help="UMAP min_dist for the single-plot mode (ignored under --sweep).")
    p.add_argument("--sweep", action="store_true",
                   help="Produce a grid sweep over (n_neighbors x min_dist) "
                        "instead of one plot. Pool is built once per extractor "
                        "and reused across UMAP fits.")
    p.add_argument("--sweep-nn", nargs="+", type=int, default=[15, 30, 50],
                   help="n_neighbors values for --sweep (default: 15 30 50)")
    p.add_argument("--sweep-md", nargs="+", type=float, default=[0.05, 0.1, 0.3],
                   help="min_dist values for --sweep (default: 0.05 0.1 0.3)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    train_roots = [r.expanduser().resolve() for r in args.image_roots]
    deploy_roots = ([r.expanduser().resolve() for r in args.deployment_roots]
                    if args.deployment_roots else None)
    missing = [r for r in train_roots if not r.is_dir()]
    if deploy_roots:
        missing += [r for r in deploy_roots if not r.is_dir()]
    if missing:
        print(f"missing image roots: {missing}", file=sys.stderr)
        return 1

    target_wh: tuple[int, int] | None = (
        tuple(args.target_wh) if args.target_wh else None)  # type: ignore[assignment]

    config.ensure_dirs()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ex_name in args.extractors:
        if args.sweep:
            sweep_one(ex_name, train_roots, deploy_roots, target_wh,
                      args.sample_per_image, args.cap_per_root,
                      args.sweep_nn, args.sweep_md, args.seed)
        else:
            visualize_one(ex_name, train_roots, deploy_roots, target_wh,
                          args.sample_per_image, args.cap_per_root,
                          args.n_neighbors, args.min_dist, args.seed)
    print(f"\nall plots -> {_OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
