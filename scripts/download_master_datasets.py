"""Extract the AnyLoc public-release tarballs into ``datasets_vg/datasets/``
and print the matching ``vpr.export_codebook`` command for master-codebook
training.

Usage
-----

    # Recommended master mix
    uv run python -m scripts.download_master_datasets

    # Pick explicitly (any subset of `--list`):
    uv run python -m scripts.download_master_datasets \\
        17places baidu_mall gardens hawkins laurel_caverns vpair

    # List everything you've staged
    uv run python -m scripts.download_master_datasets --list

After extraction the script prints a copy-paste
``uv run python -m vpr.export_codebook --image-roots ...`` invocation pointing
at every extracted dataset. The recursive image glob in ``export_codebook.py``
picks up arbitrary subfolder layouts (CVPR2022 ``train/database`` /
``test/queries``, plain image folders, etc.).
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ANYLOC = _HERE.parent
_STAGING = _ANYLOC / "datasets_vg" / "AnyLoc2023-Public-Data" / "Public" / "Datasets-All"
_DATA_DIR = _ANYLOC / "datasets_vg" / "datasets"

# short_name -> (tarball filename in _STAGING, extracted-dir sentinel under _DATA_DIR)
_REGISTRY: dict[str, tuple[str, str]] = {
    # indoor
    "baidu_mall":     ("baidu_datasets.tar.gz",          "baidu_datasets"),
    "17places":       ("17places.tar.gz",                "17places"),
    "hawkins":        ("hawkins_long_corridor.tar.gz",   "hawkins_long_corridor"),
    # campus / outdoor / urban
    "gardens":        ("gardens.tar.gz",                 "gardens"),
    "pitts30k":       ("pitts30k.tar.gz",                "pitts30k"),
    "st_lucia":       ("st_lucia.tar.gz",                "st_lucia"),
    "oxford":         ("Oxford_Robotcar.tar.gz",         "Oxford_Robotcar"),
    # subt
    "laurel_caverns": ("laurel_caverns.tar.gz",          "laurel_caverns"),
    # aerial
    "vpair":          ("VPAir.tar.gz",                   "VPAir"),
    "nardo_rot0":     ("test_40_midref_rot0.tar.gz",     "test_40_midref_rot0"),
    "nardo_rot90":    ("test_40_midref_rot90.tar.gz",    "test_40_midref_rot90"),
    # underwater (probably not useful for VINS but you have it)
    "eiffel":         ("eiffel.tar.gz",                  "eiffel"),
}

# Smallest set giving full coverage evaluation datasets.
# (Add ``svox`` separately if you grabbed it via the previous downloader.)
_DEFAULTS: tuple[str, ...] = (
    "baidu_mall",      # indoor mall
    "17places",        # indoor varied
    "gardens",         # campus day/night
    "vpair",           # aerial UAV
    "hawkins",         # indoor corridor
    "laurel_caverns",  # sub-terranean
)

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def _is_extracted(sentinel: str) -> bool:
    """True iff _DATA_DIR/<sentinel> already contains real images."""
    p = _DATA_DIR / sentinel
    if not p.is_dir():
        return False
    for ext in _IMAGE_EXTS:
        if next(p.rglob(f"*{ext}"), None) is not None:
            return True
    return False


def _list_staged() -> None:
    print(f"staging dir: {_STAGING}")
    if not _STAGING.is_dir():
        print("  (does not exist)")
        return
    for name, (tarball, sentinel) in _REGISTRY.items():
        arc = _STAGING / tarball
        if arc.is_file():
            size_mb = arc.stat().st_size / (1 << 20)
            extracted = "EXTRACTED" if _is_extracted(sentinel) else "staged"
            print(f"  {name:<16} {tarball:<36} {size_mb:7.0f} MiB  [{extracted}]")
        else:
            print(f"  {name:<16} {tarball:<36} {'-':>11}  [MISSING]")


def _extract(name: str, force: bool) -> bool:
    tarball, sentinel = _REGISTRY[name]
    arc = _STAGING / tarball
    out = _DATA_DIR / sentinel
    if not arc.is_file():
        print(f"[{name}] MISSING staged tarball: {arc}", file=sys.stderr)
        return False
    if not force and _is_extracted(sentinel):
        print(f"[{name}] already extracted at {out} — skip")
        return True
    out.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] extracting {tarball} -> {out}")
    try:
        shutil.unpack_archive(str(arc), str(out))
    except Exception as e:
        print(f"[{name}] FAILED: {e}", file=sys.stderr)
        return False
    return True


def _print_export_cmd(extracted: list[str]) -> None:
    if not extracted:
        return
    roots = []
    for name in extracted:
        sentinel = _REGISTRY[name][1]
        base = _DATA_DIR / sentinel
        if not base.is_dir():
            continue
        # Prefer ``images/`` if upstream uses the CVPR2022 layout, else the root.
        root = base / "images" if (base / "images").is_dir() else base
        roots.append(str(root))
    # If SVOX exists from the earlier auto-downloader, include it too.
    svox = _DATA_DIR / "svox" / "images"
    if svox.is_dir() and str(svox) not in roots:
        roots.insert(0, str(svox))
    if not roots:
        return
    joined = " \\\n        ".join(roots)
    print("\nmaster-codebook training, copy-paste-ready:\n")
    print(f"    cd {_ANYLOC}")
    print(f"    uv run python -m vpr.export_codebook \\")
    print(f"      --extractors superpoint xfeat aliked \\")
    print(f"      --image-roots {joined}\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("datasets", nargs="*", default=list(_DEFAULTS),
                   metavar="NAME",
                   help=f"datasets to extract (default: {' '.join(_DEFAULTS)})")
    p.add_argument("--list", action="store_true",
                   help="list staged tarballs and exit")
    p.add_argument("--all", action="store_true",
                   help="extract every dataset that's staged (overrides positional args)")
    p.add_argument("--force", action="store_true",
                   help="re-extract even if the target dir already has images")
    args = p.parse_args(argv)

    if args.list:
        _list_staged()
        return 0
    if not _STAGING.is_dir():
        print(f"staging dir not found: {_STAGING}", file=sys.stderr)
        return 1
    names = list(_REGISTRY) if args.all else list(args.datasets)
    unknown = [n for n in names if n not in _REGISTRY]
    if unknown:
        print(f"unknown dataset(s): {unknown}\nvalid: {sorted(_REGISTRY)}",
              file=sys.stderr)
        return 1

    done: list[str] = []
    failed: list[str] = []
    for name in names:
        if _extract(name, args.force):
            done.append(name)
        else:
            failed.append(name)

    print(f"\nsummary: done={done} failed={failed}")
    _print_export_cmd(done)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
