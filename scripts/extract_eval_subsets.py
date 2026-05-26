"""Batch-extract small cam0 subsamples from the DL-VINS evaluation rosbags
into ``tmp/vpr/eval_cam0/<key>/`` for the master-codebook UMAP overlay.

Scoped to three eval datasets for the **initial pre-validation pass**:
EuRoC + Kaist-VIO + NTU-VIRAL (sqlite3 + sqlite3 + mcap respectively). The
master codebook was trained on AnyLoc's held-out VPR datasets, so none of
these three leak into training.

After this script runs, it prints the exact ``vpr.visualize_codebook``
invocation with all three ``--deployment-roots`` populated — copy-paste to
generate the overlay PNGs at ``tmp/vpr/viz/``.

    cd /home/ubuntu/Desktop/dl-vins-factory/AnyLoc
    uv sync --group eval                  # rosbags
    uv run python -m scripts.extract_eval_subsets        # default 3 datasets
    uv run python -m scripts.extract_eval_subsets --list # show registry
    uv run python -m scripts.extract_eval_subsets EuRoC_V1_01_easy  # just one
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ANYLOC = _HERE.parent
_REPO_ROOT = _ANYLOC.parent
_OUT_ROOT = _REPO_ROOT / "tmp" / "vpr" / "eval_cam0"

# Eval datasets staged on disk. Bag dir = innermost folder with metadata.yaml.
# Topic names are best guesses; the extractor will fall back to the first
# sensor_msgs/msg/Image topic via --auto-topic if the guess is wrong.
EVAL_DATASETS: dict[str, dict] = {
    "EuRoC_V1_01_easy": {
        "bag": Path("/media/ubuntu/Crucial/dataset/2_EuRoC/V1_01_easy"),
        "topic": "/cam0/image_raw",
        "stride": 30,
    },
    "Kaist_VIO_circle": {
        "bag": Path("/media/ubuntu/Crucial/dataset/3_Kaist_VIO/circle"),
        "topic": "/stereo/left/image_raw",
        "stride": 30,
    },
    "NTU_VIRAL_eee_01": {
        "bag": Path("/media/ubuntu/Crucial/dataset/7-ntu-viral/NTU_VIRAL/"
                    "/eee_01/eee_01"),
        "topic": "/left/image_raw",
        "stride": 30,
    },
}


def _list_registry() -> None:
    print("eval-dataset registry:")
    for k, v in EVAL_DATASETS.items():
        present = "✓" if v["bag"].is_dir() else "✗ MISSING"
        print(f"  {k:<22} stride={v['stride']:<3} topic={v['topic']:<28} "
              f"{present}  {v['bag']}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("datasets", nargs="*", default=list(EVAL_DATASETS),
                   metavar="NAME",
                   help=f"datasets to extract (default: all 3; "
                        f"choices: {', '.join(EVAL_DATASETS)})")
    p.add_argument("--list", action="store_true",
                   help="list the registry + on-disk status and exit")
    p.add_argument("--stride", type=int, default=None,
                   help="override per-dataset stride (default: use registry)")
    p.add_argument("--max-frames", type=int, default=0,
                   help="hard cap on frames per dataset (0 = unlimited)")
    p.add_argument("--force", action="store_true",
                   help="re-extract even if the dataset's out dir is populated")
    args = p.parse_args(argv)

    if args.list:
        _list_registry()
        return 0

    unknown = [d for d in args.datasets if d not in EVAL_DATASETS]
    if unknown:
        print(f"unknown dataset(s): {unknown}\n"
              f"valid: {sorted(EVAL_DATASETS)}", file=sys.stderr)
        return 1

    from vpr.extract_rosbag import extract

    done: list[str] = []
    failed: list[str] = []
    for name in args.datasets:
        cfg = EVAL_DATASETS[name]
        bag = cfg["bag"]
        if not bag.is_dir():
            print(f"[{name}] MISSING: {bag}", file=sys.stderr)
            failed.append(name)
            continue
        out_dir = _OUT_ROOT / name
        stride = args.stride if args.stride is not None else cfg["stride"]
        print(f"[{name}] bag={bag}")
        n = extract(bag.resolve(), cfg["topic"], out_dir,
                    stride=stride, max_frames=args.max_frames,
                    force=args.force, auto_topic=True)
        if n > 0:
            done.append(name)
        else:
            failed.append(name)

    print(f"\nsummary: done={done} failed={failed}")
    if done:
        # Print the export+visualize commands pre-populated with master roots
        # (assumes the OneDrive defaults — adjust if you trained on a different mix).
        master_roots = [
            _ANYLOC / "datasets_vg" / "datasets" / d
            for d in ("baidu_datasets", "17places", "gardens", "VPAir",
                      "hawkins_long_corridor", "laurel_caverns")
        ]
        deploy_roots = [_OUT_ROOT / k for k in done]
        master_str = " \\\n                  ".join(str(r) for r in master_roots)
        deploy_str = " \\\n                       ".join(str(r) for r in deploy_roots)
        print("\nnext: UMAP overlay\n")
        print(f"    cd {_ANYLOC}")
        print(f"    uv run python -m vpr.visualize_codebook \\")
        print(f"      --image-roots {master_str} \\")
        print(f"      --deployment-roots {deploy_str}\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
