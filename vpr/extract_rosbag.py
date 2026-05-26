"""Extract camera images from a ROS 2 rosbag into a folder of PNGs.

Handles both **sqlite3** (`.db3`) and **mcap** storage backends transparently
via the [`rosbags`](https://gitlab.com/ternaris/rosbags) library — pure Python,
no `source /opt/ros/humble/setup.bash` required. Backend is auto-detected
from the bag's ``metadata.yaml``.

Output is a flat folder of ``%06d.png`` + a ``timestamps.txt`` index that
``vpr.export_codebook`` / ``vpr.visualize_codebook`` can consume via
``--image-roots``.

    uv sync --group eval               # installs rosbags

    uv run python -m vpr.extract_rosbag \\
        --bag /media/ubuntu/Crucial/dataset/2_EuRoC/V1_01_easy \\
        --topic /cam0/image_raw \\
        --out /tmp/euroc_cam0/V1_01_easy \\
        --stride 30           # keep every 30th frame -> ~100 imgs from V1_01_easy

mcap bags (e.g. NTU-VIRAL ros2 mirror) work the same way; point ``--bag`` at
the innermost dir containing ``<seq>.mcap`` + ``metadata.yaml``:

    uv run python -m vpr.extract_rosbag \\
        --bag .../7-ntu-viral/NTU_VIRAL/ros2/eee_01/eee_01 \\
        --topic /left/image_raw \\
        --out /tmp/eval_cam0/NTU_VIRAL_eee_01 --stride 30
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _decode_image(msg) -> "np.ndarray":  # noqa: F821 (numpy imported lazily)
    """Decode a sensor_msgs/Image into HxW or HxWx3 numpy uint8."""
    import numpy as np
    enc = msg.encoding.lower()
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    if enc in ("mono8", "8uc1"):
        return raw.reshape(msg.height, msg.width)
    if enc in ("mono16", "16uc1"):
        arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
        return (arr >> 8).astype(np.uint8)
    if enc in ("bgr8", "8uc3"):
        return raw.reshape(msg.height, msg.width, 3)
    if enc == "rgb8":
        return raw.reshape(msg.height, msg.width, 3)[..., ::-1]   # -> bgr for cv2.imwrite
    if enc == "bayer_rggb8":
        import cv2
        bayer = raw.reshape(msg.height, msg.width)
        return cv2.cvtColor(bayer, cv2.COLOR_BayerRG2BGR)
    raise ValueError(f"unsupported encoding: {msg.encoding!r}")


def _list_image_topics(reader) -> list[str]:
    """Return all topics whose message type is sensor_msgs/msg/Image."""
    return sorted(
        c.topic for c in reader.connections
        if c.msgtype == "sensor_msgs/msg/Image"
    )


def _resolve_bag_path(bag: Path) -> Path:
    """Return either the bag *directory* (if its metadata.yaml is well-formed)
    or the inner ``.mcap`` / ``.db3`` *file* (when it isn't).

    Some rosbag2 exports leave ``storage_identifier`` blank in metadata.yaml
    (e.g. NTU-VIRAL's ros2 mirror). Pointing rosbags at the storage file
    directly bypasses the broken metadata since the storage backend is then
    inferred from the file extension.
    """
    if bag.is_file():
        return bag
    if not bag.is_dir():
        return bag  # let the caller raise
    meta = bag / "metadata.yaml"
    storage_ok = False
    if meta.is_file():
        try:
            import yaml as _yaml
            info = (_yaml.safe_load(meta.read_text())
                    or {}).get("rosbag2_bagfile_information", {}) or {}
            storage_ok = bool(info.get("storage_identifier"))
        except Exception:
            storage_ok = False
    if storage_ok:
        return bag    # metadata is well-formed; use directory dispatch
    # Fall back to the storage file (rosbags Reader dispatches on suffix).
    for ext in (".mcap", ".db3"):
        cand = sorted(bag.glob(f"*{ext}"))
        if cand:
            print(f"  [{bag.name}] metadata.yaml missing/empty "
                  f"storage_identifier; opening {cand[0].name} directly")
            return cand[0]
    return bag        # nothing found; let the error surface upstream


def extract(bag_dir: Path, topic: str, out_dir: Path,
            stride: int, max_frames: int, force: bool,
            auto_topic: bool = False) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_path = out_dir / "timestamps.txt"
    if ts_path.exists() and not force:
        with open(ts_path) as f:
            n = sum(1 for _ in f if _.strip() and not _.startswith("#"))
        print(f"  [{out_dir.name}] already extracted ({n} frames) — skip "
              f"(use --force to redo)")
        return n

    try:
        from rosbags.rosbag2 import Reader as Rosbag2Reader
        from rosbags.typesys import Stores, get_typestore
        import cv2
    except ImportError as e:
        print(f"missing dependency: {e}\n"
              f"  install with: uv sync --group eval", file=sys.stderr)
        sys.exit(1)

    # Resolve to .mcap/.db3 file when metadata.yaml is malformed (NTU-VIRAL).
    open_target = _resolve_bag_path(bag_dir)
    typestore = get_typestore(Stores.LATEST)
    written = 0
    n_msgs = 0
    t_start = time.time()

    reader = Rosbag2Reader(open_target)
    reader.open()
    try:
        image_topics = _list_image_topics(reader)
        if topic not in image_topics:
            if auto_topic and image_topics:
                picked = image_topics[0]
                print(f"  [{out_dir.name}] topic {topic!r} not found; "
                      f"--auto-topic picked {picked!r}\n"
                      f"  available image topics: {image_topics}")
                topic = picked
            else:
                print(f"  [{out_dir.name}] topic {topic!r} not in bag.\n"
                      f"  image topics in this bag: {image_topics}\n"
                      f"  re-run with one of those, or pass --auto-topic.",
                      file=sys.stderr)
                return 0

        conns = [c for c in reader.connections if c.topic == topic]
        with open(ts_path, "w") as ts_file:
            ts_file.write("# idx timestamp_seconds\n")
            for connection, _bag_ts, rawdata in reader.messages(connections=conns):
                n_msgs += 1
                if stride > 1 and ((n_msgs - 1) % stride) != 0:
                    continue
                try:
                    msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                    img = _decode_image(msg)
                except Exception as e:
                    print(f"  [warn] msg {n_msgs}: {e}")
                    continue
                png = out_dir / f"{written:06d}.png"
                cv2.imwrite(str(png), img)
                ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                ts_file.write(f"{written:06d} {ts:.9f}\n")
                written += 1
                if max_frames and written >= max_frames:
                    break
    finally:
        reader.close()

    dt = time.time() - t_start
    print(f"  [{out_dir.name}] wrote {written} frames (saw {n_msgs} on {topic}) "
          f"in {dt:.1f}s")
    return written


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bag", type=Path, required=True,
                   help="rosbag2 directory (sqlite3 .db3 or mcap; contains metadata.yaml)")
    p.add_argument("--topic", default="/cam0/image_raw")
    p.add_argument("--out", type=Path, required=True,
                   help="output directory for PNGs + timestamps.txt")
    p.add_argument("--stride", type=int, default=1,
                   help="keep every Nth message on the topic (default 1 = keep all)")
    p.add_argument("--max-frames", type=int, default=0,
                   help="hard cap on frames written (0 = unlimited)")
    p.add_argument("--auto-topic", action="store_true",
                   help="if --topic isn't in the bag, fall back to the first "
                        "sensor_msgs/msg/Image topic that is")
    p.add_argument("--list-topics", action="store_true",
                   help="just list the bag's image topics and exit (no extraction)")
    p.add_argument("--force", action="store_true",
                   help="re-extract even if timestamps.txt is already present")
    args = p.parse_args(argv)

    if not args.bag.is_dir():
        print(f"bag dir not found: {args.bag}", file=sys.stderr)
        return 1

    if args.list_topics:
        try:
            from rosbags.rosbag2 import Reader as Rosbag2Reader
        except ImportError as e:
            print(f"missing dependency: {e}\n"
                  f"  install with: uv sync --group eval", file=sys.stderr)
            return 1
        reader = Rosbag2Reader(_resolve_bag_path(args.bag.resolve()))
        reader.open()
        try:
            print(f"image topics in {args.bag}:")
            for t in _list_image_topics(reader):
                print(f"  {t}")
        finally:
            reader.close()
        return 0

    extract(args.bag.resolve(), args.topic, args.out.resolve(),
            args.stride, args.max_frames, args.force,
            auto_topic=args.auto_topic)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
