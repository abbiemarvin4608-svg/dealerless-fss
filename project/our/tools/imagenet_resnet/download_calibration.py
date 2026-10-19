#!/usr/bin/env python3
"""Read class headers and two training images per class from the official tar."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import http.client
import io
import json
from pathlib import Path
import signal
import tarfile
import threading
import time
from urllib.parse import urlsplit


SOURCE = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar"
ARCHIVE_BYTES = 147897477120
LOCAL = threading.local()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def read_range(url, start, count):
    parsed = urlsplit(url)
    if parsed.scheme != "https":
        raise ValueError("training archive must use HTTPS")
    for attempt in range(5):
        try:
            key = (parsed.hostname, parsed.port or 443)
            if getattr(LOCAL, "key", None) != key or getattr(LOCAL, "connection", None) is None:
                LOCAL.connection = http.client.HTTPSConnection(*key, timeout=60)
                LOCAL.key = key
            LOCAL.connection.request("GET", parsed.path, headers={
                "Range": "bytes=%d-%d" % (start, start + count - 1),
                "User-Agent": "ImageNet training calibration sampler"})
            response = LOCAL.connection.getresponse()
            if response.status != 206 or response.getheader("Content-Range") != (
                    "bytes %d-%d/%d" % (start, start + count - 1, ARCHIVE_BYTES)):
                raise ValueError("server did not return the requested official archive range")
            result = response.read(count + 1)
            if len(result) != count:
                raise ValueError("incomplete archive range")
            return result
        except (OSError, http.client.HTTPException, ValueError):
            if getattr(LOCAL, "connection", None):
                LOCAL.connection.close()
                LOCAL.connection = None
            if attempt == 4:
                raise
            time.sleep(min(2 ** attempt, 8))


def header(data):
    return tarfile.TarInfo.frombuf(data, "utf-8", "surrogateescape")


def index_archive(root, classes, url):
    path = root / "archive_index.json"
    state = json.loads(path.read_text()) if path.exists() else dict(
        source=url, archive_bytes=ARCHIVE_BYTES, next_offset=0, classes=[])
    if state["source"] != url or state["archive_bytes"] != ARCHIVE_BYTES:
        raise ValueError("existing index belongs to another archive")
    observed = {row["wnid"] for row in state["classes"]}
    offset = state["next_offset"]
    while observed != set(classes):
        if offset + 512 > ARCHIVE_BYTES:
            raise ValueError("training tar ended before all 1000 classes")
        info = header(read_range(url, offset, 512))
        wnid = Path(info.name).stem
        if not info.isfile() or Path(info.name).name != wnid + ".tar" or wnid not in classes or wnid in observed:
            raise ValueError("unexpected outer archive member: " + info.name)
        row = dict(wnid=wnid, offset=offset + 512, size=info.size)
        state["classes"].append(row)
        observed.add(wnid)
        offset += 512 + ((info.size + 511) // 512) * 512
        state["next_offset"] = offset
        write_json(path, state)
        if len(observed) % 20 == 0 or len(observed) == 1000:
            print(json.dumps(dict(indexed_classes=len(observed), total=1000)), flush=True)
    return state["classes"]


def first_images(prefix, count):
    found, position = [], 0
    while len(found) < count:
        if position + 512 > len(prefix):
            return None
        info = header(prefix[position:position + 512])
        start, end = position + 512, position + 512 + info.size
        if end > len(prefix):
            return None
        if info.isfile() and info.name.lower().endswith((".jpeg", ".jpg")):
            found.append((Path(info.name).name, prefix[start:end]))
        position = start + ((info.size + 511) // 512) * 512
    return found


def sample_class(row, root, url, count):
    from PIL import Image

    directory = root / "train" / row["wnid"]
    completion = root / "class_records" / (row["wnid"] + ".json")
    if completion.exists():
        record = json.loads(completion.read_text())
        if record["count"] != count:
            raise ValueError("existing sample count differs")
        if all((directory / name).is_file() for name in record["files"]):
            return record
    requested = min(512 << 10, row["size"])
    prefix = b""
    while True:
        prefix += read_range(url, row["offset"] + len(prefix), requested - len(prefix))
        images = first_images(prefix, count)
        if images is not None:
            break
        if requested >= row["size"] or requested >= 32 << 20:
            raise ValueError("could not obtain initial training images: " + row["wnid"])
        requested = min(requested * 2, row["size"])
    directory.mkdir(parents=True, exist_ok=True)
    for name, data in images:
        if not name.startswith(row["wnid"] + "_") or "ILSVRC2012_val_" in name:
            raise ValueError("unexpected training image identity: " + name)
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        target = directory / name
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(target)
    record = dict(wnid=row["wnid"], count=count, files=[name for name, _ in images],
                  selection="first regular JPEG members in the official class tar", source=url)
    write_json(completion, record)
    return record


def run(args):
    classes = json.loads(args.class_map.read_text())
    if len(classes) != 1000 or sorted(classes.values()) != list(range(1000)):
        raise ValueError("require the existing ImageNet-1K class mapping")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "class_records").mkdir(exist_ok=True)
    status_path = args.output / "DOWNLOAD_STATUS.json"
    write_json(status_path, dict(status="running", source=args.url, phase="index"))
    try:
        index = index_archive(args.output, classes, args.url)
        records = []
        write_json(status_path, dict(status="running", source=args.url, phase="images"))
        pool = ThreadPoolExecutor(max_workers=args.workers)
        futures = [pool.submit(sample_class, row, args.output, args.url, args.per_class) for row in index]
        try:
            for future in as_completed(futures):
                records.append(future.result())
                if len(records) % 20 == 0:
                    progress = dict(sampled_classes=len(records), images=len(records) * args.per_class)
                    write_json(status_path, dict(status="running", source=args.url, phase="images", **progress))
                    print(json.dumps(progress), flush=True)
        except BaseException:
            for future in futures:
                future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            pool.shutdown()
        write_json(args.output / "class_to_idx.json", classes)
        write_json(status_path, dict(status="complete", source=args.url, split="ILSVRC2012 train",
                   imagefolder=str((args.output / "train").resolve()), classes=len(records),
                   samples=sum(row["count"] for row in records), per_class=args.per_class,
                   selection="first regular JPEG members per official class tar", records=sorted(records, key=lambda r: r["wnid"])))
    except BaseException as error:
        write_json(status_path, dict(status="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                   source=args.url, reason="%s: %s" % (type(error).__name__, error)))
        raise


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise SystemExit("download interrupted by signal %d" % signum)

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--url", default=SOURCE)
    options = parser.parse_args()
    if options.per_class < 1 or options.workers < 1:
        parser.error("per-class and workers must be positive")
    run(options)
