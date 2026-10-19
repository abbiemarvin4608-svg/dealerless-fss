#!/usr/bin/env python3
"""Download ImageNet-1k validation images and official labels as ImageFolder."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import threading
import time
import zipfile


SOURCES = {
    "val": {
        "url": "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar",
        "size_bytes": 6744924160,
        "md5": "29b22e2961454d5413ddabcf34fc5622",
    },
    "devkit": {
        "url": "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz",
        "size_bytes": 2568145,
        "md5": "fa75699e90414af021442c21a62c3abf",
    },
    "kaggle_val": {
        "url": "https://www.kaggle.com/api/v1/datasets/download/titericz/imagenet1k-val?datasetVersionNumber=1",
        "filename": "kaggle_titericz_imagenet1k_val_v1.zip",
        "size_bytes": 6671276585,
        "md5": "dc2554263aa9fafe735a4d06b2ece298",
        "md5_reference": "Public GCS response x-goog-hash: md5=3CVUJjqp+v5zWk0GsuzimA==",
        "dataset_page": "https://www.kaggle.com/datasets/titericz/imagenet1k-val",
    },
}
USER_AGENT = "ImageNet validation research downloader (Python urllib)"


def digest(path, algorithm):
    result = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


class DownloadPending(Exception):
    pass


def download(source, directory, workers, chunk_bytes, deadline, resolve):
    path = directory / source.get("filename", source["url"].rsplit("/", 1)[1])
    if path.exists():
        if path.stat().st_size != source["size_bytes"] or digest(path, "md5") != source["md5"]:
            raise ValueError("Existing archive is incomplete or corrupt: " + str(path))
        print("Verified existing archive: " + str(path), flush=True)
        return path
    parts = directory / (path.name + ".parts")
    parts.mkdir(parents=True, exist_ok=True)
    ranges = [(offset, min(offset + chunk_bytes, source["size_bytes"]))
              for offset in range(0, source["size_bytes"], chunk_bytes)]
    initial_bytes = sum(item.stat().st_size for item in parts.iterdir())
    stopping = threading.Event()
    started = time.monotonic()

    def part(bounds):
        start, end = bounds
        target = parts / ("%012d-%012d" % (start, end))
        if target.is_file() and target.stat().st_size == end - start:
            return target
        for attempt in range(6):
            if stopping.is_set() or time.monotonic() >= deadline:
                raise DownloadPending("Download time budget reached; byte range checkpoints are retained")
            try:
                transfer = subprocess.run(["curl", "--location", "--fail", "--silent", "--show-error",
                    "--connect-timeout", "20", "--max-time", "180", "--user-agent", USER_AGENT,
                    "--range", "%d-%d" % (start, end - 1), "--output", str(target),
                    "--write-out", "%{http_code}", *[item for entry in resolve for item in ("--resolve", entry)], source["url"]],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=195)
                if transfer.returncode != 0:
                    raise RuntimeError("curl exit %d: %s" % (transfer.returncode, transfer.stderr[:300]))
                if transfer.stdout != "206" or target.stat().st_size != end - start:
                    raise ValueError("Incomplete byte range")
                return target
            except Exception as error:
                print(json.dumps({"archive": path.name, "range_start": start,
                    "attempt": attempt + 1, "error": repr(error)}), flush=True)
                if attempt == 5:
                    raise
                time.sleep(min(2 ** attempt, 15))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(part, bounds) for bounds in ranges}
        last_report = started
        try:
            while pending:
                done, pending = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()
                now = time.monotonic()
                if now - last_report >= 10 or not pending:
                    available = sum(item.stat().st_size for item in parts.iterdir())
                    elapsed = now - started
                    rate = max(0, available - initial_bytes) / elapsed if elapsed else 0
                    print(json.dumps({"archive": path.name, "checkpoint_bytes": available,
                        "total_bytes": source["size_bytes"], "elapsed_seconds": round(elapsed, 1),
                        "average_mib_per_second": round(rate / 2 ** 20, 3),
                        "estimated_remaining_seconds": round((source["size_bytes"] - available) / rate) if rate else None}), flush=True)
                    last_report = now
                if time.monotonic() >= deadline:
                    raise DownloadPending("Download time budget reached; checkpoints are retained")
        except BaseException:
            stopping.set()
            for future in pending:
                future.cancel()
            raise
    temporary = path.with_suffix(path.suffix + ".assembling")
    with temporary.open("wb") as output:
        for start, end in ranges:
            with (parts / ("%012d-%012d" % (start, end))).open("rb") as stream:
                shutil.copyfileobj(stream, output, length=1024 * 1024)
    if temporary.stat().st_size != source["size_bytes"] or digest(temporary, "md5") != source["md5"]:
        raise ValueError("Official archive MD5 verification failed: " + str(temporary))
    temporary.replace(path)
    shutil.rmtree(parts)
    print("Downloaded and verified: " + str(path), flush=True)
    return path


def prepare(root, val_path, devkit_path, source_name, resolve):
    from scipy.io import loadmat
    from PIL import Image

    prefix = "ILSVRC2012_devkit_t12/data/"
    with tarfile.open(devkit_path, "r:gz") as archive:
        metadata = archive.extractfile(prefix + "meta.mat").read()
        ground_truth = archive.extractfile(prefix + "ILSVRC2012_validation_ground_truth.txt").read()
    synsets = loadmat(io.BytesIO(metadata), squeeze_me=True)["synsets"]
    leaves = [row for row in synsets if int(row[4]) == 0]
    official_to_wnid = {int(row[0]): str(row[1]) for row in leaves}
    names = {str(row[1]): str(row[2]) for row in leaves}
    wnids = sorted(official_to_wnid.values())
    class_to_idx = {wnid: index for index, wnid in enumerate(wnids)}
    labels = [int(line) for line in ground_truth.decode("ascii").splitlines()]
    if len(official_to_wnid) != 1000 or len(set(wnids)) != 1000 or len(labels) != 50000:
        raise ValueError("Official devkit does not contain the expected ImageNet-1k validation labels")
    expected = {"ILSVRC2012_val_%08d.JPEG" % (index + 1): official_to_wnid[label]
                for index, label in enumerate(labels)}
    output = root / "val"
    for wnid in wnids:
        (output / wnid).mkdir(parents=True, exist_ok=True)
    count = 0
    archive = zipfile.ZipFile(val_path) if source_name == "kaggle_val" else tarfile.open(val_path, "r:")
    with archive:
        seen = set()
        members = archive.infolist() if source_name == "kaggle_val" else archive
        for member in members:
            if source_name == "kaggle_val":
                if member.is_dir():
                    continue
                if member.filename == "imagenet-val/dataset-metadata.json":
                    archive.read(member)
                    continue
                filename = Path(member.filename).name
                if member.filename != "imagenet-val/%s/%s" % (expected.get(filename), filename):
                    raise ValueError("Mirror folder disagrees with official devkit: " + member.filename)
                size, stream = member.file_size, archive.open(member)
            else:
                filename = member.name
                if not member.isfile():
                    raise ValueError("Unexpected archive member: " + filename)
                size, stream = member.size, archive.extractfile(member)
            if filename not in expected or filename in seen:
                raise ValueError("Unexpected or duplicate archive member: " + filename)
            seen.add(filename)
            target = output / expected[filename] / filename
            with stream, target.open("wb") as destination:
                shutil.copyfileobj(stream, destination)
            if target.stat().st_size != size:
                raise ValueError("Extracted image has an incorrect byte count: " + filename)
            with Image.open(target) as image:
                image.verify()
            count += 1
            if count % 5000 == 0:
                print(json.dumps({"prepared_and_verified_images": count, "total": 50000}), flush=True)
        if seen != set(expected):
            raise ValueError("Validation archive has missing images")
    observed = {path.name: len(list(path.glob("*.JPEG"))) for path in output.iterdir() if path.is_dir()}
    if set(observed) != set(wnids) or any(count != 50 for count in observed.values()):
        raise ValueError("ImageFolder does not contain exactly 50 images in each of 1000 classes")
    (root / "ILSVRC2012_validation_ground_truth.txt").write_bytes(ground_truth)
    (root / "meta.mat").write_bytes(metadata)
    (root / "class_to_idx.json").write_text(json.dumps(class_to_idx, indent=2, sort_keys=True) + "\n")
    with (root / "labels.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["filename", "official_ilsvrc_id", "wnid", "torchvision_class_index", "class_name"])
        for index, label in enumerate(labels):
            wnid = official_to_wnid[label]
            writer.writerow(["ILSVRC2012_val_%08d.JPEG" % (index + 1), label,
                wnid, class_to_idx[wnid], names[wnid]])
    manifest = {"status": "complete", "image_count": count, "class_count": len(wnids),
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "images_per_class": 50, "imagefolder": str(output.resolve()),
        "image_archive": val_path.name, "label_archive": devkit_path.name,
        "sources": {"images": SOURCES[source_name], "labels": SOURCES["devkit"]},
        "checksum_reference": "Official devkit MD5: torchvision.datasets.imagenet.ARCHIVE_META; mirror ZIP MD5: public GCS x-goog-hash" if source_name == "kaggle_val" else "torchvision.datasets.imagenet.ARCHIVE_META (torchvision 0.19.1)",
        "archive_md5_verified": True, "zip_crc_verified": source_name == "kaggle_val", "all_images_pillow_verified": True,
        "class_mapping": "Official devkit ILSVRC2012 ID -> WNID -> lexicographically sorted WNID 0-based index; same as torchvision ImageNet/ImageFolder",
        "labels_file": "labels.csv", "official_labels_file": "ILSVRC2012_validation_ground_truth.txt",
        "class_to_idx_file": "class_to_idx.json",
        "access": "Public download endpoint and official devkit respond without credentials or cookies; no authentication bypass used",
        "transport_dns_pins": resolve,
        "usage_terms": "https://www.image-net.org/download-images.php (non-commercial research and/or educational purposes under ImageNet terms)"}
    (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-mib", type=int, default=32)
    parser.add_argument("--source", choices=("official", "kaggle"), default="official")
    parser.add_argument("--max-minutes", type=float, default=60)
    parser.add_argument("--resolve", action="append", default=[], help="Optional curl HOST:PORT:IP DNS pin; TLS verification remains enabled")
    args = parser.parse_args()
    if args.workers < 1 or args.chunk_mib < 1 or args.max_minutes <= 0:
        parser.error("workers, chunk-mib, and max-minutes must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.max_minutes * 60
    source_name = "kaggle_val" if args.source == "kaggle" else "val"
    try:
        devkit = download(SOURCES["devkit"], args.output, args.workers, args.chunk_mib * 1024 * 1024, deadline, args.resolve)
        images = download(SOURCES[source_name], args.output, args.workers, args.chunk_mib * 1024 * 1024, deadline, args.resolve)
        prepare(args.output, images, devkit, source_name, args.resolve)
    except (DownloadPending, RuntimeError, OSError, subprocess.SubprocessError) as error:
        status = {"status": "pending", "reason": str(error), "source": source_name}
        (args.output / "DOWNLOAD_STATUS.json").write_text(json.dumps(status, indent=2) + "\n")
        print(json.dumps(status), flush=True)
        return 2
    except (ValueError, tarfile.TarError, zipfile.BadZipFile) as error:
        status = {"status": "failed", "reason": str(error), "source": source_name}
        (args.output / "DOWNLOAD_STATUS.json").write_text(json.dumps(status, indent=2) + "\n")
        print(json.dumps(status), flush=True)
        return 1
    (args.output / "DOWNLOAD_STATUS.json").write_text(json.dumps({"status": "complete"}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
