#!/usr/bin/env python3
"""Download one HTTP object with parallel, resumable byte ranges."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import time
import urllib.request
from pathlib import Path


def remote_size(url: str) -> int:
    request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        content_range = response.headers.get("Content-Range", "")
        if "/" not in content_range:
            raise RuntimeError(f"server does not advertise a ranged size: {content_range}")
        return int(content_range.rsplit("/", 1)[1])


def download_part(url: str, part: Path, start: int, end: int,
                  retries: int = 20) -> int:
    expected = end - start + 1
    existing = part.stat().st_size if part.exists() else 0
    if existing > expected:
        raise RuntimeError(f"oversized partial file: {part}")
    while existing < expected:
        offset = start + existing
        request = urllib.request.Request(
            url, headers={"Range": f"bytes={offset}-{end}"}
        )
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(request, timeout=120) as response, \
                        part.open("ab") as output:
                    if response.status != 206:
                        raise RuntimeError(f"expected HTTP 206, got {response.status}")
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                break
            except Exception:
                if attempt + 1 == retries:
                    raise
                time.sleep(min(30, 2 ** min(attempt, 4)))
        existing = part.stat().st_size
    return existing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("output")
    parser.add_argument("--connections", type=int, default=8)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    size = remote_size(args.url)
    if output.exists() and output.stat().st_size == size:
        print(f"complete: {output} ({size} bytes)", flush=True)
        return

    parts_dir = output.with_name(output.name + ".parts")
    parts_dir.mkdir(exist_ok=True)
    ranges = []
    for index in range(args.connections):
        start = size * index // args.connections
        end = size * (index + 1) // args.connections - 1
        ranges.append((index, start, end, parts_dir / f"part-{index:03d}"))

    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.connections) as pool:
        futures = {
            pool.submit(download_part, args.url, part, start, end):
                (index, start, end, part)
            for index, start, end, part in ranges
        }
        for future in concurrent.futures.as_completed(futures):
            index, start, end, _ = futures[future]
            downloaded = future.result()
            print(
                f"part {index + 1}/{len(ranges)} complete: {downloaded} bytes "
                f"[{start}, {end}]",
                flush=True,
            )

    temporary = output.with_name(output.name + ".assembling")
    with temporary.open("wb") as destination:
        for _, start, end, part in ranges:
            if part.stat().st_size != end - start + 1:
                raise RuntimeError(f"incomplete part: {part}")
            with part.open("rb") as source:
                while True:
                    chunk = source.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    destination.write(chunk)
    if temporary.stat().st_size != size:
        raise RuntimeError("assembled output has the wrong size")
    os.replace(temporary, output)
    for _, _, _, part in ranges:
        part.unlink()
    parts_dir.rmdir()
    elapsed = time.time() - started
    print(f"complete: {output} ({size} bytes, {elapsed:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
