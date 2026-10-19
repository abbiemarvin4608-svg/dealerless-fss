#!/usr/bin/env bash
set -euo pipefail

# Download and extract the official ILSVRC2012 classification data.  Downloads
# are resumable.  The validation images are reorganized into wnid directories
# using the official devkit ground truth, which makes torchvision ImageFolder's
# label order agree with the pretrained ImageNet-1K classifier.

if [[ $# -ne 1 ]]; then
  echo "usage: $0 DATA_ROOT" >&2
  exit 2
fi

data_root=$(realpath -m "$1")
archive_dir="$data_root/archives"
train_dir="$data_root/train"
val_dir="$data_root/val"
mkdir -p "$archive_dir" "$train_dir" "$val_dir"

base_url="https://image-net.org/data/ILSVRC/2012"
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
train_tar="$archive_dir/ILSVRC2012_img_train.tar"
val_tar="$archive_dir/ILSVRC2012_img_val.tar"
devkit_tar="$archive_dir/ILSVRC2012_devkit_t12.tar.gz"

wget -c -O "$devkit_tar" "$base_url/ILSVRC2012_devkit_t12.tar.gz"
python3 "$script_dir/download_ranges.py" \
  "$base_url/ILSVRC2012_img_val.tar" "$val_tar" --connections 8
python3 "$script_dir/download_ranges.py" \
  "$base_url/ILSVRC2012_img_train.tar" "$train_tar" --connections 8

# Expected official archive sizes catch interrupted or HTML/error downloads.
[[ $(stat -c %s "$train_tar") -eq 147897477120 ]]
[[ $(stat -c %s "$val_tar") -eq 6744924160 ]]

if ! find "$train_dir" -mindepth 2 -type f -print -quit | grep -q .; then
  tmp_train=$(mktemp -d "$data_root/.train_extract.XXXXXX")
  trap 'rm -rf -- "$tmp_train"' EXIT
  tar -xf "$train_tar" -C "$tmp_train"
  find "$tmp_train" -maxdepth 1 -type f -name '*.tar' -print0 |
    while IFS= read -r -d '' class_tar; do
      wnid=$(basename "$class_tar" .tar)
      mkdir -p "$train_dir/$wnid"
      tar -xf "$class_tar" -C "$train_dir/$wnid"
    done
  rm -rf -- "$tmp_train"
  trap - EXIT
fi

if ! find "$val_dir" -mindepth 2 -type f -print -quit | grep -q .; then
  tmp_val=$(mktemp -d "$data_root/.val_extract.XXXXXX")
  trap 'rm -rf -- "$tmp_val"' EXIT
  mkdir -p "$tmp_val/images" "$tmp_val/devkit"
  tar -xf "$val_tar" -C "$tmp_val/images"
  tar -xzf "$devkit_tar" -C "$tmp_val/devkit"
  python3 - "$tmp_val" "$val_dir" <<'PY'
import shutil
import sys
from pathlib import Path

import scipy.io

tmp = Path(sys.argv[1])
destination = Path(sys.argv[2])
devkit = tmp / "devkit" / "ILSVRC2012_devkit_t12"
ground_truth = [
    int(line) for line in
    (devkit / "data" / "ILSVRC2012_validation_ground_truth.txt").read_text().splitlines()
]
meta = scipy.io.loadmat(devkit / "data" / "meta.mat", squeeze_me=True)["synsets"]
index_to_wnid = {}
for synset in meta:
    ilsvrc_id = int(synset[0])
    wnid = str(synset[1])
    num_children = int(synset[4])
    if num_children == 0 and ilsvrc_id <= 1000:
        index_to_wnid[ilsvrc_id] = wnid
if len(index_to_wnid) != 1000 or len(ground_truth) != 50000:
    raise RuntimeError(
        f"invalid devkit: {len(index_to_wnid)} classes, {len(ground_truth)} labels"
    )
for image_index, class_index in enumerate(ground_truth, start=1):
    wnid = index_to_wnid[class_index]
    class_dir = destination / wnid
    class_dir.mkdir(parents=True, exist_ok=True)
    source = tmp / "images" / f"ILSVRC2012_val_{image_index:08d}.JPEG"
    shutil.move(str(source), class_dir / source.name)
PY
  rm -rf -- "$tmp_val"
  trap - EXIT
fi

train_count=$(find "$train_dir" -type f | wc -l)
val_count=$(find "$val_dir" -type f | wc -l)
class_count=$(find "$train_dir" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "ImageNet-1K prepared: train=$train_count val=$val_count classes=$class_count"
[[ "$train_count" -eq 1281167 ]]
[[ "$val_count" -eq 50000 ]]
[[ "$class_count" -eq 1000 ]]
