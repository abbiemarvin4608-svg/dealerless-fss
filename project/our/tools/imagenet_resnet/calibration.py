"""Freeze a class-balanced ImageNet training manifest independent of validation."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader


def build_manifest(train_dir, val_dir, per_class=2, seed=20261015):
    if per_class < 1:
        raise ValueError("per_class must be positive")
    train, validation = ImageFolder(str(train_dir)), ImageFolder(str(val_dir))
    if len(train.classes) != 1000 or train.class_to_idx != validation.class_to_idx:
        raise ValueError("calibration requires the same 1000 ImageNet classes as validation")
    val_ids = {Path(path).name for path, _ in validation.samples}
    groups = [[] for _ in train.classes]
    for path, label in train.samples:
        identifier = Path(path).name
        if identifier in val_ids or identifier.startswith("ILSVRC2012_val_"):
            raise ValueError("training source contains an ImageNet validation image: " + identifier)
        groups[label].append(path)
    rng = random.Random(seed)
    selected = []
    for label, paths in enumerate(groups):
        if len(paths) < per_class:
            raise ValueError("insufficient training images for " + train.classes[label])
        for path in sorted(rng.sample(sorted(paths), per_class)):
            selected.append(dict(path=str(Path(path).resolve()), label=label,
                                 wnid=train.classes[label], image_id=Path(path).name))
    document = dict(schema="imagenet-training-calibration-v1", split="train", seed=seed,
                per_class=per_class, samples=len(selected), classes=train.classes,
                class_to_idx=train.class_to_idx, train_dir=str(Path(train_dir).resolve()),
                validation_dir=str(Path(val_dir).resolve()),
                selection="seeded sample without replacement within each class",
                validation_overlap=0, entries=selected)
    source_path = Path(train_dir).parent / "DOWNLOAD_STATUS.json"
    if source_path.exists():
        source = json.loads(source_path.read_text())
        if source.get("status") != "complete" or Path(source.get("imagefolder", "")).resolve() != Path(train_dir).resolve():
            raise ValueError("downloaded calibration source is incomplete or inconsistent")
        document["source_subset"] = {key: source[key] for key in
            ("source", "split", "classes", "samples", "per_class", "selection")}
    return document


class ManifestDataset(Dataset):
    def __init__(self, manifest_path, transform=None, validation_dir=None):
        self.manifest = json.loads(Path(manifest_path).read_text())
        document = self.manifest
        if document.get("schema") != "imagenet-training-calibration-v1" or document.get("split") != "train":
            raise ValueError("expected an independent training calibration manifest")
        self.classes, self.class_to_idx = document["classes"], document["class_to_idx"]
        if len(self.classes) != 1000 or self.class_to_idx != {name: i for i, name in enumerate(self.classes)}:
            raise ValueError("invalid ImageNet class mapping")
        self.samples = [(entry["path"], entry["label"]) for entry in document["entries"]]
        counts = [0] * 1000
        ids = set()
        for entry in document["entries"]:
            path, label = Path(entry["path"]), entry["label"]
            if not 0 <= label < 1000 or entry["wnid"] != self.classes[label]:
                raise ValueError("invalid calibration label")
            if not path.is_file() or path.name != entry["image_id"]:
                raise ValueError("missing or inconsistent calibration image: " + str(path))
            if path.name in ids or path.name.startswith("ILSVRC2012_val_"):
                raise ValueError("duplicate or validation image in calibration")
            ids.add(path.name)
            counts[label] += 1
        if len(self.samples) != document["samples"] or set(counts) != {document["per_class"]}:
            raise ValueError("calibration must be exactly class balanced")
        if validation_dir is not None:
            validation = ImageFolder(str(validation_dir))
            if validation.class_to_idx != self.class_to_idx:
                raise ValueError("calibration/validation class mapping differs")
            if ids & {Path(path).name for path, _ in validation.samples}:
                raise ValueError("calibration overlaps validation")
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = default_loader(path)
        return self.transform(image) if self.transform else image, label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261015)
    args = parser.parse_args()
    document = build_manifest(args.train_dir, args.val_dir, args.per_class, args.seed)
    if args.output.exists():
        if json.loads(args.output.read_text()) != document:
            raise ValueError("existing frozen calibration manifest differs")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps({"manifest": str(args.output), "samples": document["samples"]}))


if __name__ == "__main__":
    main()
