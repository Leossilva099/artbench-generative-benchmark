#!/usr/bin/env python3
import csv
import pickle
from pathlib import Path

import numpy as np
from datasets import Dataset as HFDataset
from datasets import DatasetDict, Features, Image, ClassLabel, load_dataset


KAGGLE_SOURCE_NAMES = {"kaggle", "local", "artbench10"}


def dataset_source_name(dataset_source, default_source="hf"):
    src = str(dataset_source).strip().lower()
    if not src:
        return str(default_source).strip().lower()
    return src


def _get_pickle_value(obj, key):
    if key in obj:
        return obj[key]
    bkey = key.encode("utf-8")
    if bkey in obj:
        return obj[bkey]
    raise KeyError(f"Missing key '{key}' in pickle object")


def _resolve_kaggle_paths(kaggle_root):
    root = Path(kaggle_root)
    csv_path = root / "ArtBench-10.csv"
    batch_dir = root / "artbench-10-python" / "artbench-10-batches-py"
    return root, csv_path, batch_dir


def load_kaggle_artbench10_splits(kaggle_root):
    root, csv_path, batch_dir = _resolve_kaggle_paths(kaggle_root)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Kaggle CSV not found: {csv_path}. "
            "Expected the original ArtBench-10 folder structure."
        )
    if not batch_dir.exists():
        raise FileNotFoundError(
            f"Kaggle CIFAR batches not found: {batch_dir}. "
            "Expected ArtBench-10/artbench-10-python/artbench-10-batches-py"
        )

    with open(batch_dir / "meta", "rb") as f:
        meta = pickle.load(f)
    styles = _get_pickle_value(meta, "styles")
    if not isinstance(styles, list) or len(styles) == 0:
        raise ValueError(f"Could not read class names from {batch_dir / 'meta'}")
    styles = [str(s).strip() for s in styles]
    style_to_id = {name: i for i, name in enumerate(styles)}

    csv_label_ids = {"train": {}, "test": {}}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"split", "label", "cifar_index"}
        missing = required.difference(set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"CSV is missing required columns {sorted(missing)}: {csv_path}")

        for row in reader:
            split = str(row.get("split", "")).strip().lower()
            if split not in csv_label_ids:
                continue

            label_name = str(row.get("label", "")).strip()
            if label_name not in style_to_id:
                raise ValueError(
                    f"Unknown label '{label_name}' in {csv_path}. "
                    f"Known labels: {styles}"
                )

            try:
                idx = int(row.get("cifar_index"))
            except Exception as exc:
                raise ValueError(f"Invalid cifar_index '{row.get('cifar_index')}' in {csv_path}") from exc

            csv_label_ids[split][idx] = int(style_to_id[label_name])

    def _load_batch(path):
        with open(path, "rb") as f:
            batch = pickle.load(f)
        data = np.asarray(_get_pickle_value(batch, "data"), dtype=np.uint8)
        labels = np.asarray(_get_pickle_value(batch, "labels"), dtype=np.int64)
        if data.ndim != 2 or data.shape[1] != 3072:
            raise ValueError(f"Unexpected data shape in {path}: {data.shape}")
        images = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        return images, labels

    train_images_chunks = []
    train_labels_chunks = []
    for batch_idx in range(1, 6):
        images, labels = _load_batch(batch_dir / f"data_batch_{batch_idx}")
        train_images_chunks.append(images)
        train_labels_chunks.append(labels)
    train_images = np.concatenate(train_images_chunks, axis=0)
    train_labels_raw = np.concatenate(train_labels_chunks, axis=0)
    test_images, test_labels_raw = _load_batch(batch_dir / "test_batch")

    def _labels_from_csv(split, n, labels_raw):
        ids = csv_label_ids[split]
        out = np.full((n,), -1, dtype=np.int64)
        for idx, label_id in ids.items():
            if idx < 0 or idx >= n:
                raise ValueError(
                    f"CSV {split} index {idx} out of bounds for {n} samples ({csv_path})"
                )
            out[idx] = int(label_id)
        missing = int(np.sum(out < 0))
        if missing > 0:
            raise ValueError(
                f"CSV {csv_path} is missing {missing} labels for split '{split}'."
            )
        mismatches = int(np.sum(out != labels_raw))
        if mismatches > 0:
            raise ValueError(
                f"CSV labels and batch labels disagree for {mismatches} samples in split '{split}'."
            )
        return out

    train_labels = _labels_from_csv("train", train_images.shape[0], train_labels_raw)
    test_labels = _labels_from_csv("test", test_images.shape[0], test_labels_raw)

    features = Features({
        "image": Image(),
        "label": ClassLabel(names=styles),
    })

    train_ds = HFDataset.from_dict(
        {
            "image": [train_images[i] for i in range(train_images.shape[0])],
            "label": train_labels.tolist(),
        },
        features=features,
    )
    test_ds = HFDataset.from_dict(
        {
            "image": [test_images[i] for i in range(test_images.shape[0])],
            "label": test_labels.tolist(),
        },
        features=features,
    )

    print(f"Dataset source: kaggle root='{root}'")
    return DatasetDict(train=train_ds, test=test_ds)


def resolve_dataset_splits(dataset_id, seed=42, dataset_source="hf", kaggle_root="ArtBench-10", default_source="hf"):
    source = dataset_source_name(dataset_source, default_source=default_source)
    if source in KAGGLE_SOURCE_NAMES:
        return load_kaggle_artbench10_splits(kaggle_root)
    if source != "hf":
        raise ValueError(
            f"Invalid dataset_source='{dataset_source}'. Use 'kaggle' or 'hf'."
        )

    ds = load_dataset(dataset_id)
    if isinstance(ds, dict) and not isinstance(ds, DatasetDict):
        ds = DatasetDict(ds)

    if "train" not in ds:
        first = next(iter(ds.keys()))
        spl = ds[first].train_test_split(test_size=1 / 6, seed=seed)
        ds = DatasetDict(train=spl["train"], test=spl["test"])
    elif "test" not in ds:
        spl = ds["train"].train_test_split(test_size=1 / 6, seed=seed)
        ds = DatasetDict(train=spl["train"], test=spl["test"])

    print(f"Dataset source: hf dataset_id='{dataset_id}'")
    return ds
