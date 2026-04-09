"""Dataset and dataloader utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from src.utils import seed_worker


class WithIndexDataset(Dataset):
    """Wrap a torchvision dataset to return sample indices."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        return image, label, index


@dataclass
class DatasetBundle:
    """Container for dataset splits and metadata."""

    pretrain: Dataset
    probe_train: Dataset
    probe_val: Dataset
    test: Dataset
    num_classes: int
    channels: int
    image_size: int


def build_train_transform(image_size: int) -> transforms.Compose:
    """Modest augmentations for self-supervised pretraining."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
        ]
    )


def build_eval_transform(image_size: int) -> transforms.Compose:
    """Deterministic transform for evaluation and export."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


def _load_cifar10(root: Path, train_transform, eval_transform):
    pretrain_train = datasets.CIFAR10(root=root, train=True, transform=train_transform, download=True)
    eval_train = datasets.CIFAR10(root=root, train=True, transform=eval_transform, download=False)
    test = datasets.CIFAR10(root=root, train=False, transform=eval_transform, download=True)
    return pretrain_train, eval_train, test, 10, 3


def _load_stl10(root: Path, train_transform, eval_transform):
    pretrain_train = datasets.STL10(root=root, split="train", transform=train_transform, download=True)
    eval_train = datasets.STL10(root=root, split="train", transform=eval_transform, download=False)
    test = datasets.STL10(root=root, split="test", transform=eval_transform, download=True)
    return pretrain_train, eval_train, test, 10, 3


def build_datasets(config: Dict) -> DatasetBundle:
    """Build all dataset splits used across pretraining and evaluation."""
    dataset_name = config["dataset"]["name"].lower()
    image_size = int(config["dataset"]["image_size"])
    val_split = int(config["dataset"]["val_split"])
    data_root = Path(config["paths"]["data_dir"])
    train_transform = build_train_transform(image_size)
    eval_transform = build_eval_transform(image_size)

    if dataset_name == "cifar10":
        pretrain_train, eval_train, test, num_classes, channels = _load_cifar10(
            data_root, train_transform, eval_transform
        )
    elif dataset_name == "stl10":
        pretrain_train, eval_train, test, num_classes, channels = _load_stl10(
            data_root, train_transform, eval_transform
        )
    else:
        raise ValueError("dataset.name must be either 'cifar10' or 'stl10'.")

    generator = torch.Generator().manual_seed(int(config["seed"]))
    train_size = len(eval_train) - val_split
    if train_size <= 0:
        raise ValueError("dataset.val_split is too large for the selected dataset.")

    all_indices = torch.randperm(len(eval_train), generator=generator).tolist()
    probe_train_indices = all_indices[:train_size]
    probe_val_indices = all_indices[train_size:]

    return DatasetBundle(
        pretrain=Subset(WithIndexDataset(pretrain_train), list(range(len(pretrain_train)))),
        probe_train=Subset(WithIndexDataset(eval_train), probe_train_indices),
        probe_val=Subset(WithIndexDataset(eval_train), probe_val_indices),
        test=WithIndexDataset(test),
        num_classes=num_classes,
        channels=channels,
        image_size=image_size,
    )


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    drop_last: bool = False,
) -> DataLoader:
    """Build a standard dataloader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
        worker_init_fn=seed_worker,
    )

