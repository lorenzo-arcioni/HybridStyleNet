"""
data/
-----
Data loading, preprocessing and augmentation for RAG-ColorNet.
"""

from .raw_pipeline import load_image, load_pair, gamma_encode, gamma_decode
from .datasets import (
    FiveKDataset,
    PPR10KDataset,
    LightroomPresetsDataset,
    CombinedDataset,
    build_pretrain_dataset,
)
from .photographer_dataset import (
    PhotographerDataset,
    split_dataset,
    build_photographer_datasets,
)
from .task_sampler import Task, TaskSampler, SyntheticInterpolationDataset, build_task_sampler
from .augmentations import (
    GeometricAug,
    StyleAug,
    CrossPhotographerAug,
    Compose,
    interpolate_styles,
    random_crop_pair,
    build_train_transforms,
)

__all__ = [
    # raw pipeline
    "load_image", "load_pair", "gamma_encode", "gamma_decode",
    # datasets
    "FiveKDataset", "PPR10KDataset", "LightroomPresetsDataset",
    "CombinedDataset", "build_pretrain_dataset",
    # photographer
    "PhotographerDataset", "split_dataset", "build_photographer_datasets",
    # task sampler
    "Task", "TaskSampler", "SyntheticInterpolationDataset", "build_task_sampler",
    # augmentations
    "GeometricAug", "StyleAug", "CrossPhotographerAug", "Compose",
    "interpolate_styles", "random_crop_pair", "build_train_transforms",
]
