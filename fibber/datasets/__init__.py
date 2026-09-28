from fibber.datasets.dataset_utils import (
    DatasetForTransformers, clip_sentence, get_dataset, subsample_dataset,
    verify_dataset)

__all__ = [
    "get_dataset",
    "subsample_dataset",
    "verify_dataset",
    "DatasetForTransformers",
    "get_demo_dataset",
    "builtin_datasets",
    "clip_sentence"]

builtin_datasets = ["RSTPReid","CUHK-PEDES","ICFG-PEDES"]
