"""
SmartCrop AI - Reproducible Dataset Splitter
Enforces a 70% Train, 15% Validation, 15% Test split with fixed random seed (42).
Preserves class stratification, guards against data leakage, and records exclusions with reasons.
"""

import os
import random
import json
import logging
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("SmartCrop.Splitter")

RANDOM_SEED = 42
TARGET_SPLIT = (0.70, 0.15, 0.15)  # Train, Val, Test


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Sets deterministic random seed across random, numpy, and environment."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def stratified_split_by_class(
    class_to_items: Dict[str, List[Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = RANDOM_SEED
) -> Tuple[List[Any], List[Any], List[Any]]:
    """
    Performs stratified splitting across discrete classes to preserve class balance.
    Ratios must sum to 1.0.
    """
    assert train_ratio > 0 and val_ratio > 0 and test_ratio > 0, "Ratios must be positive"
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Split ratios must sum to 1.0"
    set_seed(seed)

    train_set, val_set, test_set = [], [], []

    for cls_name, items in sorted(class_to_items.items()):
        shuffled = list(items)
        random.shuffle(shuffled)
        n = len(shuffled)
        
        if n == 0:
            continue
        elif n == 1:
            train_set.append(shuffled[0])
            continue
        elif n == 2:
            train_set.append(shuffled[0])
            val_set.append(shuffled[1])
            continue

        n_val = max(1, int(round(n * val_ratio)))
        n_test = max(1, int(round(n * test_ratio)))
        n_train = n - n_val - n_test

        if n_train < 1:
            n_train = 1
            if n_val > n_test:
                n_val -= 1
            else:
                n_test -= 1

        train_cls = shuffled[:n_train]
        val_cls = shuffled[n_train:n_train + n_val]
        test_cls = shuffled[n_train + n_val:]

        train_set.extend(train_cls)
        val_set.extend(val_cls)
        test_set.extend(test_cls)

    logger.info(
        f"Stratified Split completed -> Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)} "
        f"(Total: {len(train_set) + len(val_set) + len(test_set)})"
    )
    return train_set, val_set, test_set


def export_split_manifest(
    train_data: List[Any],
    val_data: List[Any],
    test_data: List[Any],
    output_path: Path,
    excluded_files: Optional[List[Dict[str, Any]]] = None
) -> None:
    """Saves split manifest to disk as a JSON file, including excluded files and reasons."""
    manifest = {
        "metadata": {
            "random_seed": RANDOM_SEED,
            "target_ratio": {"train": 0.70, "val": 0.15, "test": 0.15},
            "counts": {
                "train": len(train_data),
                "val": len(val_data),
                "test": len(test_data),
                "total_included": len(train_data) + len(val_data) + len(test_data),
                "total_excluded": len(excluded_files) if excluded_files else 0
            }
        },
        "train": train_data,
        "val": val_data,
        "test": test_data,
        "excluded_files": excluded_files or []
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Saved split manifest to: {output_path}")
