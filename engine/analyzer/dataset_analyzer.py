import random
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

from engine.models.dataset import DatasetProfile


class DatasetAnalyzer:
    def analyze(self, dataset_path: str) -> DatasetProfile:
        try:
            dataset_dir = Path(dataset_path)
            if not dataset_dir.exists():
                raise FileNotFoundError(f"Dataset path {dataset_dir} does not exist")

            return self._analyze(dataset_dir)
        except Exception as e:
            raise RuntimeError("Failed to analyze dataset") from e

    def _analyze(self, dataset_path: Path) -> DatasetProfile:
        class_map = self._scan_structure(dataset_path)
        corrupted = self._flag_corrupted(class_map)

        stats = self._compute_basic_stats(class_map)
        img_stats = self._compute_image_stats(class_map)

        return DatasetProfile(
            **stats,
            **img_stats,
            corrupted_files=corrupted
        )

    def _scan_structure(self, path: Path) -> Dict[str, List[Path]]:
        class_map = {}

        for class_dir in path.iterdir():
            if class_dir.is_dir():
                files = [
                    f for f in class_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']
                ]
                class_map[class_dir.name] = files
            
            if not class_map:
                raise ValueError("No class folders found in dataset")

        return class_map



    def _compute_basic_stats(self, class_map):
        samples_per_class = {
        cls: len(files) for cls, files in class_map.items()
    }

        total_samples = sum(samples_per_class.values())
        n_classes = len(samples_per_class)

        min_samples = min(samples_per_class.values())
        max_samples = max(samples_per_class.values())

        imbalance_ratio = max_samples / min_samples if min_samples > 0 else 0

        return {
            "n_classes": n_classes,
            "total_samples": total_samples,
            "samples_per_class": samples_per_class,
            "min_samples_per_class": min_samples,
            "max_samples_per_class": max_samples,
            "imbalance_ratio": imbalance_ratio,
        }

    def _compute_image_stats(self, class_map, sample_size=500):

        all_files = [f for files in class_map.values() for f in files]

        sampled = random.sample(all_files, min(len(all_files), sample_size))

        sizes = []
        pixels = []
        grayscale_count = 0
        total_bytes = 0

        for img_path in sampled:
            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    arr = np.array(img) / 255.0

                    sizes.append(img.size)
                    pixels.append(arr.reshape(-1, 3))

                    if self._detect_grayscale(arr):
                        grayscale_count += 1

                    total_bytes += img_path.stat().st_size
            except Exception:
                continue

        pixels = np.vstack(pixels)

        mean = pixels.mean(axis=0).tolist()
        std = pixels.std(axis=0).tolist()

        sizes_np = np.array(sizes)
        median_size = tuple(np.median(sizes_np, axis=0).astype(int))
        size_variance = float(np.std(sizes_np))

        grayscale_ratio = grayscale_count / len(sampled)

        estimated_mb = total_bytes / (1024 * 1024)

        return {
            "median_image_size": median_size,
            "size_variance": size_variance,
            "grayscale_ratio": grayscale_ratio,
            "pixel_mean": mean,
            "pixel_std": std,
            "estimated_mb": estimated_mb,
        }

    def _detect_grayscale(self, arr: np.ndarray) -> bool:
        if arr.ndim != 3 or arr.shape[2] != 3:
            return False

        r = arr[:, :, 0].astype(np.float32)
        g = arr[:, :, 1].astype(np.float32)
        b = arr[:, :, 2].astype(np.float32)

        diff_rg = np.mean(np.abs(r - g))
        diff_rb = np.mean(np.abs(r - b))
        diff_gb = np.mean(np.abs(g - b))
        avg_diff = (diff_rg + diff_rb + diff_gb) / 3.0

        threshold = 5.0 / 255.0 if float(np.max(arr)) <= 1.0 else 5.0
        return avg_diff < threshold

    def _flag_corrupted(self, class_map):
        corrupted = []

        for files in class_map.values():
            for f in files:
                try:
                    with Image.open(f):
                        pass
                except Exception:
                    corrupted.append(str(f))

        return corrupted
