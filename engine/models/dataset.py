from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class DatasetProfile:
    n_classes: int
    total_samples: int
    samples_per_class: Dict[str, int]
    min_samples_per_class: int
    max_samples_per_class: int
    imbalance_ratio: float

    median_image_size: Tuple[int, int]
    size_variance: float

    grayscale_ratio: float

    pixel_mean: List[float]
    pixel_std: List[float]

    estimated_mb: float

    corrupted_files: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.n_classes < 1:
            raise ValueError("Dataset must have at least one class")