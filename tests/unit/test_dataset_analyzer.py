
import pytest
from PIL import Image

from engine.analyzer.dataset_analyzer import DatasetAnalyzer


# Helper: create fake image
def create_image(path, size=(64, 64), color=(255, 0, 0), grayscale=False):
    if grayscale:
        img = Image.new("L", size, color=128)
    else:
        img = Image.new("RGB", size, color=color)
    img.save(path)

# 1. Basic dataset test
def test_valid_dataset(tmp_path):
    for cls in ["cat", "dog"]:
        cls_dir = tmp_path / cls
        cls_dir.mkdir()
        for i in range(5):
            create_image(cls_dir / f"{i}.jpg")

    analyzer = DatasetAnalyzer()
    profile = analyzer.analyze(str(tmp_path))

    assert profile.n_classes == 2
    assert profile.total_samples == 10
    assert profile.min_samples_per_class == 5
    assert profile.max_samples_per_class == 5


# 2. Imbalance detection
def test_imbalance_detection(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "c").mkdir()

    for i in range(100):
        create_image(tmp_path / "a" / f"{i}.jpg")

    for i in range(10):
        create_image(tmp_path / "b" / f"{i}.jpg")

    for i in range(5):
        create_image(tmp_path / "c" / f"{i}.jpg")

    analyzer = DatasetAnalyzer()
    profile = analyzer.analyze(str(tmp_path))

    assert profile.imbalance_ratio == 20.0


# 3. Grayscale detection
def test_grayscale_detection(tmp_path):
    cls_dir = tmp_path / "gray"
    cls_dir.mkdir()

    for i in range(5):
        create_image(cls_dir / f"{i}.jpg", grayscale=True)

    analyzer = DatasetAnalyzer()
    profile = analyzer.analyze(str(tmp_path))

    assert profile.grayscale_ratio == 1.0


# 4. Mixed grayscale + RGB
def test_mixed_color_dataset(tmp_path):
    cls_dir = tmp_path / "mixed"
    cls_dir.mkdir()

    for i in range(5):
        create_image(cls_dir / f"rgb_{i}.jpg")

    for i in range(5):
        create_image(cls_dir / f"gray_{i}.jpg", grayscale=True)

    analyzer = DatasetAnalyzer()
    profile = analyzer.analyze(str(tmp_path))

    assert 0 < profile.grayscale_ratio < 1


# 5. Corrupted file detection
def test_corrupted_files(tmp_path):
    cls_dir = tmp_path / "data"
    cls_dir.mkdir()

    create_image(cls_dir / "valid.jpg")

    # corrupted file
    (cls_dir / "bad.jpg").write_bytes(b"not an image")

    analyzer = DatasetAnalyzer()
    profile = analyzer.analyze(str(tmp_path))

    assert len(profile.corrupted_files) == 1


# 6. Image size statistics
def test_image_size_stats(tmp_path):
    cls_dir = tmp_path / "data"
    cls_dir.mkdir()

    create_image(cls_dir / "1.jpg", size=(64, 64))
    create_image(cls_dir / "2.jpg", size=(128, 128))

    analyzer = DatasetAnalyzer()
    profile = analyzer.analyze(str(tmp_path))

    assert isinstance(profile.median_image_size, tuple)
    assert profile.size_variance >= 0


# 7. Pixel statistics
def test_pixel_stats(tmp_path):
    cls_dir = tmp_path / "data"
    cls_dir.mkdir()

    create_image(cls_dir / "1.jpg", color=(255, 0, 0))

    analyzer = DatasetAnalyzer()
    profile = analyzer.analyze(str(tmp_path))

    assert len(profile.pixel_mean) == 3
    assert len(profile.pixel_std) == 3


# 8. Empty class handling
def test_empty_class(tmp_path):
    (tmp_path / "empty").mkdir()

    analyzer = DatasetAnalyzer()

    with pytest.raises(Exception):
        analyzer.analyze(str(tmp_path))


# 9. Output completeness
def test_output_fields(tmp_path):
    cls_dir = tmp_path / "data"
    cls_dir.mkdir()

    create_image(cls_dir / "1.jpg")

    analyzer = DatasetAnalyzer()
    profile = analyzer.analyze(str(tmp_path))

    required_fields = [
        "n_classes",
        "total_samples",
        "samples_per_class",
        "imbalance_ratio",
        "median_image_size",
        "pixel_mean",
        "pixel_std",
    ]

    for field in required_fields:
        assert hasattr(profile, field)