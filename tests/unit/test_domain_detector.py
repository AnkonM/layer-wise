import pytest
from engine.models.dataset import DatasetProfile
from engine.models.domain import Domain
from engine.detector.domain_detector import DomainDetector

# Helpers
def make_profile(**overrides) -> DatasetProfile:
    """
    Constructs a DatasetProfile with neutral baseline values.
    Tests override only the fields they intend to exercise.
    Baseline is designed to cause all 5 rules to MISS when not overridden,
    except D4 which always fires (see module docstring above).
    """
    defaults = dict(
        # Required structural fields
        n_classes=10,
        total_samples=1000,
        samples_per_class={"class_0": 100},
        min_samples_per_class=100,
        max_samples_per_class=100,
        imbalance_ratio=1.0,
        # Geometry
        median_image_size=(256, 256),
        size_variance=100.0,
        # D1: neutral (0.2 <= gs <= 0.7 → MISS)
        grayscale_ratio=0.5,
        # D2: neutral (not square-tight, not extreme, not high-std → MISS)
        aspect_ratio_median=1.2,
        aspect_ratio_std=0.2,
        # D3: neutral (lum in mid band, no channel spread → MISS)
        pixel_mean=[0.5, 0.5, 0.5],
        pixel_std=[0.2, 0.2, 0.2],
        # D4: cannot be neutral — always fires. Set to medium range.
        color_diversity=6.0,   # [5.0, 6.5) → DOCUMENT+2, SATELLITE+1
        # D5: neutral (res_std not < 0.05, not < 0.1 + large dim, not > 0.5 + colorful → MISS)
        resolution_std=0.2,
        estimated_mb=50.0,
    )
    defaults.update(overrides)
    return DatasetProfile(**defaults)
@pytest.fixture
def detector() -> DomainDetector:
    return DomainDetector()
# ---------------------------------------------------------------------------
# Test 1: Strong MEDICAL signal
#
# D1: grayscale_ratio=0.95 >= 0.9 (VERY_HIGH)  →  MEDICAL+3, MICROSCOPY+2
# D2: ar_median=1.0, ar_std=0.05 (square+tight) →  MEDICAL+2, MICROSCOPY+2
# D3: lum = 0.299*0.2 + 0.587*0.2 + 0.114*0.2 = 0.2
#     0.2 < 0.35, lum_std = 0.1 <= 0.2          →  MEDICAL+2
# D4: color_diversity=3.0 < 5.0                 →  MEDICAL+2
# D5: resolution_std=0.02 < 0.05                →  MEDICAL+1, DOCUMENT+1
#
# Final: MEDICAL=10, MICROSCOPY=4, DOCUMENT=1
# gap = 10-4 = 6  →  confidence = min(0.5 + 0.6, 0.95) = 0.95
# alternative = MICROSCOPY  (gap=6 > 1 → None)...
#   Wait: gap=6 > 1, so alternative = None per detector logic.
# ---------------------------------------------------------------------------
def test_medical_dataset(detector: DomainDetector) -> None:
    # Arrange
    profile = make_profile(
        grayscale_ratio=0.95,         # D1: VERY_HIGH branch
        aspect_ratio_median=1.0,      # D2: square
        aspect_ratio_std=0.05,        # D2: tight
        pixel_mean=[0.2, 0.2, 0.2],  # D3: dark (lum=0.2 < 0.35)
        pixel_std=[0.1, 0.1, 0.1],   # D3: lum_std=0.1 <= 0.2
        color_diversity=3.0,          # D4: low entropy
        resolution_std=0.02,          # D5: very tight
    )
    # Act
    result = detector.detect(profile)
    # Assert
    assert result.domain == Domain.MEDICAL
    assert result.confidence == 0.95
    assert result.alternative is None           # gap=6 > 1 threshold
    assert len(result.signals) == 5             # one signal per rule
    assert any("D1 HIT" in s for s in result.signals)
    assert any("D2 HIT" in s for s in result.signals)
    assert any("D3 HIT" in s for s in result.signals)
    assert any("D4 HIT" in s for s in result.signals)
    assert any("D5 HIT" in s for s in result.signals)
# ---------------------------------------------------------------------------
# Test 2: Strong NATURAL signal
#
# D1: grayscale_ratio=0.05 < 0.2 (LOW)          →  NATURAL+2, SATELLITE+1
# D2: ar_median=1.5, ar_std=0.5
#     w_over_h = 1.5 (median >= 1.0)
#     1.3 <= 1.5 <= 2.0 BUT std=0.5 NOT < 0.25  → not satellite branch
#     std=0.5 > 0.4                              →  NATURAL+1
# D3: pixel_mean=[0.5, 0.35, 0.2]
#     lum = 0.299*0.5 + 0.587*0.35 + 0.114*0.2 = 0.378
#     channel_spread = 0.5 - 0.2 = 0.3 > 0.05
#     0.35 <= 0.378 <= 0.7                       →  NATURAL+1
# D4: color_diversity=7.0 >= 6.5                →  NATURAL+2
# D5: resolution_std=0.8 > 0.5
#     grayscale_ratio=0.05 < 0.3                →  NATURAL+1
#
# Final: NATURAL=7, SATELLITE=1
# gap = 6  →  confidence = 0.95 (capped)
# alternative = None  (gap=6 > 1)
# ---------------------------------------------------------------------------
def test_natural_dataset(detector: DomainDetector) -> None:
    # Arrange
    profile = make_profile(
        grayscale_ratio=0.05,               # D1: LOW branch + D5 colorful condition
        aspect_ratio_median=1.5,            # D2: not square, w_over_h=1.5
        aspect_ratio_std=0.5,               # D2: high std (> 0.4) → NATURAL
        pixel_mean=[0.5, 0.35, 0.2],       # D3: mid lum, high channel spread
        pixel_std=[0.2, 0.2, 0.2],
        color_diversity=7.0,                # D4: high entropy
        resolution_std=0.8,                 # D5: high std + low grayscale
    )
    # Act
    result = detector.detect(profile)
    # Assert
    assert result.domain == Domain.NATURAL
    assert result.confidence == 0.95
    assert result.alternative is None       # gap=6 > 1
    assert any("D1 HIT" in s for s in result.signals)
    assert any("D2 HIT" in s for s in result.signals)
    assert any("D3 HIT" in s for s in result.signals)
    assert any("D4 HIT" in s for s in result.signals)
    assert any("D5 HIT" in s for s in result.signals)
# ---------------------------------------------------------------------------
# Test 3: Strong DOCUMENT signal
#
# D1: grayscale_ratio=0.75 in [0.7, 0.9) (HIGH) →  MEDICAL+2, DOCUMENT+1, NATURAL-1
# D2: ar_median=3.0 > 2.5 (tall/narrow)          →  DOCUMENT+2
# D3: lum = 0.85*(0.299+0.587+0.114) = 0.85
#     lum > 0.7 (bright)                          →  DOCUMENT+2
# D4: color_diversity=5.5 in [5.0, 6.5) (medium) →  DOCUMENT+2, SATELLITE+1
# D5: resolution_std=0.02 < 0.05                  →  MEDICAL+1, DOCUMENT+1
#
# Final: DOCUMENT=8, MEDICAL=3, SATELLITE=1, NATURAL=-1, MICROSCOPY=0
# gap = 8-3 = 5  →  confidence = min(0.5 + 0.5, 0.95) = 0.95 (capped)
# alternative = None  (gap=5 > 1)
# ---------------------------------------------------------------------------
def test_document_dataset(detector: DomainDetector) -> None:
    # Arrange
    profile = make_profile(
        grayscale_ratio=0.75,               # D1: HIGH branch
        aspect_ratio_median=3.0,            # D2: tall/narrow  > 2.5
        aspect_ratio_std=0.1,
        pixel_mean=[0.85, 0.85, 0.85],     # D3: bright (lum=0.85 > 0.7)
        pixel_std=[0.05, 0.05, 0.05],
        color_diversity=5.5,                # D4: medium entropy
        resolution_std=0.02,                # D5: very tight
    )
    # Act
    result = detector.detect(profile)
    # Assert
    assert result.domain == Domain.DOCUMENT
    assert result.confidence == 0.95
    assert result.alternative is None       # gap=5 > 1
    assert any("D1 HIT" in s for s in result.signals)
    assert any("D2 HIT" in s for s in result.signals)
    assert any("D3 HIT" in s for s in result.signals)
    assert any("D4 HIT" in s for s in result.signals)
    assert any("D5 HIT" in s for s in result.signals)
# ---------------------------------------------------------------------------
# Test 4: signals list is never empty
#
# Even with all rules in MISS mode, each rule appends a MISS signal.
# D4 always fires (HIT), so at minimum 5 signals are always produced.
# ---------------------------------------------------------------------------
def test_signals_list_never_empty(detector: DomainDetector) -> None:
    # Arrange: use baseline — D1/D2/D3/D5 all MISS, D4 HIT
    profile = make_profile()
    # Act
    result = detector.detect(profile)
    # Assert
    assert len(result.signals) >= 5         # 5 rules, each adds at least one string
# ---------------------------------------------------------------------------
# Test 5: Type guard — loud failure on wrong input type
#
# detect() must raise TypeError immediately if not given a DatasetProfile.
# This guards the pipeline boundary between M1 and M2.
# ---------------------------------------------------------------------------
def test_type_guard_raises_on_wrong_input(detector: DomainDetector) -> None:
    class FakeProfile:
        grayscale_ratio = 0.5
    with pytest.raises(TypeError, match="DomainDetector.detect\\(\\) expected DatasetProfile"):
        detector.detect(FakeProfile())   # type: ignore[arg-type]
    with pytest.raises(TypeError):
        detector.detect({"grayscale_ratio": 0.5})   # type: ignore[arg-type]
    with pytest.raises(TypeError):
        detector.detect(None)   # type: ignore[arg-type]