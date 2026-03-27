import pytest
from unittest.mock import Mock

from engine.models.domain import Domain
from engine.detector.domain_detector import DomainDetector


@pytest.fixture
def detector():
    return DomainDetector()


def test_detects_natural_dataset(detector):
    # Arrange
    profile = Mock()
    profile.grayscale_ratio = 0.1  # Rule D4 fires (NATURAL += 2)
    profile.median_image_size = (800, 600)
    profile.image_size_std = (200, 150)
    profile.pixel_mean = 0.5
    profile.pixel_std = 0.2
    
    # Act
    result = detector.detect(profile)
    
    # Assert
    assert result.domain == Domain.NATURAL_IMAGES
    assert result.confidence >= 0.5


def test_detects_medical_dataset(detector):
    # Arrange
    profile = Mock()
    profile.grayscale_ratio = 0.9  # Rule D1 fires (MEDICAL += 2)
    profile.pixel_mean = 0.2       # Rule D3 fires (MEDICAL += 2)
    profile.pixel_std = 0.1        # Rule D3 fires
    profile.median_image_size = (512, 512)
    profile.image_size_std = (0, 0) # Rule D2 & D5 fire (MEDICAL += 2 total)
    
    # Act
    result = detector.detect(profile)
    
    # Assert
    assert result.domain == Domain.MEDICAL_IMAGING
    assert result.confidence >= 0.5


def test_signals_list_is_never_empty(detector):
    # Arrange: Use an ambiguous profile that triggers no specific strong rules
    profile = Mock()
    profile.grayscale_ratio = 0.5
    profile.pixel_mean = 0.5
    profile.pixel_std = 0.2
    profile.median_image_size = (300, 400)
    profile.image_size_std = (100, 100)
    
    # Act
    result = detector.detect(profile)
    
    # Assert
    assert len(result.signals) >= 2


def test_unknown_fallback_when_ambiguous(detector):
    # Arrange: Create an ambiguous scenario where votes are split
    profile = Mock()
    profile.grayscale_ratio = 0.75  # Rule D1: MEDICAL (+2), DOCUMENT (+1), NATURAL (-1)
    profile.pixel_mean = 0.8        # Rule D3: DOCUMENT (+2)
    profile.pixel_std = 0.2
    profile.median_image_size = (500, 500)
    profile.image_size_std = (100, 100) 
    
    # In this scenario, Top Domain is DOCUMENT (3 pts), Second is MEDICAL (2 pts)
    # Confidence: (3 - 2) / max(3, 1) = 0.33 -> Fallback to UNKNOWN
    
    # Act
    result = detector.detect(profile)
    
    # Assert
    assert result.domain == Domain.UNKNOWN
    assert result.alternative == Domain.DOCUMENT # Highest scorer before fallback
    assert result.confidence < 0.5
