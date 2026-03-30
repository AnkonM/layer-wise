from engine.models.domain import Domain, DomainResults
from engine.models.dataset import DatasetProfile

# Threshold constants
# D1
_GRAYSCALE_LOW = 0.2
_GRAYSCALE_HIGH = 0.7
_GRAYSCALE_VERY_HIGH = 0.9



class DomainDetector:
    def detect(self, profile: any) -> DomainResults:
        
        if not isinstance(profile, DatasetProfile):
            raise TypeError(
                f"DomainDetector.detect() expected DatasetProfile, "
                f"got {type(profile).__name__!r}. "
                f"Please run DatasetAnalyzer first."
            )
        
        scores: dict[Domain, float] = {
            d:0.0 for d in Domain if d is not Domain.UNKNOWN
        }

        signals: list[str] = []

        # Apply rules in order
        self._d1_grayscale(profile, scores, signals)
        

    def _d1_grayscale(
        self, 
        profile: DatasetProfile, 
        scores: dict[Domain, float], 
        signals: list[str]
    ) -> None:
        """
        Rule D1: Grayscale intensity
        Source field: DatasetProfile.grayscale_ratio (float, 0.1-1.0)
        """

        gs = profile.grayscale_ratio

        # gs sanity check
        if not (0.0 <= gs <= 1.0):
            signals.append(f"D1 skipped: Grayscale ratio={gs!r} out of range [0, 1].")
            return

        
        if gs >= _GRAYSCALE_VERY_HIGH:
            scores[Domain.MEDICAL] += 3.0
            scores[Domain.MICROSCOPY] += 2.0
            signals.append(
                f"D1 HIT: Grayscale ratio={gs:.3f} > {_GRAYSCALE_VERY_HIGH}"
            )
        elif gs >= _GRAYSCALE_HIGH:
            scores[Domain.MEDICAL] += 2.0
            scores[Domain.DOCUMENT] += 1.0
            scores[Domain.NATURAL] -= 1.0
            signals.append(
                f"D1 HIT: Grayscale ratio={gs:.3f} > {_GRAYSCALE_HIGH}"
            )
        elif gs < _GRAYSCALE_LOW:
            scores[Domain.NATURAL] += 2.0
            scores[Domain.SATELLITE] += 1.0
            signals.append(
                f"D1 HIT: Grayscale ratio={gs:.3f} < {_GRAYSCALE_LOW}"
            )

def _d2_aspect_ratio(
    self,
    profile: DatasetProfile,
    scores: dict[Domain, float],
    signals: list[str]
) -> None:

    median = profile.aspect_ratio_median
    std = profile.aspect_ratio_std

    if 0.9 <= median <= 1.1 and std < 0.15:
        scores[Domain.MEDICAL] += 2.0
        scores[Domain.MICROSCOPY] += 2.0
        signals.append(
            f"D2 HIT: Aspect ratio median={median:.3f} near 1.0 and size variance={std:.3f} small"
        )
    
    if median > 2.5 or (1/median) > 2.5:
        scores[Domain.DOCUMENT] += 2.0
        signals.append(
            f"D2 HIT: Aspect ratio median={median:.3f} implying tall/narrow images"
        )
    
    w_over_h = 1.0 / median if median < 1.0 else median
    if 1.3 <= w_over_h <= 2.0 and std < 0.25:
        scores[Domain.SATELLITE] += 1.0
        signals.append(
            f"D2 HIT: Aspect ratio median={median:.3f} "
        )

    if std > 0.4:
        scores[Domain.NATURAL] += 1
        signals.append(
            f"D2 HIT: High aspect ratio std={std:.3f} implies organic/ unsanitised images"
        )
    
    return None