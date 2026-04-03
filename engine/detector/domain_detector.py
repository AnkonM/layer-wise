from engine.models.domain import Domain, DomainResult
from engine.models.dataset import DatasetProfile

# D1
_GRAYSCALE_LOW = 0.2
_GRAYSCALE_HIGH = 0.7
_GRAYSCALE_VERY_HIGH = 0.9

def _confidence_score(scores: dict) -> float:
    sorted_scores = sorted(scores.values(), reverse = True)
    top = sorted_scores[0]
    second = sorted_scores[1] if len(sorted_scores) > 1 else 0
    gap = top - second
    return round(min(0.5 + gap * 0.1, 0.95), 2)

class DomainDetector:
    def detect(self, profile: DatasetProfile) -> DomainResult:
        
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
        self._d2_aspect_ratio(profile, scores, signals)
        self._d3_intensity(profile, scores, signals)
        self._d4_color_diversity(profile, scores, signals)
        self._d5_resolution_consistency(profile, scores, signals)

        sorted_domains = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_domain,    top_score    = sorted_domains[0]
        second_domain, second_score = sorted_domains[1]
        gap = top_score - second_score

        if top_score < 2:
            return DomainResult(
                domain=Domain.UNKNOWN,
                confidence=0.35,
                signals=signals,
                alternative=top_domain
            )
        
        confidence = _confidence_score(scores)
        alternative = second_domain if gap <= 1 else None

        return DomainResult(
            domain=top_domain,
            confidence=confidence,
            signals=signals,
            alternative=alternative
        )

        

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

        else:
            signals.append(
                f"D1 MISS: Grayscale ratio={gs:.3f}"
            )

    def _d2_aspect_ratio(
        self,
        profile: DatasetProfile,
        scores: dict[Domain, float],
        signals: list[str]
    ) -> None:

        median = profile.aspect_ratio_median
        std = profile.aspect_ratio_std
        w_over_h = 0.0 if median == 0 else 1.0 / median if median < 1.0 else median

        if 0.9 <= median <= 1.1 and std < 0.15:
            scores[Domain.MEDICAL] += 2.0
            scores[Domain.MICROSCOPY] += 2.0
            signals.append(
                f"D2 HIT: Aspect ratio median={median:.3f} near 1.0 and size variance={std:.3f} small"
            )
        
        elif median > 2.5 or (1/median) > 2.5:
            scores[Domain.DOCUMENT] += 2.0
            signals.append(
                f"D2 HIT: Aspect ratio median={median:.3f} implying tall/narrow images"
            )
        
        elif 1.3 <= w_over_h <= 2.0 and std < 0.25:
            scores[Domain.SATELLITE] += 1.0
            signals.append(
                f"D2 HIT: Aspect ratio median={median:.3f} "
            )

        elif std > 0.4:
            scores[Domain.NATURAL] += 1
            signals.append(
                f"D2 HIT: High aspect ratio std={std:.3f} implies organic/ unsanitised images"
            )
        
        else:
            signals.append(
                f"D2 MISS: Aspect ratio median={median:.3f} and std={std:.3f}"
            )

    def _d3_intensity(
        self,
        profile: DatasetProfile,
        scores: dict[Domain, float],
        signals: list[str]
    ) -> None:
        # Luminance proxy: standart ITU-R BT.601 (r, g, b)
        if len(profile.pixel_mean) == 3:
            r, g, b = profile.pixel_mean
            lum_mean = 0.299 * r + 0.587 * g + 0.114 * b
            r_std, g_std, b_std = profile.pixel_std
            lum_std = 0.299 * r_std + 0.587 * g_std + 0.114 * b_std
            # Alternate lum_std measure
            # lum_std = np.sqrt(0.299 * r_std**2 + 0.587 * g_std**2 + 0.114 * b_std**2)

            # Inter-channel spread: how different are the channel means?
            channel_spread = max(profile.pixel_mean) - min(profile.pixel_mean)
        else:
            # for grayscale images
            lum_mean = profile.pixel_mean[0]
            lum_std = profile.pixel_std[0] if profile.pixel_std else 0.0
            channel_spread = 0.0

        if lum_mean < 0.35:
            if not lum_std:
                signals.append(
                    f"D3 MISS: Empty profile.pixel_std: {profile.pixel_std}"
                )
                return
            if lum_std <= 0.2:
                scores[Domain.MEDICAL] += 2.0
                signals.append(
                    f"D3 HIT: Low luminance mean={lum_mean:.3f} and std={lum_std:.3f}"
                )
            elif lum_std > 0.2:
                scores[Domain.MICROSCOPY] += 1.0
                signals.append(
                    f"D3 HIT: Low luminance mean={lum_mean:.3f} and high std={lum_std:.3f}"
                )

        elif lum_mean > 0.7:
            scores[Domain.DOCUMENT] += 2.0
            signals.append(
                f"D3 HIT: High luminance mean={lum_mean:.3f}"
            )

        elif 0.35 <= lum_mean <= 0.7 and channel_spread > 0.05:
            scores[Domain.NATURAL] += 1.0
            signals.append(
                f"D3 HIT: Medium luminance mean={lum_mean:.3f}"
            )

        else:
            signals.append(
                f"D3 MISS: Luminance mean={lum_mean:.3f} and std={lum_std:.3f}"
            )

    def _d4_color_diversity(
        self,
        profile: DatasetProfile,
        scores: dict[Domain, float],
        signals: list[str]
    ) -> None:
        
        cd = profile.color_diversity

        if cd < 5.0:
            scores[Domain.MEDICAL] += 2.0
            signals.append(
                f"D4 HIT: Low color diversity={cd:.3f}"
            )

        elif 5.0 <= cd < 6.5:
            scores[Domain.DOCUMENT] += 2.0
            scores[Domain.SATELLITE] += 1.0
            signals.append(
                f"D4 HIT: Medium color diversity={cd:.3f}"
            )

        elif cd >= 6.5:
            scores[Domain.NATURAL] += 2.0
            signals.append(
                f"D4 HIT: High color diversity={cd:.3f}"
            )

        else:
            signals.append(
                f"D4 MISS: Color diversity={cd:.3f}"
            )

    def _d5_resolution_consistency(
        self,
        profile: DatasetProfile,
        scores: dict[Domain, float],
        signals: list[str]
    ) -> None:
       
        res_std = profile.resolution_std
        h, w = profile.median_image_size
        min_dim = min(h, w)
        gs = profile.grayscale_ratio

        if res_std < 0.05:
            scores[Domain.MEDICAL] += 1
            scores[Domain.DOCUMENT] += 1
            signals.append(
                f"D5 HIT: Low resolution std={res_std:.3f}"
            )

        elif res_std < 0.1 and min_dim > 512:
            scores[Domain.SATELLITE] += 1
            signals.append(
                f"D5 HIT: High resolution (min_dim={min_dim}, std={res_std:.3f})"
            )

        elif res_std > 0.5 and gs < 0.3:
            scores[Domain.NATURAL] += 1.0
            signals.append(
                f"D5 HIT: High resolution std={res_std:.3f}"
            )

        else:
            signals.append(
                f"D5 MISS: Resolution std={res_std:.3f}"
            )