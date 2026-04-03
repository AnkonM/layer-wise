from dataclasses import dataclass, field   
from enum import Enum
from typing import List, Optional

class Domain(str, Enum):
    NATURAL = "NATURAL"
    MEDICAL = "MEDICAL"
    SATELLITE = "SATELLITE"
    DOCUMENT = "DOCUMENT"
    MICROSCOPY = "MICROSCOPY"
    UNKNOWN = "UNKNOWN"

@dataclass
class DomainResult:
    domain: Domain
    confidence: float
    signals: List[str] = field(default_factory=list)
    alternative: Optional[Domain] = None