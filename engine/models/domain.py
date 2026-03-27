from dataclasses import dataclass   
from enum import Enum
from typing import List, Optional

class Domain(str, Enum):
    NATURAL_IMAGES = "Natural Images"
    MEDICAL_IMAGES = "Medical Images"
    SATELLITE_IMAGES = "Satellite Images"
    DOCUMENT = "DOCUMENT"
    MICROSCOPY = "MICROSCOPY"
    UNKNOWN = "UNKNOWN"

@dataclass
class DomainResults:
    domain: Domain
    confidence: float
    signals: List[str]
    alternative: Optional[Domain] = None