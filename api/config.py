"""Configuration placeholders for the LayerWise API."""

from dataclasses import dataclass


@dataclass(slots=True)
class Settings:
    """Minimal API settings stub."""

    app_name: str = "LayerWise API"

