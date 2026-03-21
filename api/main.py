"""FastAPI application entrypoint for LayerWise."""

from fastapi import FastAPI


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="LayerWise API")
    return app


app = create_app()

