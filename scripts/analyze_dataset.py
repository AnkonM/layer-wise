import json
from pathlib import Path
from typing import Optional

import typer

from engine.analyzer.dataset_analyzer import DatasetAnalyzer

app = typer.Typer(help="LayerWise Dataset Analyzer CLI")

# Convert DatasetProfile Dataclass to serializable JSON dict
def _serialize_profile(profile) -> dict:
    def convert(value):
        import numpy as np

        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}

        elif isinstance(value, list):
            return [convert(v) for v in value]

        elif isinstance(value, tuple):
            return [convert(v) for v in value]

        elif isinstance(value, np.ndarray):
            return value.tolist()

        elif isinstance(value, (np.integer,)):
            return int(value)

        elif isinstance(value, (np.floating,)):
            return float(value)

        else:
            return value
    
    data = convert(profile.__dict__.copy())

    if "median_image_size" in data and isinstance(data["median_image_size"], tuple):
        data["median_image_size"] = list(data["median_image_size"])

    return data

@app.command()
def analyze(
    path: str = typer.Option(..., "--path", "-p", help="Path to dataset directory"),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Optional path to save JSON output"
    ),
):
    dataset_path = Path(path)

    # Input Validation
    if not dataset_path.exists():
        if not dataset_path.is_dir():
            typer.secho(f"Path is not a valid directory: {dataset_path}", fg=typer.colors.RED)
            typer.Exit(code=1)
        typer.secho(f"Path does not exist: {dataset_path}", fg=typer.colors.RED) 
        typer.Exit(code=1)


    analyzer = DatasetAnalyzer()

    try:
        # Call analyzer engine
        profile = analyzer.analyze(str(dataset_path))

        # Serialize output to JSON dict
        result = _serialize_profile(profile)
        json_output = json.dumps(result, indent=2)

        typer.echo(json_output)
        
        if output:
            output_path = Path(output)
            output_path.write_text(json_output)
            typer.secho(f"\n Saved output to {output_path}", fg=typer.colors.GREEN)

    except Exception as e:
        typer.secho("Failed to analyze dataset", fg=typer.colors.RED)
        typer.secho(f"Error: {str(e)}", fg=typer.colors.YELLOW)

        if typer.get_app_dir("layerwise"):
            typer.secho(f"\nDebug: {repr(e)}", fg = typer.colors.BLACK)

        raise typer.Exit(code=1)

def main() -> None:
    app()


if __name__ == "__main__":
    main()

