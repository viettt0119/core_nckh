"""Command-line entry point: ``eyegaze <command>``.

Commands are added milestone by milestone; each one is a thin shell over an
application-layer use case so the same logic is callable from notebooks/tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from eyegaze import __version__
from eyegaze.cli_data import data_app
from eyegaze.contracts import DEFAULT_SPEC_PATH, ContractError, load_spec, spec_hash
from eyegaze.log import configure_logging

app = typer.Typer(add_completion=False, no_args_is_help=True, help="EyeGaze research toolkit.")
app.add_typer(data_app, name="data")


@app.callback()
def _main(
    log_level: Annotated[str, typer.Option(help="DEBUG, INFO, WARNING, ERROR")] = "INFO",
    log_json: Annotated[bool, typer.Option(help="Emit JSON log lines")] = False,
) -> None:
    configure_logging(log_level, log_json)


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command()
def spec(
    path: Annotated[Path, typer.Option(help="Preprocess spec JSON")] = DEFAULT_SPEC_PATH,
) -> None:
    """Validate a preprocess spec and print its fingerprint."""
    try:
        parsed = load_spec(path)
        digest = spec_hash(path)
    except ContractError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    summary = {
        "path": str(path),
        "version": parsed.version,
        "sha256": digest,
        "eye_size": parsed.eye.size,
        "face_size": parsed.face.size,
        "geometry_dim": parsed.geometry.dim,
    }
    typer.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    app()
