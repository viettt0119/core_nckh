"""``eyegaze export | verify-bundle | golden`` (export needs the ``[ml]`` and ``[export]`` extras)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from eyegaze.contracts import CONTRACTS_DIR


def _fail(message: str) -> typer.Exit:
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(code=1)


def export_command(
    run: Annotated[Path, typer.Option(help="Run directory containing best.pt")],
    out: Annotated[Path, typer.Option(help="Output bundle directory")],
    parity_samples: Annotated[int, typer.Option(min=1, help="Samples per parity check")] = 64,
    manifest: Annotated[Path | None, typer.Option(help="manifest.parquet, adds a real-frame parity check")] = None,
    frames_root: Annotated[Path | None, typer.Option(help="GazeCapture root (with --manifest)")] = None,
) -> None:
    """Export a run to a verified LiteRT bundle (model + spec + model card)."""
    from eyegaze.adapters.checkpoint import CheckpointError
    from eyegaze.adapters.litert import LiteRtExportError
    from eyegaze.application.export import export_run

    if (manifest is None) != (frames_root is None):
        raise _fail("--manifest and --frames-root must be given together")
    try:
        result = export_run(run, out, parity_samples, manifest, frames_root)
    except (CheckpointError, LiteRtExportError, ValueError) as exc:
        raise _fail(str(exc)) from exc
    card = result.card
    typer.echo(
        json.dumps(
            {
                "out_dir": str(result.out_dir),
                "model_sha256": card.model_sha256,
                "spec_sha256": card.spec_sha256,
                "model_bytes": card.model_bytes,
                "parity": [p.model_dump() for p in card.parity],
            },
            indent=2,
        )
    )


def verify_bundle_command(
    bundle: Annotated[Path, typer.Option("--dir", help="Bundle directory")],
) -> None:
    """Check a bundle's SHA-256 hashes against its model card."""
    from pydantic import ValidationError

    from eyegaze.adapters.litert import LiteRtExportError
    from eyegaze.application.export import verify_bundle

    try:
        card = verify_bundle(bundle)
    except (OSError, ValidationError, LiteRtExportError) as exc:
        raise _fail(str(exc)) from exc
    typer.echo(f"ok: {card.model_file} ({card.model_sha256[:12]}), spec {card.spec_sha256[:12]}")


def golden_command(
    out: Annotated[Path, typer.Option(help="Fixture directory")] = CONTRACTS_DIR / "golden",
    with_model: Annotated[bool, typer.Option(help="Also export the golden LiteRT model (needs [export])")] = False,
) -> None:
    """Regenerate the cross-language golden fixtures from the Python reference."""
    from eyegaze.application.golden import generate, generate_golden_model

    artifacts = generate(out)
    if with_model:
        generate_golden_model(out, artifacts)
    typer.echo(f"wrote {len(artifacts)} cases to {out}")


def register(app: typer.Typer) -> None:
    app.command("export")(export_command)
    app.command("verify-bundle")(verify_bundle_command)
    app.command("golden")(golden_command)
