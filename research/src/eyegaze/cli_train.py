"""``eyegaze train | evaluate | ablation-table`` (needs the ``[ml]`` extra)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer


def _fail(message: str) -> typer.Exit:
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(code=1)


def train_command(
    config: Annotated[Path, typer.Option(help="Experiment YAML (may set `base:`)")],
    override: Annotated[list[str] | None, typer.Option("--set", help="Override, e.g. optim.lr=0.001")] = None,
    resume: Annotated[bool, typer.Option(help="Continue from runs/<name>/last.pt")] = False,
) -> None:
    """Train EyeGazeNet v2."""
    from eyegaze.adapters.checkpoint import CheckpointError
    from eyegaze.adapters.torch_data import DataError
    from eyegaze.application.train import train
    from eyegaze.config import ConfigError, load_config

    try:
        cfg = load_config(config, override or [])
        result = train(cfg, resume=resume)
    except (ConfigError, DataError, CheckpointError, FileExistsError, FloatingPointError) as exc:
        raise _fail(str(exc)) from exc
    typer.echo(
        json.dumps(
            {
                "run_dir": str(result.run_dir),
                "best_epoch": result.best_epoch,
                "best_val_subject_mean_cm": result.best_val_subject_mean,
                "epochs_run": result.epochs_run,
                "stopped_early": result.stopped_early,
            },
            indent=2,
        )
    )


def evaluate_command(
    run: Annotated[Path, typer.Option(help="Run directory containing best.pt")],
    manifest: Annotated[Path, typer.Option(help="manifest.parquet")],
    frames_root: Annotated[Path, typer.Option(help="GazeCapture root holding the frames")],
    split: Annotated[str, typer.Option(help="train | val | test")] = "test",
    k: Annotated[list[int] | None, typer.Option("--k", help="Calibration targets to simulate")] = None,
    svr: Annotated[bool, typer.Option(help="Also simulate SVR calibration (E2 comparison)")] = False,
    limit: Annotated[int | None, typer.Option(min=1, help="Evaluate a random subset")] = None,
    batch_size: Annotated[int, typer.Option(min=1)] = 512,
    workers: Annotated[int, typer.Option(min=0)] = 2,
    device: Annotated[str, typer.Option()] = "auto",
) -> None:
    """Evaluate a run (E1 accuracy + E2 calibration simulation) and write eval_<split>.json."""
    from eyegaze.adapters.checkpoint import CheckpointError
    from eyegaze.adapters.torch_data import DataError
    from eyegaze.application.calibration_sim import CalibrationMethod
    from eyegaze.application.evaluate import evaluate_run

    methods = [CalibrationMethod.RIDGE, *([CalibrationMethod.SVR] if svr else [])]
    try:
        result = evaluate_run(
            run,
            manifest,
            frames_root,
            split=split,
            calibration_k=k or [5, 9, 13],
            methods=methods,
            batch_size=batch_size,
            num_workers=workers,
            limit=limit,
            device_name=device,
        )
    except (CheckpointError, DataError, ValueError) as exc:
        raise _fail(str(exc)) from exc
    acc = result["accuracy"]
    typer.echo(
        json.dumps(
            {
                "subject_mean_cm": acc["subject_mean"],
                "frame_mean_cm": acc["frame"]["mean"],
                "calibration": result["calibration"],
            },
            indent=2,
        )
    )


def ablation_table_command(
    runs: Annotated[list[Path], typer.Argument(help="Run directories")],
    baseline: Annotated[str, typer.Option(help="Run name used as reference")] = "base",
    split: Annotated[str, typer.Option()] = "test",
    k: Annotated[int, typer.Option(help="Calibration size shown in the table")] = 9,
) -> None:
    """Print the E3 ablation table (Markdown) from evaluated runs."""
    from eyegaze.application.evaluate import ablation_table

    try:
        typer.echo(ablation_table(runs, baseline, split, k), nl=False)
    except ValueError as exc:
        raise _fail(str(exc)) from exc


def register(app: typer.Typer) -> None:
    app.command("train")(train_command)
    app.command("evaluate")(evaluate_command)
    app.command("ablation-table")(ablation_table_command)
