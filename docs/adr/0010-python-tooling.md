# ADR-0010: Python tooling

- **Status:** Accepted
- **Date:** 2026-11-19

## Context

Reproducibility and code quality must be visible to reviewers.

## Decision

uv for environments, Python 3.11, Ruff (lint + format), mypy (strict on `domain/`), pytest + hypothesis, Typer CLI, Pydantic v2 at boundaries (domain stays stdlib dataclasses), structlog JSON logs. Experiment tracking = run directory with config snapshot, `metrics.json` and model card (no MLflow/W&B: YAGNI).

## Consequences

`make check` gives one command for lint/type/test locally and in CI.
