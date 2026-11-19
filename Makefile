SHELL := /bin/bash
RESEARCH := research
UV := uv --directory $(RESEARCH)
LANDMARKER_URL := https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

.PHONY: help setup lint format typecheck test check models clean

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

setup:  ## Create the research virtualenv (core + dev tools)
	$(UV) sync --python 3.11

lint:  ## Ruff lint + format check
	$(UV) run ruff check src tests
	$(UV) run ruff format --check src tests

format:  ## Apply Ruff formatting and safe fixes
	$(UV) run ruff check --fix src tests
	$(UV) run ruff format src tests

typecheck:  ## mypy (strict on domain/)
	$(UV) run mypy

test:  ## Run the test suite
	$(UV) run pytest

check: lint typecheck test  ## Everything CI runs for research/

models:  ## Download the MediaPipe Face Landmarker model into models/
	mkdir -p models
	curl -fsSL -o models/face_landmarker.task $(LANDMARKER_URL)

clean:  ## Remove caches
	find $(RESEARCH) -type d \( -name __pycache__ -o -name .pytest_cache -o -name .mypy_cache -o -name .ruff_cache \) -prune -exec rm -rf {} +
