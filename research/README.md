# eyegaze (research package)

Python side of EyeGaze: dataset building, training, evaluation and LiteRT export.
See the [repository README](../README.md) and [architecture](../docs/architecture.md).

```bash
uv sync --python 3.11            # core + dev tools
uv sync --extra data --extra ml  # + MediaPipe/OpenCV/pandas and PyTorch
uv run pytest
uv run eyegaze --help
```

Layout: `src/eyegaze/domain` (pure logic, strict typing) · `application` (use cases) ·
`adapters` (files, MediaPipe, PyTorch, LiteRT) · `tests/unit`, `tests/integration`.
