# Repository guidelines

This is an independent Python training framework. All runtime modules must live under src/training_framework and use this namespace. Never introduce source imports, symlinks, dependencies, absolute paths or run artifacts pointing into another training project.

Keep algorithm models in application plugins. The built-in models are only small CPU-safe contract examples. BraTS dependencies must remain optional and lazily imported.

Use uv sync --extra dev --extra brats for development. Run uv run pytest, uv run ruff check src tests examples, uv run ruff format --check src tests examples, uv run mypy src and uv build before delivering framework changes. Keep generated data and runs untracked.

Tests must use synthetic data, temporary paths and deterministic small models. Do not depend on private datasets, server home directories or third-party source checkouts.
