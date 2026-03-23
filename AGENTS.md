# AGENTS.md

## Cursor Cloud specific instructions

### Project Overview

This repository contains two major components:

1. **RT-2 (Robotics Transformer 2)** — A VLA model implementation at the repo root (`rt2/`).
2. **NVIDIA Isaac GR00T N1.6** — An open VLA foundation model for generalist humanoid robots at `Isaac-GR00T/`.

### Environments

There are **two separate Python virtual environments**:

| Component | Venv | Python | Activate |
|-----------|------|--------|----------|
| RT-2 + Isaac Sim | `.venv/` (root) | 3.11 | `source .venv/bin/activate` |
| Isaac GR00T | `Isaac-GR00T/.venv/` | 3.12 | `source Isaac-GR00T/.venv/bin/activate` |

Always activate the correct venv before running commands for each component.

### RT-2 Commands

| Task | Command |
|------|---------|
| Install deps | `source .venv/bin/activate && pip install -r requirements.txt` |
| Run tests | `source .venv/bin/activate && python -m pytest tests/test.py -v` |
| Run example | `source .venv/bin/activate && python example.py` |

### Isaac GR00T Commands

| Task | Command |
|------|---------|
| Install deps | `cd Isaac-GR00T && uv sync && uv pip install -e .` |
| Install dev extras | `cd Isaac-GR00T && uv pip install pytest pytest-timeout ruff` |
| Run tests | `cd Isaac-GR00T && source .venv/bin/activate && python -m pytest tests/ -v -m "not gpu"` |
| Lint | `cd Isaac-GR00T && source .venv/bin/activate && python -m ruff check gr00t/` |
| Verify env | `cd Isaac-GR00T && source .venv/bin/activate && python -c "import gr00t; print('OK')"` |

### Important Gotchas

- **Two separate venvs:** RT-2 uses `.venv/` at repo root (Python 3.11); GR00T uses `Isaac-GR00T/.venv/` (Python 3.12). Do not mix them.
- **GR00T uses `uv`:** Dependencies are managed via `uv sync` (lockfile at `Isaac-GR00T/uv.lock`). Do NOT use `pip install` for GR00T deps; use `uv pip install` inside the GR00T dir.
- **GPU required for model inference:** Both RT-2 (for production use) and GR00T require NVIDIA GPU for actual model inference. CPU mode works for imports, tests, and development.
- **GR00T `flash-attn` and `tensorrt`:** These GPU-specific packages install via `uv sync` but require CUDA at runtime.
- **NVIDIA Isaac Sim EULA:** On first import of `isaacsim`, pipe `echo "Yes"` to accept non-interactively.
- **`zetascale` API:** `AutoregressiveWrapper` renamed to `AutoRegressiveWrapper`; model output is `(logits, loss)` tuple.
- **GR00T external_dependencies:** The repo has git submodules for LIBERO, robocasa, etc. These are optional for core development; initialize with `git submodule update --init --recursive` if needed.
- **ruff** is used for linting GR00T code (config in `Isaac-GR00T/pyproject.toml`).
