# AGENTS.md

## Cursor Cloud specific instructions

### Project Overview

This repository contains three major components for VLA (Vision-Language-Action) robot simulation:

1. **RT-2 (Robotics Transformer 2)** — VLA model at repo root (`rt2/`).
2. **NVIDIA Isaac GR00T N1.6** — Open VLA foundation model at `Isaac-GR00T/`.
3. **NVIDIA Isaac Lab** — GPU-accelerated robotics research framework at `IsaacLab/`.

### Environments

| Component | Venv | Python | Activate |
|-----------|------|--------|----------|
| RT-2 + Isaac Lab | `.venv/` (root) | 3.11 | `source .venv/bin/activate` |
| Isaac GR00T | `Isaac-GR00T/.venv/` | 3.10 | `source Isaac-GR00T/.venv/bin/activate` |

### RT-2 Commands

| Task | Command |
|------|---------|
| Run tests | `source .venv/bin/activate && python -m pytest tests/test.py -v` |
| Run example | `source .venv/bin/activate && python example.py` |

### Isaac Lab Commands

Isaac Lab is installed as editable pip packages from `IsaacLab/source/`. See `IsaacLab/README.md` for full docs.

| Task | Command |
|------|---------|
| Lint | `source .venv/bin/activate && cd IsaacLab && ruff check source/` |

### Isaac GR00T Commands

GR00T uses `uv` for dependency management. See `Isaac-GR00T/README.md` for full docs.

| Task | Command |
|------|---------|
| Install deps | `cd Isaac-GR00T && uv sync --python 3.10 && uv pip install -e .` |
| Run tests | `cd Isaac-GR00T && source .venv/bin/activate && python -m pytest tests/ -v -m "not gpu"` |
| Lint | `cd Isaac-GR00T && source .venv/bin/activate && ruff check gr00t/` |

### Important Gotchas

- **Two separate venvs:** Root `.venv/` (Python 3.11) for RT-2 + Isaac Lab; `Isaac-GR00T/.venv/` (Python 3.10) for GR00T. Do not mix.
- **GR00T requires Python 3.10:** The `flash-attn` wheel in the lockfile is pinned for `cp310`. Using 3.11 or 3.12 will fail `uv sync`.
- **GR00T uses `uv`:** Always use `uv sync` / `uv pip install` inside `Isaac-GR00T/`.
- **Isaac Lab pxr dependency:** `isaaclab.utils`, `isaaclab.envs` etc. import `pxr` (USD) which requires the Isaac Sim Omniverse runtime + GPU. Packages are installed for IDE development; full runtime requires an NVIDIA RTX GPU.
- **`zetascale` API:** `AutoregressiveWrapper` → `AutoRegressiveWrapper`; RT-2 model output is `(logits, loss)` tuple.
- **ruff** is used for linting both Isaac Lab and GR00T code.
- **GR00T external_dependencies:** Has git submodules (LIBERO, robocasa, etc.) — optional for core dev. Init with `git submodule update --init --recursive` if needed.
