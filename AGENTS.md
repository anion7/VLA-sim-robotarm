# AGENTS.md

## Cursor Cloud specific instructions

### Project Overview

This repository contains **RT-2 (Robotics Transformer 2)**, a Vision-Language-Action model implementation in PyTorch, intended for integration with **NVIDIA Isaac Sim** for robot arm simulation.

### Python Environment

- A **Python 3.11** virtual environment lives at `.venv/` (required by Isaac Sim 5.1.0).
- Always activate before running anything: `source .venv/bin/activate`
- Python 3.11 is installed from `ppa:deadsnakes/ppa`. System Python is 3.12; do not use it for this project.

### Key Commands

| Task | Command |
|------|---------|
| Activate venv | `source .venv/bin/activate` |
| Install RT-2 deps | `pip install -r requirements.txt` |
| Install Isaac Sim deps | `pip install -r requirements_isaacsim.txt` |
| Install dev extras | `pip install pytest matplotlib datasets` |
| Run tests | `python -m pytest tests/test.py -v` |
| Run RT-2 example | `python example.py` |
| Verify full env | `python verify_env.py` |

### Important Gotchas

- **NVIDIA Isaac Sim EULA:** On first import of `isaacsim`, a EULA prompt appears. Pipe `echo "Yes"` to accept non-interactively. Once accepted, it persists for the user profile.
- **Isaac Sim submodules need GPU:** `isaacsim.core`, `isaacsim.robot`, etc. are Omniverse Kit extensions that load only with the Kit runtime + NVIDIA RTX GPU. The pip packages are installed for IDE support and type checking; actual simulation requires GPU hardware.
- **`zetascale` API change:** `AutoregressiveWrapper` was renamed to `AutoRegressiveWrapper` in recent zetascale. The import in `rt2/model.py` is updated accordingly.
- **Model output is a tuple:** `AutoRegressiveWrapper.forward()` returns `(logits, loss)`, not a single tensor.
- **Undeclared transitive deps:** `matplotlib` and `datasets` are needed by `zetascale` at import time but not listed in `requirements.txt`. Install separately.
- **click version pin:** `isaacsim-kernel` requires `click==8.1.7`. After installing RT-2 deps (which may pull a newer click), run `pip install click==8.1.7` to fix.
