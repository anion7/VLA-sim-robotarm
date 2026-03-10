# AGENTS.md

## Cursor Cloud specific instructions

### Project Overview

This repository contains **RT-2 (Robotics Transformer 2)**, a Vision-Language-Action model implementation in PyTorch. The model uses a ViTransformer encoder and an autoregressive Transformer decoder to translate vision + language inputs into robot action tokens.

### Key Commands

| Task | Command |
|------|---------|
| Install deps | `pip install -r requirements.txt && pip install pytest matplotlib datasets` |
| Run tests | `python3 -m pytest tests/test.py -v` |
| Run example | `python3 example.py` |

### Important Gotchas

- **No GPU required for CPU inference.** The model runs on CPU (slow but functional). No CUDA/GPU is available in the Cursor Cloud VM.
- **`zetascale` API change:** The upstream `zetascale` package renamed `AutoregressiveWrapper` to `AutoRegressiveWrapper` (capital R). The import in `rt2/model.py` has been updated accordingly.
- **Model output is a tuple:** `AutoRegressiveWrapper.forward()` returns `(logits, loss)`, not a single tensor. The `example.py` and `tests/test.py` have been updated to handle this.
- **Undeclared transitive deps:** `matplotlib` and `datasets` are needed by `zetascale` at import time but are not listed in `requirements.txt`. They must be installed separately.
- **PATH:** Scripts installed by pip go to `~/.local/bin`. Run `export PATH="$HOME/.local/bin:$PATH"` if needed, or use `python3 -m pytest` instead of bare `pytest`.
