# AGENTS.md

## Cursor Cloud specific instructions

### Repository State

This repository (`VLA-sim-robotarm`) is currently a blank project stub containing only a `README.md`. There is no source code, no dependency manifests, no build system, no tests, and no runnable application.

### Project Intent

Per the README, the project targets **VLA (Vision-Language-Action) models** integrated with **NVIDIA Isaac Sim** for robot arm simulation. When code is added, expect:

- **Python** as the primary language (VLA models, Isaac Sim scripting).
- Dependencies on **PyTorch** or similar ML frameworks for VLA model inference.
- **NVIDIA Isaac Sim** (requires an NVIDIA GPU with RTX support and the Omniverse platform) — this cannot run on standard cloud VMs without GPU access.
- Possible **ROS 2** integration for robot control messaging.

### Environment Notes

- Python 3.12 is available in the VM.
- No GPU is available in the Cursor Cloud VM, so NVIDIA Isaac Sim and GPU-dependent ML inference cannot run here.
- Once dependencies are added (e.g., `requirements.txt`, `pyproject.toml`), update the VM startup script accordingly.

### Lint / Test / Build / Run

No lint, test, build, or run commands exist yet. When they are added, document the commands here.
