"""
Environment verification script for VLA-sim-robotarm.

Checks that both RT-2 and Isaac Sim packages are installed and importable.
Run with: python verify_env.py
"""
import importlib.metadata
import sys


def check_package(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main():
    print("=" * 60)
    print("VLA-sim-robotarm Environment Verification")
    print(f"Python {sys.version}")
    print("=" * 60)

    all_ok = True

    # RT-2 dependencies
    print("\n[RT-2 Dependencies]")
    rt2_pkgs = ["torch", "zetascale", "transformers", "einops", "beartype"]
    for pkg in rt2_pkgs:
        ver = check_package(pkg)
        status = f"v{ver}" if ver else "MISSING"
        if not ver:
            all_ok = False
        print(f"  {pkg:30s} {status}")

    # RT-2 model import
    print("\n[RT-2 Model]")
    try:
        from rt2.model import RT2
        import torch

        model = RT2()
        img = torch.randn(1, 3, 256, 256)
        caption = torch.randint(0, 20000, (1, 1024))
        logits, loss = model(img, caption)
        print(f"  Forward pass OK — logits {logits.shape}, loss {loss.item():.4f}")
    except Exception as e:
        print(f"  Forward pass FAILED: {e}")
        all_ok = False

    # Isaac Sim packages
    print("\n[Isaac Sim Packages]")
    isaacsim_pkgs = [
        "isaacsim", "isaacsim-kernel", "isaacsim-core",
        "isaacsim-robot", "isaacsim-sensor", "isaacsim-utils",
    ]
    for pkg in isaacsim_pkgs:
        ver = check_package(pkg)
        status = f"v{ver}" if ver else "MISSING"
        if not ver:
            all_ok = False
        print(f"  {pkg:30s} {status}")

    # GPU check
    print("\n[GPU Status]")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("  CUDA not available (Isaac Sim runtime requires NVIDIA RTX GPU)")
    except Exception:
        print("  Could not check CUDA status")

    print("\n" + "=" * 60)
    if all_ok:
        print("All packages installed. Environment ready for development.")
    else:
        print("Some packages missing. See above for details.")
    print("=" * 60)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
