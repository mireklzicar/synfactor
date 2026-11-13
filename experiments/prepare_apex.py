"""Fetch pinned upstream APEX outside this repo; defer its optional top-k import.

APEX is CC BY-NC 4.0. Its code/weights are not redistributed by SynFactor.
The only source modification makes the CUDA search extension a lazy import;
the native encoder, linear probe and factorizer are unchanged.
"""
import argparse
from pathlib import Path
import subprocess

REVISION = "80e544eff6bbc2b09022a4f780fc9665256a884e"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if not args.path.exists():
        subprocess.run(["git", "clone", "https://github.com/NumerionLabs/apex", str(args.path)], check=True)
        subprocess.run(["git", "-C", str(args.path), "checkout", REVISION], check=True)
    rev = subprocess.check_output(["git", "-C", str(args.path), "rev-parse", "HEAD"], text=True).strip()
    if rev != REVISION:
        raise ValueError(f"Expected upstream {REVISION}; got {rev}")
    source = args.path / "apex/nn/apex.py"
    old = "from apex_topk import topk\n"
    new = "def topk(*args, **kwargs):\n    from apex_topk import topk as native_topk\n    return native_topk(*args, **kwargs)\n"
    text = source.read_text()
    if old in text:
        source.write_text(text.replace(old, new, 1))
    elif new not in text:
        raise ValueError("Unexpected source; refusing to patch")
    print(f"Prepared {args.path} at {rev}; lazy top-k import only.")


if __name__ == "__main__":
    main()
