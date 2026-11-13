"""Execute the compatibility notebook, preserving partial output on failure."""
import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    notebook = nbformat.read(args.notebook, as_version=4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        NotebookClient(
            notebook, timeout=args.timeout, kernel_name="python3",
            resources={"metadata": {"path": str(args.output.parent)}},
        ).execute()
    finally:
        nbformat.write(notebook, args.output)
        print(f"Saved notebook output to {args.output}", flush=True)


if __name__ == "__main__":
    main()
