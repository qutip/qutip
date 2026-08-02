#!/usr/bin/env python3
"""Git hash suffix for dev builds e.g. 5.4.0.dev0+g1a2b3c4."""
import pathlib
import subprocess

root = pathlib.Path(__file__).resolve().parent.parent
version = (root / "VERSION").read_text().strip()

if "dev" in version:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--verify", "--short=7", "HEAD"],
            check=True, capture_output=True, cwd=root,
        )
        version += "+" + out.stdout.decode().strip()
    except (subprocess.CalledProcessError, OSError):
        pass

print(version)