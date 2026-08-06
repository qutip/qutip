import pathlib
import sys

PKG = pathlib.Path(__file__).resolve().parent.parent / "qutip"
SKIP = {"core/data/matmul", "core/cy/openmp/parfuncs"}
WIDTH = 22

modules = {}
for path in sorted(PKG.rglob("*.pyx")):
    rel = path.relative_to(PKG)
    if rel.with_suffix("").as_posix() in SKIP:
        continue
    if rel.name.startswith(("qtcoeff_", "compiled_coeff")):  # Coefficient strings meant to be compiled at run time
        continue

    reldir = rel.parent.as_posix()
    modules.setdefault("" if reldir == "." else reldir, []).append(rel.stem)


if "--pairs" in sys.argv:
    for reldir, names in sorted(modules.items()):
        for name in sorted(names):
            print(f"{reldir}:{name}")

# Just for manually running (debugging)
else:
    print("cy_modules = {")
    for reldir, names in sorted(modules.items()):
        key = f"  '{reldir}':".ljust(WIDTH)
        line = key + " ["
        for i, name in enumerate(sorted(names)):
            item = f"'{name}'" + ("]," if i == len(names) - 1 else ", ")
            if len(line) + len(item) > 79:
                print(line.rstrip())
                line = " " * (WIDTH + 2)
            line += item
        print(line)
        print("}")