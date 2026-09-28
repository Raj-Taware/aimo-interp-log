"""Package the self-perturb solution into a Codabench ZIP.

    python scripts/build_bundle.py            # small track (adds small.txt)
    python scripts/build_bundle.py --main     # main track

Layout inside the ZIP, all at the root as the harness requires:
solution.py, selfperturb_infer.py, selfperturb/ (the package), probe_artifacts/.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parent.parent
SOLUTION = ROOT / "solutions" / "self-perturb"
PACKAGE = ROOT / "src" / "selfperturb"
ARTIFACTS = ROOT / "artifacts" / "probe_artifacts"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", action="store_true", help="build for the main track")
    arguments = parser.parse_args()

    suffix = "" if arguments.main else "-small"
    destination = ROOT / "dist" / f"solution-self-perturb{suffix}.zip"
    destination.parent.mkdir(exist_ok=True)

    with ZipFile(destination, "w", ZIP_DEFLATED) as bundle:
        for path in sorted(SOLUTION.glob("*.py")):
            bundle.write(path, path.name)
        for path in sorted(PACKAGE.glob("*.py")):
            bundle.write(path, f"selfperturb/{path.name}")
        for path in sorted(ARTIFACTS.glob("*.npz")) if ARTIFACTS.is_dir() else []:
            bundle.write(path, f"probe_artifacts/{path.name}")
        if not arguments.main:
            bundle.writestr("small.txt", "")

    with ZipFile(destination) as bundle:
        names = bundle.namelist()
    print(f"wrote {destination}")
    for name in names:
        print(f"  {name}")


if __name__ == "__main__":
    main()
