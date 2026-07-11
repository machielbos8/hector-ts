import shutil
import sys
from pathlib import Path


def main():
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("hector-examples")
    if dest.exists():
        sys.exit(
            f"Destination already exists: {dest.resolve()}\n"
            "Remove it first or choose a different name."
        )
    src = Path(__file__).parent / "examples"
    if not src.is_dir():
        sys.exit("Examples directory not found in the installed package.")
    shutil.copytree(src, dest)
    print(f"Examples copied to: {dest.resolve()}")
    print("  ex1  Synthetic GNSS: removeoutliers → estimatetrend → PSD")
    print("  ex2  Monthly sea-level data from Cascais tide gauge")
    print("  ex3  Multi-station GNSS batch processing")
    print("  ex4  Trend estimation with known offsets")
    print("  ex5  Offset detection on a real GNSS station")
    print("  ex6  Multi-station offset detection from NGL tenv files")
    print("  ex7  Piecewise linear (multi-trend) estimation")
    print("  ex8  Toeplitz factorisation: Levinson vs Generalised Schur (Jupyter)")
