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
    manual_src = Path(__file__).parent / "docs" / "hector_manual_v3.2.pdf"
    if manual_src.exists():
        shutil.copy(manual_src, dest / "hector_manual_v3.2.pdf")
    print(f"Examples copied to: {dest.resolve()}")
    print("  ex1  Synthetic GNSS: removeoutliers → estimatetrend → PSD")
    print("  ex2  Monthly sea-level data from Cascais tide gauge")
    print("  ex3  Synthetic coloured noise: simulatenoise → estimate_all_trends")
    print("  ex4  Post-seismic relaxation")
    print("  ex5  Single-station offset detection (synthetic GNSS station)")
    print("  ex6  Multi-station offset detection from NGL tenv files")
    print("  ex7  Piecewise linear (multi-trend) estimation")
    print("  ex8  Toeplitz factorisation: Levinson vs Generalised Schur (Jupyter)")
    if (dest / "paper_scripts").is_dir():
        print("  paper_scripts  Numerical experiments of the paper")
        print('                 "Faster analysis of GNSS time series"')
        print("                 (see paper_scripts/README.md; driver: run_all.py)")
