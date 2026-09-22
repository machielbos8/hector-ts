#!/usr/bin/env python3
"""
setup_tools.py
--------------
Locate (or ask for) the optional external tools used by the timing
comparisons of run_all.py, and store the result in tools.json next to
this script:

  - the Hector C++ v2.2 `estimatetrend` binary (any name or location);
  - the `Rscript` executable of an R installation with the gmwmx2 package.

Run it once before `run_all.py --full` if the automatic detection of
`run_all.py --check` does not find your installations:

    python3 setup_tools.py            # interactive
    python3 setup_tools.py --status   # show current resolution and exit

Everything else in this directory is pure Python and needs no setup.
"""
# This file is part of Hector 3.1.
#
# Hector is distributed under a source-available license.
# It may be used free of charge for academic, research, and other
# non-commercial purposes.
# Commercial use is not permitted under this license and requires a
# separate agreement with TeroMovigo - Earth Innovation Lda.
# The complete license terms are provided in the LICENSE file.


import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
TOOLS_JSON = HERE / "tools.json"


def looks_like_python_v3(path):
    """True if `path` is the Hector v3 Python entry point, not a C++ binary."""
    try:
        with open(path, "rb") as fp:
            return fp.read(2) == b"#!"
    except OSError:
        return False


def validate_cpp(path):
    """Return an error string, or None if `path` is a usable v2.2 binary."""
    p = Path(path).expanduser()
    if not p.is_file():
        return f"not a file: {p}"
    if not (p.stat().st_mode & 0o111):
        return f"not executable: {p}"
    if looks_like_python_v3(p):
        return (f"{p} is a script (starts with '#!') -- this looks like the "
                "Hector v3 Python estimatetrend, not the C++ v2.2 binary.")
    v3 = shutil.which("estimatetrend")
    if v3 and p.resolve() == Path(v3).resolve():
        return (f"{p} is the same file as the Hector v3 'estimatetrend' on "
                "your PATH -- the comparison needs the separate C++ v2.2 "
                "binary (https://teromovigo.com/hector/).")
    return None


def validate_rscript(path):
    """Return an error string, or None if Rscript runs and gmwmx2 loads."""
    p = Path(path).expanduser()
    if not p.is_file():
        return f"not a file: {p}"
    ret = subprocess.run([str(p), "--vanilla", "-e", "library(gmwmx2)"],
                         capture_output=True, text=True)
    if ret.returncode != 0:
        return (f"{p} runs, but 'library(gmwmx2)' failed -- install the "
                "package first: install.packages(\"gmwmx2\")  (CRAN; "
                "https://github.com/SMAC-Group/gmwmx2).\n"
                f"    R said: {ret.stderr.strip().splitlines()[-1] if ret.stderr else '?'}")
    return None


def autodetect_cpp():
    cands = [shutil.which("estimatetrend_2.2"),
             "/usr/local/bin/estimatetrend_2.2"]
    for c in cands:
        if c and Path(c).is_file() and validate_cpp(c) is None:
            return str(Path(c).resolve())
    return None


def autodetect_rscript():
    cands = [shutil.which("Rscript"), "/usr/local/bin/Rscript",
             "/opt/homebrew/bin/Rscript"]
    for c in cands:
        if c and Path(c).is_file() and validate_rscript(c) is None:
            return str(Path(c).resolve())
    return None


def load_tools():
    """Resolution used by the other scripts: tools.json first, then PATH."""
    cfg = {}
    if TOOLS_JSON.exists():
        try:
            cfg = json.loads(TOOLS_JSON.read_text())
        except json.JSONDecodeError:
            print(f"warning: {TOOLS_JSON} is not valid JSON; ignoring it.")
    return {
        "estimatetrend_2.2": cfg.get("estimatetrend_2.2") or autodetect_cpp(),
        "rscript": cfg.get("rscript") or autodetect_rscript(),
    }


def ask(prompt):
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


def interactive_one(name, current, validate, hint):
    if current:
        err = validate(current)
        if err is None:
            print(f"  {name}: found automatically -> {current}")
            ans = ask("    keep this? [Y/n/other path] ")
            if ans.lower() in ("", "y", "yes"):
                return current
            if ans.lower() not in ("n", "no"):
                current = ans          # user typed a path directly
            else:
                current = None
        else:
            print(f"  {name}: candidate rejected -- {err}")
            current = None
    while True:
        if current is None:
            print(f"  {name}: not found.  {hint}")
            current = ask("    enter path (empty = skip, comparisons will "
                          "be disabled): ")
            if not current:
                return None
        err = validate(current)
        if err is None:
            return str(Path(current).expanduser().resolve())
        print(f"    rejected: {err}")
        current = None


def main():
    tools = load_tools()
    if "--status" in sys.argv:
        for k, v in tools.items():
            print(f"  {k:20s} {v or 'NOT FOUND (run setup_tools.py)'}")
        return

    print("Configuring the optional external tools for run_all.py")
    print("(both are needed for the Hector-v2.2 / gmwmx2 comparisons;")
    print(" everything else runs without them)\n")
    cpp = interactive_one(
        "Hector C++ v2.2 estimatetrend", tools["estimatetrend_2.2"],
        validate_cpp,
        "It is the C++ binary from https://teromovigo.com/hector/ (any "
        "file name is fine here).")
    rsc = interactive_one(
        "Rscript with gmwmx2", tools["rscript"], validate_rscript,
        "Install R, then in R: install.packages(\"gmwmx2\").")

    cfg = {}
    if cpp:
        cfg["estimatetrend_2.2"] = cpp
    if rsc:
        cfg["rscript"] = rsc
    TOOLS_JSON.write_text(json.dumps(cfg, indent=1) + "\n")
    print(f"\nsaved -> {TOOLS_JSON}")
    print("run_all.py --check will now use these paths.")


if __name__ == "__main__":
    main()
