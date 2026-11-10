"""
run_all.py -- one command for everything.

    python run_all.py --quick   # figures and paper numbers from the shipped results (a few minutes)
    python run_all.py           # full reproduction from the raw data (about 40 minutes on one CPU)

The full run downloads both datasets, deletes the cached stages, recomputes every
fit and test, draws the figures, and prints every number in the paper.
"""
import os
import shutil
import subprocess
import sys
import time

import config


def step(script, *args):
    t0 = time.time()
    print("\n=== %s %s" % (script, " ".join(args)), flush=True)
    subprocess.run([sys.executable, os.path.join(config.PKG, script), *args], check=True, cwd=config.PKG)
    print("=== %s done in %.0f s" % (script, time.time() - t0), flush=True)


if __name__ == "__main__":
    t0 = time.time()
    if "--quick" in sys.argv:
        step("get_data.py", "--wcs")
        step("make_figures.py")
        step("print_results.py")
    else:
        step("get_data.py")
        shutil.rmtree(os.path.join(config.RESULTS, "cache"), ignore_errors=True)
        os.makedirs(os.path.join(config.RESULTS, "cache"), exist_ok=True)
        step("run_stages.py")
        step("run_sens.py")
        step("run_nsweep.py")
        step("run_extra.py")
        step("make_figures.py")
        step("print_results.py")
    print("\nall done in %.0f minutes; figures in paper_figures/, numbers in results/paper_numbers.md" % ((time.time() - t0) / 60))
