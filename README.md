# Common Ground Is Convex: experiment package

## Run everything
1. Install the packages: `pip install -r requirements.txt` (Python 3.10 or later).
2. Run one of:
   - `python run_all.py --quick`: draws every figure and prints every number in the paper from the shipped `results/` folder. It downloads only the World Color Survey (WCS) files, about 2 MB. A few minutes.
   - `python run_all.py`: full reproduction from the raw data. It downloads both datasets, deletes the cached stages, refits every speaker, reruns every test, draws the figures, and prints the numbers. About 40 minutes on one CPU.
3. Results:
   - figures: `paper_figures/`, with the file names that `main.tex` includes;
   - numbers: `results/paper_numbers.md`, grouped by the section of the paper that reports them.

On Google Colab, open `SoCG_color.ipynb`, upload the zip of this package in the first cell, and run the cells in order.

No paths need editing. Every script finds its files relative to this folder. To use data stored elsewhere, set `CG_WCS_DIR` and `CG_XKCD_DB`.

## Files
- `run_all.py`: the single entry point.
- `get_data.py`: downloads the WCS raw files and the xkcd color survey, checks them against `results/input_checksums.txt`, and builds the xkcd SQLite database. Files already present are not downloaded again.
- `run_stages.py`: fits every speaker and runs the synthetic validation, H1 with both baselines, H2, H3, the exploratory H4, H5, and the xkcd fits. Writes the result tables to `results/`.
- `run_sens.py`: H2 with the looser eigenvalue floor $M \succeq 0.01 I$.
- `run_nsweep.py`: the follow-up predictions on sampling. `--summary-only` recomputes the summary from the saved tables.
- `run_extra.py`: weights fitted with the metric held fixed, the protected-region counts of the worked example, and the tilt and axis statistics.
- `make_figures.py`: every figure in the paper.
- `print_results.py`: every number in the paper.
- `config.py`: all paths. `cg_lib.py` and `analysis.py`: the library code. All prespecified constants are at the top of `cg_lib.py`.
- `results/`: the tables behind the paper as CSV and text files, and the fitted metrics as plain numpy arrays, which any version of numpy reads:
  - `fitted_metrics.npz`: for each of the 2,441 WCS speakers, the language, the speaker, the fitted metric with shared prototypes (`M_shared`, used by every test), the metric with private prototypes, the metric fitted jointly with weights, and the weights. A full run also adds the fitted metric of each xkcd session (`xkcd_M`, `xkcd_user_id`).
  - `fitted_metrics_eps001.npz`: the refit with $M \succeq 0.01 I$.
  - `worked_example_fits.npz`: the metrics of the worked example (Figure 1).
  - `h1_convexity.csv`: one row per term, with the scattered baseline and the connected baseline of H1. `h4_pooling.csv`: the exploratory H4.
- `results/cache/`: every stored stage of the analysis as pickle files, including the fitted metrics. These files need the pandas version that wrote them. The quick run does not read them, and the full run rebuilds them.

## Data
- WCS: `term.txt`, `foci-exp.txt`, `cnum-vhcm-lab-new.txt`, `dict.txt`, and `spkr-lsas.txt` from the WCS Data Archives (Cook, Kay, Regier), release WCS-Data-20110316, through the byte-identical mirror https://github.com/jvosten/wcs.
- xkcd: `colorsurvey.tar.gz` from https://xkcd.com/color/colorsurvey.tar.gz.
- Preprocessing, filtering rules, and the settings changed after data diagnostics are described in Appendix B of the paper and at the top of `cg_lib.py`.

## Environment and settings
- Tested with Python 3.12, cvxpy 1.9 (solver Clarabel, SCS as fallback), colour-science 0.4.7 (the package keeps its British name), numpy 2.4, scipy 1.17, and pandas 3.0.
- Fitter: coordinates divided by 30, trace 3, $M \succeq 0.05 I$. The margin in squared distance is 0.04 without weights and 0.02 with weights.
- Seeds: master seed 20260929; each stage has its own generator (synthetic +1000, H1 +2000, H2 +3000, H3 +4000, H4 +5000, H5 +6000, xkcd +7000, H1 connected baseline +8000, sensitivity +13000, sweep +21000).
- Sampling: the pattern samples are uniform in the bounding box of the 330 chips in CIELAB D65. For H3, each $n \in \{2,4,8,16,24\}$ uses 5 random speaker groups plus the whole language.

## What was fixed, and when
There is no external registration. The record is as follows.
- The five hypotheses H1 to H5 and the decision rules of H2 and H3 were written down before any fit of the data.
- The fitter, the lexicon rule, the speaker threshold, and the prototypes were then changed after diagnostic fits on a few languages, which looked at fit accuracy only. The changes are marked at the top of `cg_lib.py`.
- The test code was written next, before any test output. Its first H1 baseline draws scattered random chips. The plan had named random connected chip sets, so that baseline was added later, with the same statistic and the same rule. `results/h1_convexity.csv` holds both.
- The thresholds of H1 (80%) and H5 (70%) were written after a trial run on three languages. The paper therefore reports H5 without a pass or fail rule.
- H4 was marked exploratory in the plan. Most of its measurements are zero, and the paper does not count it as support.
- No statistic or rule changed after the full run. The follow-up predictions on sampling were added after the first H3 result and are labeled as such in the paper.
