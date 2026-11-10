"""
run_sens.py -- sensitivity of H2 to the eigenvalue floor: refit every speaker
with M >= 0.01 I instead of 0.05 I and repeat the H2 statistics.
"""
import numpy as np, pandas as pd
import config
import cg_lib as cg, analysis as an
from scipy.stats import spearmanr
w = cg.load_wcs(config.WCS_DIR)
langs, ltab = an.cached("langs", lambda: an.build_all(w))
def fit_eps(eps):
    rows=[]
    for lang, L in langs.items():
        for k, s in enumerate(L.speakers):
            f = cg.fit_metric(w.lab, L.labels[k], L.prototypes, eps=eps)
            rows.append(dict(lang=lang, spkr=s, k=k, acc_shared=f["accuracy"], kappa_shared=f["kappa"], M_shared=f["M"]))
    return pd.DataFrame(rows)
fits2 = an.cached("fits_eps001", lambda: fit_eps(0.01))
h2b = an.cached("h2_eps001", lambda: an.h2_core(w, langs, fits2, an.stage_rng(13)))
cap = np.sqrt((3-0.02)/0.01)
out = ["# Sensitivity: M >= 0.01 I (kappa cap %.1f instead of 7.6)" % cap,
       "median kappa %.2f; share at cap %.3f; mean accuracy %.3f" % (fits2.kappa_shared.median(), (fits2.kappa_shared>0.98*cap).mean(), fits2.acc_shared.mean())]
for a,b,lab in [("kappa_max","relative_core_median","kappa_max"),("kappa_median","relative_core_median","kappa_median"),("fixedn_kappa_max","fixedn_relative_core","n fixed at 5")]:
    r,p = spearmanr(h2b[a], h2b[b]); out.append("H2 Spearman(%s, relative core): rho=%.3f p=%.2e" % (lab, r, p))
open("results/sensitivity_eps.md","w").write("\n".join(out)); print("\n".join(out))
h2b.to_csv("results/h2_core_eps001.csv", index=False)
fits2.drop(columns=["M_shared"]).to_csv("results/fits_eps001.csv", index=False)
f2 = fits2.sort_values(["lang", "k"]).reset_index(drop=True)
np.savez("results/fitted_metrics_eps001.npz", lang=f2.lang.values, k=f2.k.values, speaker=f2.spkr.values,
         M_shared=np.stack(list(f2.M_shared)))
