"""
run_stages.py -- the main analysis: fit every speaker, run the synthetic
validation, H1 with both baselines, H2, H3, the exploratory H4, H5, and the
xkcd fits, and write the result tables to results/. Each stage is cached in
results/cache/.
"""
import time
import numpy as np
import pandas as pd

import config
import cg_lib as cg
import analysis as an


def P(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


w = cg.load_wcs(config.WCS_DIR); P("WCS loaded")
langs, ltab = an.cached("langs", lambda: an.build_all(w)); P("languages: %d" % len(langs))
fits = an.cached("fits", lambda: an.fit_all(w, langs)); P("fits: %d speakers" % len(fits))
# the worked example (Figure 1) as plain arrays, readable with any pandas version
WORKED_LANG = 87
L = langs[WORKED_LANG]; sub = fits[fits.lang == WORKED_LANG].sort_values("k")
np.savez(config.RESULTS + "/worked_example_fits.npz", lang=WORKED_LANG, prototypes=L.prototypes, labels=L.labels,
         M=np.stack(list(sub.M_shared)), acc=sub.acc_shared.values.astype(float))
syn = an.cached("synthetic", lambda: an.synthetic(w, langs, an.stage_rng(1), trials=20)); P("synthetic validation")
h1 = an.cached("h1", lambda: an.h1_convexity(w, langs, fits, an.stage_rng(2))); P("H1")
h1c = an.cached("h1_connected", lambda: an.h1_connected(w, langs, h1, an.stage_rng(8))); P("H1, connected baseline")
h2 = an.cached("h2", lambda: an.h2_core(w, langs, fits, an.stage_rng(3))); P("H2")
h3, slopes = an.cached("h3", lambda: an.h3_patterns(w, langs, fits, an.stage_rng(4))); P("H3")
h4 = an.cached("h4", lambda: an.h4_pooling(w, langs, fits, an.stage_rng(5))); P("H4 (exploratory)")
h5 = an.cached("h5", lambda: an.h5_tilt(w, langs, fits, an.stage_rng(6))); P("H5")
xk_fits, xk_pat, xk_coll, xk_meta = an.cached("xkcd", lambda: an.xkcd_run(config.XKCD_DB, an.stage_rng(7))); P("xkcd")

# result tables
R = config.RESULTS
ltab.to_csv(R + "/languages.csv", index=False)
an.color_checks(w).to_csv(R + "/color_checks.csv", index=False)
fits.drop(columns=["M_shared", "M_private", "W", "M_weighted"]).to_csv(R + "/fits_per_speaker.csv", index=False)
an.fit_summary(fits).to_csv(R + "/fit_summary.csv", index=False)
syn.to_csv(R + "/synthetic.csv", index=False)
pd.concat([h1, h1c], axis=1).to_csv(R + "/h1_convexity.csv", index=False)   # both baselines, one row per term
h2.to_csv(R + "/h2_core.csv", index=False)
h3.to_csv(R + "/h3_patterns.csv", index=False)
slopes.to_csv(R + "/h3_slopes.csv", index=False)
h4.to_csv(R + "/h4_pooling.csv", index=False)
h5.to_csv(R + "/h5_tilt.csv", index=False)
xk_fits.drop(columns=["M"], errors="ignore").to_csv(R + "/xkcd_fits.csv", index=False)
xk_pat.to_csv(R + "/xkcd_patterns.csv", index=False)
xk_coll.to_csv(R + "/xkcd_collapse.csv", index=False)
pd.DataFrame(xk_meta["prototypes"], index=cg.BASIC_TERMS, columns=["L", "a", "b"]).round(1).to_csv(R + "/xkcd_prototypes.csv")
pd.DataFrame([dict(n_users_with_basic=xk_meta["n_users_with_basic"], n_eligible=xk_meta["n_eligible"], n_fitted=xk_meta["n_fitted"])]).to_csv(R + "/xkcd_counts.csv", index=False)
# every fitted metric as plain arrays, readable with any pandas version
f = fits.sort_values(["lang", "k"]).reset_index(drop=True)
mmax = max(len(x) for x in f.W)
W = np.full((len(f), mmax), np.nan)
for i, x in enumerate(f.W):
    W[i, :len(x)] = x
arrays = dict(lang=f.lang.values, k=f.k.values, speaker=f.spkr.values,
              M_shared=np.stack(list(f.M_shared)), M_private=np.stack(list(f.M_private)),
              M_weighted=np.stack(list(f.M_weighted)), weights=W)
if "M" in xk_fits.columns:
    arrays.update(xkcd_user_id=xk_fits.user_id.values, xkcd_M=np.stack(list(xk_fits.M)))
np.savez(R + "/fitted_metrics.npz", **arrays)
P("result tables and fitted metrics written to results/")
