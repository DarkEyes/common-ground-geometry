"""
print_results.py -- print every number reported in the paper, computed from the
tables in results/, grouped by the section of the paper that reports it.
The output is also written to results/paper_numbers.md.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

import config

R = config.RESULTS
CAP_005 = np.sqrt((3 - 2 * 0.05) / 0.05)   # cap on kappa-hat for M >= 0.05 I and trace 3
CAP_001 = np.sqrt((3 - 2 * 0.01) / 0.01)
lines = []


def out(s=""):
    lines.append(s)
    print(s)


def csv(name):
    return pd.read_csv(os.path.join(R, name))


def text(name):
    p = os.path.join(R, name)
    return open(p).read().strip() if os.path.exists(p) else "(missing: run the full analysis)"


out("# Numbers reported in the paper")

# ---------------------------------------------------------------- Section 5 and Appendix B
out("\n## Data (Section 5, Table 1, Appendix B)")
lt = csv("languages.csv"); k = lt[lt.kept == True]
out("WCS languages kept: %d of %d" % (len(k), len(lt)))
out("WCS speakers kept: %d of %d" % (k.n_speakers.sum(), lt.n_speakers_total.sum()))
out("terms per language: median %.0f, range %d-%d" % (k.n_terms.median(), k.n_terms.min(), k.n_terms.max()))
out("kept speakers per language: median %.0f" % k.n_speakers.median())
out("labeled chips per kept speaker: mean share %.2f" % k.labeled_frac.mean())
xc = csv("xkcd_counts.csv").iloc[0]
out("xkcd sessions: %d eligible, %d fitted" % (xc.n_eligible, xc.n_fitted))
for r in csv("color_checks.csv").itertuples():
    out("color check, %s: mean dE %.2f, max %.2f (n = %d)" % (r.check, r.mean_dE, r.max_dE, r.n))

# ---------------------------------------------------------------- Section 4 and Appendix B
out("\n## Synthetic validation (Section 4, Appendix B, Table 4)")
syn = csv("synthetic.csv")
tab = syn[syn.kappa == 2.0].pivot_table(index="noise", columns="T", values="frob_err", aggfunc="mean")
out("relative error, kappa = 2, rows = flip rate nu, columns = T:")
out(tab.round(3).to_string())
s15 = syn[syn.noise == 0.15]
out("at nu = 0.15, mean of (kappa-hat minus kappa): %.2f" % (s15.kappa_hat - s15.kappa).mean())

# ---------------------------------------------------------------- Section 6
out("\n## Fit quality (Section 6, Table 2)")
fits = csv("fits_per_speaker.csv"); wo = csv("weights_only.csv")
out("Euclidean, no weights:           %.3f" % fits.acc_euclid.mean())
out("fitted metric, no weights:       %.3f" % fits.acc_shared.mean())
out("fitted metric, then weights:     %.3f" % wo.acc_weights_only.mean())
out("metric and weights fitted jointly: %.3f" % fits.acc_weighted.mean())
out("private prototypes:              %.3f" % fits.acc_private.mean())
out("focal-chip prototypes:           %.3f" % fits.acc_focal.mean())
out("fitted kappa-hat: median %.2f, share at the cap %.3f" % (fits.kappa_shared.median(), (fits.kappa_shared > 0.98 * CAP_005).mean()))
ta = csv("tilts_and_axes.csv").iloc[0] if os.path.exists(os.path.join(R, "tilts_and_axes.csv")) else None
if ta is not None:
    out("most compressed axis: L %.2f, a %.2f, b %.2f" % (ta.compressed_axis_share_L, ta.compressed_axis_share_a, ta.compressed_axis_share_b))
xf = csv("xkcd_fits.csv")
out("xkcd: accuracy Euclidean %.2f, fitted %.2f; kappa-hat median %.1f; share at the cap %.3f"
    % (xf.acc_euclid.mean(), xf.acc_shared.mean(), xf.kappa.median(), (xf.kappa > 0.98 * CAP_005).mean()))

out("\n## Worked example (Figure 1)")
out(text("worked_example.txt"))
out(text("protected_region_lang87.txt"))

# ---------------------------------------------------------------- H1
out("\n## H1 (Section 6)")
h1 = csv("h1_convexity.csv"); t = h1[h1.testable == True]
out("terms with at least 3 common-ground chips: %d; testable: %d" % ((h1.n_common >= 3).sum(), len(t)))
out("not testable: %s" % h1[(h1.n_common >= 3) & (h1.testable != True)].reason.value_counts().to_dict())
out("scattered baseline: hull-fill above the baseline median in %d of %d" % (t.beats_baseline.astype(bool).sum(), len(t)))
out("median hull-fill %.2f, median scattered baseline %.2f, share with p < 0.05: %.2f" % (t.hull_fill.median(), t.baseline_median.median(), (t.p_perm < 0.05).mean()))
c = t[t.connected_testable == True]
out("connected baseline: testable %d of %d; hull-fill above the baseline median in %d, equal in %d, below in %d"
    % (len(c), len(t), (c.hull_fill > c.connected_median).sum(), (c.hull_fill == c.connected_median).sum(), (c.hull_fill < c.connected_median).sum()))
out("median hull-fill %.2f, median connected baseline %.2f, share with p < 0.05: %.2f" % (c.hull_fill.median(), c.connected_median.median(), (c.p_perm_connected < 0.05).mean()))
out("decision rule (at least 80%% of testable terms above the baseline median): scattered %s, connected %s"
    % ("met" if t.beats_baseline.astype(bool).mean() >= 0.8 else "not met", "met" if (c.hull_fill > c.connected_median).mean() >= 0.8 else "not met"))
out("model vs empirical common ground, median Jaccard: %.2f" % h1[h1.n_common > 0].jaccard_model_vs_empirical.median())

# ---------------------------------------------------------------- H2
out("\n## H2 (Section 6)")
h2 = csv("h2_core.csv")
r1, p1 = spearmanr(h2.kappa_max, h2.relative_core_median)
r3, p3 = spearmanr(h2.fixedn_kappa_max, h2.fixedn_relative_core)
out("languages: %d" % len(h2))
out("r_S(kappa-hat, relative core) = %.2f, p = %.3f" % (r1, p1))
out("five speakers: r_S = %.2f, p = %.3f" % (r3, p3))
out("smallest margin of the core bound: %.2f CIELAB units" % h2.min_radius_minus_bound.min())
e = os.path.join(R, "h2_core_eps001.csv")
if os.path.exists(e):
    h2b = pd.read_csv(e)
    rb, pb = spearmanr(h2b.kappa_max, h2b.relative_core_median)
    rc, pc = spearmanr(h2b.fixedn_kappa_max, h2b.fixedn_relative_core)
    out("refit with M >= 0.01 I: r_S = %.2f (p = %.1e); five speakers: r_S = %.2f (p = %.1e)" % (rb, pb, rc, pc))
f2 = os.path.join(R, "fits_eps001.csv")
if os.path.exists(f2):
    out("refit with M >= 0.01 I: cap %.1f, share at the cap %.3f" % (CAP_001, (pd.read_csv(f2).kappa_shared > 0.98 * CAP_001).mean()))

# ---------------------------------------------------------------- H3
out("\n## H3 (Section 6, Figure 2 right)")
sl = csv("h3_slopes.csv")
out("languages: %d; slope median %.2f, IQR %.2f-%.2f, max %.2f" % (len(sl), sl.slope.median(), sl.slope.quantile(.25), sl.slope.quantile(.75), sl.slope.max()))
xp = csv("xkcd_patterns.csv").groupby("n").patterns.mean()
out("xkcd: %.0f patterns at n = %d, %.0f at n = %d" % (xp.iloc[0], xp.index[0], xp.iloc[-1], xp.index[-1]))

# ---------------------------------------------------------------- follow-up predictions
out("\n## Follow-up predictions on sampling (Section 6, Figure 3)")
df = csv("h3_nsweep.csv"); dv = csv("disagreement_volume.csv")
out("languages: %d" % df.lang.nunique())
for N, g in df.groupby("N"):
    s = [np.polyfit(np.log(m.index.values), np.log(m.values), 1)[0] for _, m in ((l, gg.groupby("n").patterns.mean()) for l, gg in g.groupby("lang"))]
    out("N = %d: exponent median %.2f, IQR %.2f-%.2f" % (N, np.median(s), np.percentile(s, 25), np.percentile(s, 75)))
sizes = [2, 4, 8, 16, 24]
g = df[df.N == df.N.max()].groupby(["lang", "n"]).patterns.mean().unstack()
loc = np.log(g[sizes[1:]].values / g[sizes[:-1]].values) / np.log(np.array(sizes[1:]) / np.array(sizes[:-1]))
out("N = %d, local exponents between n = 2-4, 4-8, 8-16, 16-24: %s" % (df.N.max(), np.round(np.nanmedian(loc, axis=0), 1)))
out("share of the box in D_1 by n (median): %s" % dv.groupby("n").nonunanimous.median().round(2).to_dict())
mterms = dict(zip(lt.lang, lt.n_terms))
mm = df.merge(dv.groupby(["lang", "n"]).nonunanimous.mean().reset_index(), on=["lang", "n"])
mm["ratio"] = mm.patterns / (mm.lang.map(mterms) + mm["N"] * mm.nonunanimous)
out("observed count / (m + N * share of D_1): max %.2f; median at n = 24: %s" % (mm.ratio.max(), mm[mm.n == 24].groupby("N").ratio.median().round(2).to_dict()))

# ---------------------------------------------------------------- tilts (Appendix B)
if ta is not None:
    out("\n## Tilts of the fitted walls (Appendix B)")
    out("tilt from the base diagram: mean %.1f deg, max %.1f deg" % (ta.tilt_vs_base_mean, ta.tilt_vs_base_max))
    out("tilt between two speakers, over all speaker pairs of all languages: mean %.1f deg" % ta.tilt_between_speakers_mean)

# ---------------------------------------------------------------- H4 (exploratory)
out("\n## H4, exploratory (Appendix C)")
h4 = csv("h4_pooling.csv"); g4 = h4.groupby("pool_size").core_radius
out("mean core radius by number of pooled languages: %s" % g4.mean().round(1).to_dict())
out("share of measurements with radius zero by number of pooled languages: %s" % g4.apply(lambda x: (x == 0).mean()).round(2).to_dict())
r4, p4 = spearmanr(h4.pool_size, h4.core_radius)
out("r_S(number of pooled languages, core radius) = %.2f, p = %.1e, measurements: %d" % (r4, p4, len(h4)))

# ---------------------------------------------------------------- H5
out("\n## H5 (Appendix C)")
h5 = csv("h5_tilt.csv"); v = h5.dropna(subset=["rho"])
out("languages: %d; r_S median %.3f, IQR %.2f-%.2f" % (len(v), v.rho.median(), v.rho.quantile(.25), v.rho.quantile(.75)))
out("positive in %d of %d languages; positive with p < 0.05 in %d; positive with p < 0.01 in %d; Wilcoxon p = %.1e"
    % ((v.rho > 0).sum(), len(v), ((v.rho > 0) & (v.p < 0.05)).sum(), ((v.rho > 0) & (v.p < 0.01)).sum(), wilcoxon(v.rho).pvalue))
out("mean over languages of the mean tilt between sampled speaker pairs: %.0f deg" % v.mean_tilt_deg.mean())

open(os.path.join(R, "paper_numbers.md"), "w").write("\n".join(lines) + "\n")
print("\nwritten to results/paper_numbers.md")
