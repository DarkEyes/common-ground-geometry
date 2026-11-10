"""
analysis.py -- the experiment stages (fits, synthetic validation, H1 with both
baselines, H2, H3, the exploratory H4, H5, xkcd). Each function returns pandas
tables. run_stages.py calls them.
"""
from __future__ import annotations
import os, time, itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, wilcoxon
from scipy.spatial import ConvexHull
import config
import cg_lib as cg

OUT = config.RESULTS
os.makedirs(OUT, exist_ok=True)
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
os.makedirs(os.path.join(OUT, "cache"), exist_ok=True)
import pickle


def cached(name, fn):
    """Run fn() once and pickle the result; delete results/cache/<name>.pkl to recompute."""
    p = os.path.join(OUT, "cache", name + ".pkl")
    if os.path.exists(p):
        try:
            with open(p, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            # a cache written by a different pandas or numpy version may not load; recompute it
            print("cache %s could not be read (%s); recomputing" % (name, type(e).__name__), flush=True)
    t0 = time.time()
    out = fn()
    with open(p, "wb") as f:
        pickle.dump(out, f)
    with open(os.path.join(OUT, "cache", "timings.txt"), "a") as f:
        f.write("%s\t%.1f s\n" % (name, time.time() - t0))
    return out


def stage_rng(stage: int) -> np.random.Generator:
    """Each stage gets its own seeded generator, so results do not depend on execution order."""
    return np.random.default_rng(cg.SEED + 1000 * stage)


def savefig(name):
    p = os.path.join(OUT, "figures", name)
    plt.tight_layout(); plt.savefig(p, dpi=150); plt.close()
    return p


# ---------------------------------------------------------------- color checks
def color_checks(w: cg.WCS) -> pd.DataFrame:
    errs = []
    for _, r in w.chips.iterrows():
        l = cg.munsell_to_lab_c(r.munH, r.munV, r.chroma)
        if l is not None:
            errs.append(np.linalg.norm(l - np.array([r.L_c, r.a_c, r.b_c])))
    shift = np.linalg.norm(w.lab - w.chips[["L_c", "a_c", "b_c"]].values, axis=1)
    return pd.DataFrame([
        dict(check="archive Illuminant-C values vs the colour-science package's Munsell renotation",
             n=len(errs), mean_dE=np.mean(errs), max_dE=np.max(errs)),
        dict(check="shift from Illuminant C to D65 (Bradford), all 330 chips",
             n=330, mean_dE=shift.mean(), max_dE=shift.max())])


# ---------------------------------------------------------------- build all languages
def build_all(w: cg.WCS):
    langs, rows = {}, []
    for lang in sorted(w.terms.lang.unique()):
        L = cg.build_language(w, lang)
        n_all = len(w.terms[w.terms.lang == lang].spkr.unique())
        if L is None:
            rows.append(dict(lang=lang, kept=False, n_terms=None, n_speakers=0, n_speakers_total=n_all))
            continue
        langs[lang] = L
        rows.append(dict(lang=lang, kept=True, n_terms=len(L.terms), n_speakers=len(L.speakers),
                         n_speakers_total=L.n_speakers_total,
                         labeled_frac=float((L.labels >= 0).mean())))
    return langs, pd.DataFrame(rows)


# ---------------------------------------------------------------- fits
def fit_all(w: cg.WCS, langs: dict) -> pd.DataFrame:
    rows = []
    t0 = time.time()
    for lang, L in langs.items():
        for k, s in enumerate(L.speakers):
            ok = L.labels[k] >= 0
            eu = float((cg.predict_labels(w.lab, np.eye(3), L.prototypes)[ok] == L.labels[k][ok]).mean())
            f_sh = cg.fit_metric(w.lab, L.labels[k], L.prototypes)
            f_w = cg.fit_metric(w.lab, L.labels[k], L.prototypes, weights=True)
            f_pr = cg.fit_metric(w.lab, L.labels[k], L.private_prototypes[k])
            f_fo = cg.fit_metric(w.lab, L.labels[k], L.prototypes_focal)
            rows.append(dict(lang=lang, spkr=s, k=k, n_terms=len(L.terms), n_labeled=int(ok.sum()),
                             acc_euclid=eu,
                             acc_shared=f_sh["accuracy"], kappa_shared=f_sh["kappa"], slack_shared=f_sh["slack"],
                             acc_weighted=f_w["accuracy"], kappa_weighted=f_w["kappa"],
                             acc_private=f_pr["accuracy"], kappa_private=f_pr["kappa"],
                             acc_focal=f_fo["accuracy"], kappa_focal=f_fo["kappa"],
                             M_shared=f_sh["M"], M_private=f_pr["M"], W=f_w["weights"], M_weighted=f_w["M"]))
    df = pd.DataFrame(rows)
    df.attrs["seconds"] = time.time() - t0
    return df


def fit_summary(fits: pd.DataFrame) -> pd.DataFrame:
    cap = np.sqrt((cg.FIT_TRACE - 2 * cg.FIT_EPS) / cg.FIT_EPS)
    out = []
    for name in ["euclid", "focal", "shared", "weighted", "private"]:
        acc = fits[f"acc_{name}"]
        row = dict(model=name, mean_accuracy=acc.mean(), median_accuracy=acc.median())
        if name != "euclid":
            k = fits[f"kappa_{name}"]
            row.update(median_kappa=k.median(), frac_kappa_at_cap=float((k > 0.98 * cap).mean()))
        out.append(row)
    return pd.DataFrame(out)


# ---------------------------------------------------------------- synthetic validation
def random_spd(kappa: float, rng: np.random.Generator) -> np.ndarray:
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    ev = np.array([kappa ** 2, 1.0, 1.0]) if kappa > 1 else np.ones(3)
    M = Q @ np.diag(ev) @ Q.T
    return M * cg.FIT_TRACE / np.trace(M)


def synthetic(w: cg.WCS, langs: dict, rng: np.random.Generator, trials: int = 20) -> pd.DataFrame:
    # prototypes from a real language with a mid-sized lexicon
    L = max(langs.values(), key=lambda L: (len(L.terms) if len(L.terms) <= 9 else 0))
    P = L.prototypes; m = len(P)
    lo, hi = w.lab.min(axis=0), w.lab.max(axis=0)
    rows = []
    for kappa in [1.0, 1.5, 2.0, 3.0]:
        for T in [50, 100, 330, 1000]:
            for p in [0.0, 0.05, 0.15]:
                for tr in range(trials):
                    M = random_spd(kappa, rng)
                    X = w.lab if T == 330 else rng.uniform(lo, hi, size=(T, 3))
                    diff = X[:, None, :] - P[None, :, :]
                    q = np.einsum("tmi,ij,tmj->tm", diff, M, diff)
                    lab = q.argmin(axis=1)
                    if p > 0:
                        flip = rng.random(T) < p
                        second = np.argsort(q, axis=1)[:, 1]
                        lab = np.where(flip, second, lab)
                    f = cg.fit_metric(X, lab, P)
                    err = np.linalg.norm(f["M"] - M) / np.linalg.norm(M)
                    rows.append(dict(kappa=kappa, T=T, noise=p, trial=tr, frob_err=err,
                                     kappa_hat=f["kappa"], kappa_err=abs(f["kappa"] - kappa),
                                     accuracy=f["accuracy"]))
    df = pd.DataFrame(rows)
    df.attrs["n_terms"] = m; df.attrs["lang"] = L.lang
    return df


# ---------------------------------------------------------------- H1 convexity
def h1_convexity(w: cg.WCS, langs: dict, fits: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    ch = w.chips.set_index("cnum")
    rows = []
    for lang, L in langs.items():
        n = len(L.speakers)
        Ms = [fits[(fits.lang == lang) & (fits.k == k)].M_shared.iloc[0] for k in range(n)]
        preds = np.stack([cg.predict_labels(w.lab, M, L.prototypes) for M in Ms])
        for i, term in enumerate(L.terms):
            Q = [c + 1 for c in range(330) if (L.labels[:, c] == i).all()]
            U = [c + 1 for c in range(330) if (L.labels[:, c] == i).any()]
            Qm = set(c + 1 for c in range(330) if (preds[:, c] == i).all())
            jacc = len(Qm & set(Q)) / max(1, len(Qm | set(Q)))
            rec = dict(lang=lang, term=term, n_common=len(Q), n_union=len(U),
                       n_model_common=len(Qm), jaccard_model_vs_empirical=jacc)
            if len(Q) < cg.H1_MIN_COMMON:
                rec.update(testable=False); rows.append(rec); continue
            # focal hue: hue of chromatic chip in Q nearest to the prototype
            chrom = [c for c in Q if ch.loc[c, "hue"] > 0]
            if len(chrom) < 3:
                rec.update(testable=False, reason="achromatic"); rows.append(rec); continue
            near = min(chrom, key=lambda c: np.linalg.norm(w.lab[c - 1] - L.prototypes[i]))
            fh = int(ch.loc[near, "hue"])
            obs = cg.hull_fill(w, Q, fh)
            if obs is None:
                rec.update(testable=False, reason="degenerate"); rows.append(rec); continue
            base = []
            for _ in range(cg.H1_PERMUTATIONS):
                draw = list(rng.choice(U, size=len(Q), replace=False))
                v = cg.hull_fill(w, draw, fh)
                if v is not None:
                    base.append(v)
            base = np.array(base) if base else np.array([np.nan])
            rec.update(testable=True, hull_fill=obs, baseline_median=np.nanmedian(base),
                       p_perm=float(np.nanmean(base >= obs)), beats_baseline=bool(obs > np.nanmedian(base)))
            rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- H1, second baseline: connected sets
def _fill(all_pts: np.ndarray, members) -> float | None:
    """Hull-fill of a chip set on the chart. Same definition as cg.hull_fill, but the chart
    coordinates of all 330 chips (all_pts) are passed in, so they are computed once per term."""
    idx = np.asarray(members, dtype=int) - 1
    pts = all_pts[idx]
    pts = pts[~np.isnan(pts[:, 0])]
    if len(pts) < 3:
        return None
    try:
        eq = ConvexHull(pts).equations
    except Exception:
        return None            # collinear / degenerate
    chromatic = ~np.isnan(all_pts[:, 0])
    inside = np.zeros(len(all_pts), dtype=bool)
    inside[chromatic] = np.all(all_pts[chromatic] @ eq[:, :2].T + eq[:, 2] <= 1e-9, axis=1)
    if not inside.any():
        return None
    member = np.zeros(len(all_pts), dtype=bool)
    member[idx] = True
    return float((inside & member).sum() / inside.sum())


def h1_connected(w: cg.WCS, langs: dict, h1: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Second baseline for H1: random connected chip sets (the baseline of the original plan).

    For each testable term, a set starts at a random chromatic chip of the term's region U (the
    chips that some speaker labels with the term) and grows by adding a random chip of U that is
    next to the set in a row or a column of the chart, until it has as many chromatic chips as
    the common ground Q. Neighbors are taken in the unwrapped chart coordinates that the hull
    uses. The statistic, the number of draws, the p-value, and the decision rule are those of the
    first baseline. A term is not testable against this baseline if no connected part of U has
    as many chromatic chips as Q. Returns one row per row of h1, in the same order."""
    hue = w.chips.set_index("cnum")["hue"].to_dict()
    rows = []
    for r in h1.itertuples(index=False):
        rec = dict(n_chromatic=np.nan, connected_testable=False, connected_median=np.nan,
                   p_perm_connected=np.nan, beats_connected=np.nan, connected_draws=0)
        if not bool(r.testable):
            rows.append(rec); continue
        L = langs[r.lang]; i = L.terms.index(r.term)
        Q = [c + 1 for c in range(330) if (L.labels[:, c] == i).all()]
        U = [c + 1 for c in range(330) if (L.labels[:, c] == i).any()]
        chrom = [c for c in Q if hue[c] > 0]
        near = min(chrom, key=lambda c: np.linalg.norm(w.lab[c - 1] - L.prototypes[i]))
        all_pts = cg.chart_coords(w, list(range(1, 331)), int(hue[near]))
        obs = _fill(all_pts, Q)
        assert obs is not None and abs(obs - r.hull_fill) < 1e-9, "the H1 table and the labels disagree"
        q = len(chrom); rec["n_chromatic"] = q
        Uc = [c for c in U if hue[c] > 0]
        pos = {(int(all_pts[c - 1, 0]), int(all_pts[c - 1, 1])): c for c in Uc}
        nbrs = {c: [pos[p] for p in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)) if p in pos]
                for (x, y), c in pos.items()}
        size = {}                                  # chip -> size of its connected part of U
        for c in Uc:
            if c in size:
                continue
            part = [c]; seen = {c}; k = 0
            while k < len(part):
                for v in nbrs[part[k]]:
                    if v not in seen:
                        seen.add(v); part.append(v)
                k += 1
            for u in part:
                size[u] = len(part)
        seeds = [c for c in Uc if size[c] >= q]    # a set of q chips fits only in a part with q chips
        if not seeds:
            rows.append(rec); continue
        vals = []
        for _ in range(cg.H1_PERMUTATIONS):
            s0 = seeds[int(rng.integers(len(seeds)))]
            S = [s0]; in_S = {s0}; frontier = list(nbrs[s0]); in_F = set(frontier)
            while len(S) < q:
                j = int(rng.integers(len(frontier)))
                v = frontier[j]; frontier[j] = frontier[-1]; frontier.pop(); in_F.discard(v)
                S.append(v); in_S.add(v)
                for u in nbrs[v]:
                    if u not in in_S and u not in in_F:
                        frontier.append(u); in_F.add(u)
            f = _fill(all_pts, S)
            if f is not None:
                vals.append(f)
        if not vals:
            rows.append(rec); continue
        vals = np.array(vals); med = float(np.median(vals))
        rec.update(connected_testable=True, connected_median=med,
                   p_perm_connected=float((vals >= obs).mean()),
                   beats_connected=bool(obs > med), connected_draws=len(vals))
        rows.append(rec)
    return pd.DataFrame(rows, index=h1.index)


# ---------------------------------------------------------------- H2 core vs diversity
def h2_core(w: cg.WCS, langs: dict, fits: pd.DataFrame, rng: np.random.Generator,
            min_speakers: int = 5, fixed_n: int = 5, reps: int = 20) -> pd.DataFrame:
    rows = []
    for lang, L in langs.items():
        n = len(L.speakers)
        if n < min_speakers:
            continue
        sub = fits[fits.lang == lang].sort_values("k")
        Ms = list(sub.M_shared); ks = sub.kappa_shared.values
        P = L.prototypes; m = len(P)
        def relcore(Msel):
            vals = []
            for i in range(m):
                r = cg.core_radius(P, i, Msel)
                dmin = min(np.linalg.norm(P[i] - P[j]) for j in range(m) if j != i)
                vals.append((r, r / dmin))
            return np.array(vals)
        full = relcore(Ms)
        kmax, kmed = float(ks.max()), float(np.median(ks))
        bound = np.array([cg.theory_bound(P, i, kmax) for i in range(m)])
        # fixed-n subsamples
        sub_rel, sub_kmax = [], []
        for _ in range(reps):
            idx = rng.choice(n, size=fixed_n, replace=False)
            rc = relcore([Ms[i] for i in idx])
            sub_rel.append(np.median(rc[:, 1])); sub_kmax.append(ks[idx].max())
        rows.append(dict(lang=lang, n_speakers=n, n_terms=m, kappa_max=kmax, kappa_median=kmed,
                         core_radius_median=float(np.median(full[:, 0])),
                         relative_core_median=float(np.median(full[:, 1])),
                         min_radius_minus_bound=float((full[:, 0] - bound).min()),
                         fixedn_relative_core=float(np.mean(sub_rel)), fixedn_kappa_max=float(np.mean(sub_kmax))))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- H3 patterns
def h3_patterns(w: cg.WCS, langs: dict, fits: pd.DataFrame, rng: np.random.Generator,
                min_speakers: int = 8, reps: int = 5) -> pd.DataFrame:
    G = cg.random_grid(w.lab, cg.GRID_POINTS, rng)
    rows = []
    for lang, L in langs.items():
        n = len(L.speakers)
        if n < min_speakers:
            continue
        sub = fits[fits.lang == lang].sort_values("k")
        Ms = list(sub.M_shared)
        sizes = [s for s in [2, 4, 8, 16, 24] if s <= n]
        if n not in sizes:
            sizes.append(n)
        for s in sizes:
            for _ in range(reps if s < n else 1):
                idx = rng.choice(n, size=s, replace=False)
                rows.append(dict(lang=lang, n_terms=len(L.terms), n=s,
                                 patterns=cg.count_patterns(G, [Ms[i] for i in idx], L.prototypes)))
    df = pd.DataFrame(rows)
    slopes = []
    for lang, g in df.groupby("lang"):
        gg = g.groupby("n").patterns.mean()
        if len(gg) >= 3:
            sl = np.polyfit(np.log(gg.index.values), np.log(gg.values), 1)[0]
            slopes.append(dict(lang=lang, n_terms=g.n_terms.iloc[0], slope=sl,
                               patterns_at_max_n=gg.values[-1], max_n=gg.index.values[-1]))
    return df, pd.DataFrame(slopes)


# ---------------------------------------------------------------- H4 pooling (exploratory)
ANCHORS = {"white": "A0", "black": "J0", "red": "G1", "yellow": "C9", "green": "F17", "blue": "F29"}


def h4_pooling(w: cg.WCS, langs: dict, fits: pd.DataFrame, rng: np.random.Generator,
               pool_sizes=(1, 2, 4, 8, 16), n_total: int = 16, reps: int = 10,
               match_dE: float = 25.0) -> pd.DataFrame:
    grid_to_cnum = dict(zip(w.chips.grid, w.chips.cnum))
    anchor_lab = {c: w.lab[grid_to_cnum[g] - 1] for c, g in ANCHORS.items()}
    cats = list(ANCHORS)
    # per language: map each term to the nearest anchor within match_dE
    lang_map = {}
    for lang, L in langs.items():
        mp = {}
        for i in range(len(L.terms)):
            d = {c: np.linalg.norm(L.prototypes[i] - a) for c, a in anchor_lab.items()}
            c = min(d, key=d.get)
            if d[c] <= match_dE and c not in mp.values():
                mp[i] = c
        lang_map[lang] = mp
    dirs = rng.normal(size=(100, 3)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    radii = np.arange(1, 41, 1.0)
    eligible = [l for l in langs if len(langs[l].speakers) >= 4]
    rows = []
    for r in pool_sizes:
        for rep in range(reps):
            chosen = rng.choice(eligible, size=r, replace=False)
            # speakers pooled, held at n_total in total
            pool = []
            for lang in chosen:
                L = langs[lang]; sub = fits[fits.lang == lang].sort_values("k")
                for k in range(len(L.speakers)):
                    pool.append((lang, k, sub.M_shared.iloc[k], sub.kappa_shared.iloc[k]))
            idx = rng.choice(len(pool), size=min(n_total, len(pool)), replace=False)
            pool = [pool[i] for i in idx]
            kmax = max(p[3] for p in pool)
            for c in cats:
                rad = 0.0
                ok_all = True
                for lang, k, M, _ in pool:
                    mp = lang_map[lang]
                    tid = [i for i, cc in mp.items() if cc == c]
                    if not tid:
                        ok_all = False; break
                    tid = tid[0]
                    if cg.predict_labels(anchor_lab[c][None, :], M, langs[lang].prototypes)[0] != tid:
                        ok_all = False; break
                if not ok_all:
                    rows.append(dict(pool_size=r, rep=rep, category=c, n_pooled=len(pool),
                                     kappa_max=kmax, core_radius=0.0, all_have_term=False)); continue
                for rr in radii:
                    pts = anchor_lab[c][None, :] + rr * dirs
                    good = True
                    for lang, k, M, _ in pool:
                        tid = [i for i, cc in lang_map[lang].items() if cc == c][0]
                        if (cg.predict_labels(pts, M, langs[lang].prototypes) != tid).any():
                            good = False; break
                    if not good:
                        break
                    rad = rr
                rows.append(dict(pool_size=r, rep=rep, category=c, n_pooled=len(pool),
                                 kappa_max=kmax, core_radius=rad, all_have_term=True))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- H5 tilt
def h5_tilt(w: cg.WCS, langs: dict, fits: pd.DataFrame, rng: np.random.Generator,
            min_speakers: int = 5, max_pairs: int = 60) -> pd.DataFrame:
    rows = []
    for lang, L in langs.items():
        n = len(L.speakers)
        if n < min_speakers:
            continue
        sub = fits[fits.lang == lang].sort_values("k"); Ms = list(sub.M_shared)
        P = L.prototypes; m = len(P)
        pairs = list(itertools.combinations(range(n), 2))
        if len(pairs) > max_pairs:
            pairs = [pairs[i] for i in rng.choice(len(pairs), size=max_pairs, replace=False)]
        recs = []
        for a, b in pairs:
            la, lb = L.labels[a], L.labels[b]
            for i in range(m):
                for j in range(i + 1, m):
                    d = P[i] - P[j]
                    t = cg.tilt(Ms[a], Ms[b], d)
                    cnt = int(((la == i) & (lb == j)).sum() + ((la == j) & (lb == i)).sum())
                    recs.append((t, cnt))
        recs = np.array(recs)
        if len(recs) > 5 and recs[:, 1].std() > 0 and recs[:, 0].std() > 0:
            rho, p = spearmanr(recs[:, 0], recs[:, 1])
        else:
            rho, p = np.nan, np.nan
        rows.append(dict(lang=lang, n_speakers=n, n_terms=m, n_records=len(recs), rho=rho, p=p,
                         mean_tilt_deg=recs[:, 0].mean(), mean_disagreements=recs[:, 1].mean()))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- xkcd
def xkcd_run(db_path: str, rng: np.random.Generator):
    sub, protos, n_eligible, n_users_basic = cg.load_xkcd_basic(db_path, rng)
    fits = []
    Ms = []
    for uid, g in sub.groupby("user_id"):
        X = g[["L", "a", "b_"]].values; lab = g.tid.values.astype(int)
        f = cg.fit_metric(X, lab, protos)
        if f is None:
            continue
        ok = lab >= 0
        eu = float((cg.predict_labels(X, np.eye(3), protos) == lab).mean())
        fits.append(dict(user_id=uid, n_answers=len(g), acc_euclid=eu, acc_shared=f["accuracy"],
                         kappa=f["kappa"], slack=f["slack"], M=f["M"]))
        Ms.append(f["M"])
    fits = pd.DataFrame(fits)
    # grid: random sRGB colors inside the gamut
    rgb = rng.integers(0, 256, size=(cg.GRID_POINTS, 3))
    G = cg.srgb_to_lab_d65(rgb)
    pat = []
    for n in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        if n > len(Ms):
            break
        for rep in range(3 if n < len(Ms) else 1):
            idx = rng.choice(len(Ms), size=n, replace=False)
            pat.append(dict(n=n, patterns=cg.count_patterns(G, [Ms[i] for i in idx], protos)))
    pat = pd.DataFrame(pat)
    # collapse curve: fraction of grid points every one of n users labels with term i
    coll = []
    preds = np.stack([cg.predict_labels(G, M, protos) for M in Ms])
    for n in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        if n > len(Ms):
            break
        for rep in range(3):
            idx = rng.choice(len(Ms), size=n, replace=False)
            pr = preds[idx]
            for i, t in enumerate(cg.BASIC_TERMS):
                coll.append(dict(n=n, term=t, common_frac=float((pr == i).all(axis=0).mean()),
                                 kappa_max=float(fits.kappa.values[idx].max())))
    coll = pd.DataFrame(coll)
    meta = dict(n_users_with_basic=n_users_basic, n_eligible=n_eligible, n_fitted=len(fits),
                prototypes=protos)
    return fits, pat, coll, meta
