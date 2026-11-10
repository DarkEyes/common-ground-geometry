"""
cg_lib.py -- Common Ground Is Convex: experiment library.

Everything the notebook needs: color conversion to CIELAB (D65),
World Color Survey (WCS) and xkcd loading, the semidefinite metric fitter,
common-ground / pattern geometry, and the five hypothesis tests.

All thresholds are module-level constants so the values are visible in one
place. They were fixed before any hypothesis test. There is no external
registration.
"""
from __future__ import annotations

import os
import re
import sqlite3
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import cvxpy as cp
import colour  # third-party package name (colour-science); spelling is the package's, not ours
from scipy.spatial import ConvexHull
from scipy.stats import spearmanr

warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------------------------------------------------------
# Constants (fixed before any hypothesis test)
# ----------------------------------------------------------------------------
TERM_MIN_SPEAKER_FRAC = 0.5      # term kept if >= this fraction of speakers use it
TERM_MIN_MODAL_CHIPS = 5         # ... and its modal map covers >= this many chips
SPEAKER_MIN_LABELED = 200       # speaker kept if >= this many of 330 chips carry a kept term
# (set before any hypothesis test: 300 dropped most speakers of languages with rich non-basic vocabularies;
#  unlabeled chips are simply skipped by the fitter, so 200 keeps speakers without biasing labels)
FIT_EPS = 0.05                   # M >= eps * I
FIT_TRACE = 3.0                  # trace(M) fixed (d = 3)
FIT_MARGIN = 0.02                # fixed margin (scaled units); objective minimizes total violation
# (set before any hypothesis test: the first version of the fitter maximized the margin, which gave
#  negative margins on real labels. Without weights, the margin applies to (x - m_ij)^T M d_ij, which is
#  0.04 in squared distance. With weights, it applies to the squared-distance difference, which is 0.02.)
LAB_SCALE = 30.0                 # coordinates divided by this before fitting (scale-free anyway)
GRID_POINTS = 20000              # random CIELAB points for pattern counting
H1_MIN_COMMON = 3                # minimum common-ground chips to test convexity
H1_PERMUTATIONS = 200
XKCD_MIN_BASIC = 20              # xkcd session kept if >= this many basic-term answers
XKCD_MAX_SPAMPROB = 0.5
XKCD_SAMPLE_USERS = 1500
BASIC_TERMS = ["black", "white", "red", "green", "yellow", "blue",
               "brown", "orange", "pink", "purple", "gray"]
SEED = 20260929

XY_C = np.array([0.31006, 0.31616])     # CIE illuminant C, 2 deg
XY_D65 = np.array([0.31270, 0.32900])   # CIE illuminant D65, 2 deg


# ----------------------------------------------------------------------------
# Color conversion
# ----------------------------------------------------------------------------
def lab_c_to_lab_d65(lab_c: np.ndarray) -> np.ndarray:
    """CIELAB under illuminant C -> XYZ -> Bradford to D65 -> CIELAB under D65."""
    xyz_c = colour.Lab_to_XYZ(lab_c, illuminant=XY_C)
    xyz_w_c = colour.xy_to_XYZ(XY_C)
    xyz_w_d65 = colour.xy_to_XYZ(XY_D65)
    xyz_d65 = colour.adaptation.chromatic_adaptation_VonKries(
        xyz_c, xyz_w_c, xyz_w_d65, transform="Bradford")
    return colour.XYZ_to_Lab(xyz_d65, illuminant=XY_D65)


def srgb_to_lab_d65(rgb255: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb255, dtype=float) / 255.0
    xyz = colour.sRGB_to_XYZ(rgb)
    return colour.XYZ_to_Lab(xyz, illuminant=XY_D65)


def munsell_to_lab_c(mun_h: str, mun_v: float, chroma: float) -> np.ndarray | None:
    """Independent check of the archive's Illuminant-C values via the colour-science package."""
    try:
        spec = f"N {mun_v}/" if chroma == 0 else f"{mun_h} {mun_v}/{chroma}"
        xyY = colour.munsell_colour_to_xyY(spec)
        xyz = colour.xyY_to_XYZ(xyY)
        # color's Munsell data is specified under illuminant C
        return colour.XYZ_to_Lab(xyz, illuminant=XY_C)
    except Exception:
        return None


# ----------------------------------------------------------------------------
# WCS loading
# ----------------------------------------------------------------------------
def _read_cr(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        txt = f.read()
    return [ln for ln in txt.replace("\r\n", "\n").replace("\r", "\n").split("\n") if ln.strip()]


@dataclass
class WCS:
    chips: pd.DataFrame            # cnum, grid, row(V letter), hue(int), L,a,b (D65), L_c,a_c,b_c
    terms: pd.DataFrame            # lang, spkr, cnum, term
    foci: pd.DataFrame             # lang, spkr, term, cnum
    speakers: pd.DataFrame         # lang, spkr, age, sex
    lab: np.ndarray = field(default=None)          # 330 x 3, indexed by cnum-1
    grid_of: dict = field(default_factory=dict)    # cnum -> grid code


def load_wcs(raw_dir: str) -> WCS:
    # chip table with Illuminant C Lab
    rows = []
    for ln in _read_cr(os.path.join(raw_dir, "cnum-vhcm-lab-new.txt")):
        if ln.startswith("#"):
            continue
        p = ln.split("\t")
        rows.append(dict(cnum=int(p[0]), row=p[1], hue=int(p[2]), chroma=float(p[3]),
                         munH=p[4], munV=float(p[5]),
                         L_c=float(p[6]), a_c=float(p[7]), b_c=float(p[8])))
    chips = pd.DataFrame(rows).sort_values("cnum").reset_index(drop=True)
    assert len(chips) == 330 and (chips.cnum.values == np.arange(1, 331)).all()
    lab_c = chips[["L_c", "a_c", "b_c"]].values
    lab_d65 = lab_c_to_lab_d65(lab_c)
    chips["L"], chips["a"], chips["b"] = lab_d65[:, 0], lab_d65[:, 1], lab_d65[:, 2]
    chips["grid"] = chips["row"] + chips["hue"].astype(str)

    terms = pd.DataFrame([ln.split("\t") for ln in _read_cr(os.path.join(raw_dir, "term.txt"))],
                         columns=["lang", "spkr", "cnum", "term"])
    terms["lang"] = terms.lang.astype(int); terms["spkr"] = terms.spkr.astype(int)
    terms["cnum"] = terms.cnum.astype(int)

    grid_to_cnum = dict(zip(chips.grid, chips.cnum))
    frows = []
    for ln in _read_cr(os.path.join(raw_dir, "foci-exp.txt")):
        p = ln.split("\t")
        if len(p) < 5:
            continue
        code = p[4].strip()
        # rows A (white) and J (black) are single chips spanning the whole row of the chart
        if code and code[0] in "AJ":
            code = code[0] + "0"
        if code not in grid_to_cnum:
            continue
        frows.append((int(p[0]), int(p[1]), p[3], grid_to_cnum[code]))
    foci = pd.DataFrame(frows, columns=["lang", "spkr", "term", "cnum"])

    srows = []
    for ln in _read_cr(os.path.join(raw_dir, "spkr-lsas.txt")):
        p = ln.split("\t")
        if len(p) >= 4:
            srows.append((int(p[0]), int(p[1]), p[2], p[3]))
    speakers = pd.DataFrame(srows, columns=["lang", "spkr", "age", "sex"])

    w = WCS(chips=chips, terms=terms, foci=foci, speakers=speakers)
    w.lab = lab_d65
    w.grid_of = dict(zip(chips.cnum, chips.grid))
    return w


@dataclass
class Language:
    lang: int
    terms: list[str]                       # kept lexicon, index = term id
    prototypes: np.ndarray                 # m x 3 shared prototypes (CIELAB D65)
    proto_chip: list                       # cnum of shared prototype or None
    speakers: list[int]                    # kept speaker ids
    labels: np.ndarray                     # n x 330 int, -1 = unlabeled
    private_prototypes: np.ndarray         # n x m x 3
    n_speakers_total: int
    prototypes_focal: np.ndarray = None    # m x 3 modal focal chips (alternative)
    modal: np.ndarray = None               # 330 modal labels


def build_language(w: WCS, lang: int) -> Language | None:
    t = w.terms[(w.terms.lang == lang) & (w.terms.term != "*")]   # '*' = no term given
    spk_all = sorted(w.terms[w.terms.lang == lang].spkr.unique())
    n_all = len(spk_all)
    # usage per term: fraction of speakers using it
    use = t.groupby("term").spkr.nunique() / n_all
    # modal term per chip
    modal = t.groupby(["cnum", "term"]).size().reset_index(name="c")
    modal = modal.sort_values(["cnum", "c"], ascending=[True, False]).drop_duplicates("cnum")
    modal_size = modal.groupby("term").size()
    kept = [tm for tm in use.index
            if use[tm] >= TERM_MIN_SPEAKER_FRAC and modal_size.get(tm, 0) >= TERM_MIN_MODAL_CHIPS]
    if len(kept) < 2:
        return None
    tid = {tm: i for i, tm in enumerate(kept)}
    # speaker label matrices
    lab_mat = {}
    for s in spk_all:
        ts = t[t.spkr == s]
        arr = -np.ones(330, dtype=int)
        for c, tm in zip(ts.cnum.values, ts.term.values):
            if tm in tid:
                arr[c - 1] = tid[tm]
        lab_mat[s] = arr
    kept_spk = [s for s in spk_all if (lab_mat[s] >= 0).sum() >= SPEAKER_MIN_LABELED]
    if len(kept_spk) < 2:
        return None
    labels = np.stack([lab_mat[s] for s in kept_spk])
    # modal label per chip among kept speakers
    modal_lab = -np.ones(330, dtype=int)
    for c in range(330):
        col = labels[:, c]; col = col[col >= 0]
        if len(col):
            modal_lab[c] = np.bincount(col, minlength=len(kept)).argmax()
    # shared prototypes (primary): centroid in CIELAB of the language's modal map for the term
    protos = np.zeros((len(kept), 3))
    for i in range(len(kept)):
        chips_i = np.where(modal_lab == i)[0]
        if len(chips_i) == 0:
            chips_i = np.where((labels == i).any(axis=0))[0]
        protos[i] = w.lab[chips_i].mean(axis=0)
    # alternative shared prototypes: modal focal chip per term (the survey's best examples)
    f = w.foci[(w.foci.lang == lang) & (w.foci.term.isin(kept))]
    protos_focal = protos.copy(); proto_chip = [None] * len(kept)
    for tm, i in tid.items():
        fc = f[f.term == tm].cnum
        if len(fc) > 0:
            c = Counter(fc.values).most_common(1)[0][0]
            protos_focal[i] = w.lab[c - 1]; proto_chip[i] = int(c)
    # private prototypes: centroid of each speaker's own chips per term, fallback shared
    priv = np.tile(protos[None, :, :], (len(kept_spk), 1, 1))
    for k in range(len(kept_spk)):
        for i in range(len(kept)):
            chips_i = np.where(labels[k] == i)[0]
            if len(chips_i):
                priv[k, i] = w.lab[chips_i].mean(axis=0)
    return Language(lang=lang, terms=kept, prototypes=protos, proto_chip=proto_chip,
                    speakers=kept_spk, labels=labels, private_prototypes=priv,
                    n_speakers_total=n_all, prototypes_focal=protos_focal, modal=modal_lab)


# ----------------------------------------------------------------------------
# Metric fitting (semidefinite program)
# ----------------------------------------------------------------------------
def _sym6_to_M(m6: np.ndarray) -> np.ndarray:
    a, b, c, d, e, f = m6
    return np.array([[a, b, c], [b, d, e], [c, e, f]])


def _coef_rows(X: np.ndarray, labels: np.ndarray, P: np.ndarray):
    """Rows a_{t,j} with a . m6 = (x_t - m_{ij})^T M d_{ij}, i = label(t); also (i, j) per row."""
    T = X.shape[0]; m = P.shape[0]
    rows, owner, pairs = [], [], []
    for t in range(T):
        i = labels[t]
        if i < 0:
            continue
        for j in range(m):
            if j == i:
                continue
            d = P[i] - P[j]; mid = 0.5 * (P[i] + P[j]); v = X[t] - mid
            G = np.outer(v, d)
            S = 0.5 * (G + G.T)
            rows.append([S[0, 0], 2 * S[0, 1], 2 * S[0, 2], S[1, 1], 2 * S[1, 2], S[2, 2]])
            owner.append(t); pairs.append((i, j))
    return np.array(rows), np.array(owner), np.array(pairs)


def fit_metric(X: np.ndarray, labels: np.ndarray, P: np.ndarray,
               eps: float = FIT_EPS, weights: bool = False, solver: str = "CLARABEL") -> dict:
    """Fit a symmetric positive definite M (trace = 3) so that nearest-prototype
    labeling under M reproduces `labels` as far as possible: minimize the total
    hinge violation of the wall constraints at a fixed small margin.
    With weights=True, fit a power (Laguerre) diagram: per-term weights w are added,
    which shift walls parallel to themselves without tilting them."""
    Xs = X / LAB_SCALE; Ps = P / LAB_SCALE
    A, owner, pairs = _coef_rows(Xs, labels, Ps)
    m = P.shape[0]
    labeled = np.where(labels >= 0)[0]
    if len(A) == 0:
        return None
    m6 = cp.Variable(6)
    M = cp.bmat([[m6[0], m6[1], m6[2]], [m6[1], m6[3], m6[4]], [m6[2], m6[4], m6[5]]])
    xi = cp.Variable(len(labeled), nonneg=True)
    pos = {t: k for k, t in enumerate(labeled)}
    Sel = np.zeros((len(A), len(labeled)))
    for r, t in enumerate(owner):
        Sel[r, pos[t]] = 1.0
    if weights:
        wv = cp.Variable(m)
        Wsel = np.zeros((len(A), m))
        for r, (i, j) in enumerate(pairs):
            Wsel[r, i] += 1.0; Wsel[r, j] -= 1.0
        lhs = 2 * (A @ m6) + Wsel @ wv
        extra = [cp.sum(wv) == 0]
    else:
        lhs = A @ m6
        extra = []
    cons = [lhs >= FIT_MARGIN - Sel @ xi,
            M >> eps * np.eye(3),
            cp.trace(M) == FIT_TRACE] + extra
    prob = cp.Problem(cp.Minimize(cp.sum(xi)), cons)
    try:
        prob.solve(solver=solver)
    except Exception:
        prob.solve(solver="SCS")
    if m6.value is None:
        return None
    Mv = _sym6_to_M(m6.value)
    Mv = 0.5 * (Mv + Mv.T)
    ev = np.clip(np.linalg.eigvalsh(Mv), 1e-9, None)
    kappa = float(np.sqrt(ev.max() / ev.min()))
    W = (wv.value * LAB_SCALE ** 2) if weights else None    # back to CIELAB^2 units
    pred = predict_labels(X, Mv, P, W)
    ok = labels >= 0
    acc = float((pred[ok] == labels[ok]).mean())
    return dict(M=Mv, kappa=kappa, weights=W, slack=float(xi.value.sum()),
                accuracy=acc, n_labeled=int(ok.sum()), status=prob.status)


def predict_labels(X: np.ndarray, M: np.ndarray, P: np.ndarray, W: np.ndarray | None = None) -> np.ndarray:
    diff = X[:, None, :] - P[None, :, :]                  # T x m x 3
    q = np.einsum("tmi,ij,tmj->tm", diff, M, diff)
    if W is not None:
        q = q - W[None, :]
    return q.argmin(axis=1)


# ----------------------------------------------------------------------------
# Geometry: core radius, patterns, tilt
# ----------------------------------------------------------------------------
def wall_distance(P: np.ndarray, i: int, j: int, M: np.ndarray) -> float:
    """Distance from prototype i to the wall between i and j under metric M."""
    d = P[i] - P[j]
    Md = M @ d
    return float(d @ Md / (2.0 * np.linalg.norm(Md)))


def core_radius(P: np.ndarray, i: int, Ms: list[np.ndarray]) -> float:
    """Largest ball around prototype i inside the intersection of all agents' cells."""
    m = P.shape[0]
    return min(wall_distance(P, i, j, M) for M in Ms for j in range(m) if j != i)


def theory_bound(P: np.ndarray, i: int, kappa: float) -> float:
    m = P.shape[0]
    dmin = min(np.linalg.norm(P[i] - P[j]) for j in range(m) if j != i)
    return dmin * kappa / (1.0 + kappa ** 2)


def kappa_of(M: np.ndarray) -> float:
    ev = np.clip(np.linalg.eigvalsh(M), 1e-9, None)
    return float(np.sqrt(ev.max() / ev.min()))


def count_patterns(G: np.ndarray, Ms: list[np.ndarray], P: np.ndarray) -> int:
    prof = np.stack([predict_labels(G, M, P) for M in Ms], axis=1)
    return len(np.unique(prof, axis=0))


def tilt(M1: np.ndarray, M2: np.ndarray, d: np.ndarray) -> float:
    u, v = M1 @ d, M2 @ d
    c = u @ v / (np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def random_grid(lab: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    lo, hi = lab.min(axis=0), lab.max(axis=0)
    return rng.uniform(lo, hi, size=(n, 3))


# ----------------------------------------------------------------------------
# H1: convexity of empirical common ground on the chart grid
# ----------------------------------------------------------------------------
def chart_coords(w: WCS, cnums: list[int], focal_hue: int) -> np.ndarray:
    """(hue unwrapped around focal, value row) for chromatic chips; achromatic -> NaN."""
    ch = w.chips.set_index("cnum")
    out = []
    for c in cnums:
        h = int(ch.loc[c, "hue"]); r = ch.loc[c, "row"]
        if h == 0:
            out.append([np.nan, np.nan]); continue
        hh = ((h - focal_hue + 20) % 40)
        out.append([hh, ord(r) - ord("A")])
    return np.array(out, dtype=float)


def hull_fill(w: WCS, Q: list[int], focal_hue: int) -> float | None:
    """Fraction of chips inside the convex hull of Q (chart coordinates) that are in Q."""
    pts = chart_coords(w, Q, focal_hue)
    pts = pts[~np.isnan(pts[:, 0])]
    if len(pts) < 3:
        return None
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None            # collinear / degenerate
    eq = hull.equations
    all_pts = chart_coords(w, list(range(1, 331)), focal_hue)
    inside = []
    for c in range(1, 331):
        p = all_pts[c - 1]
        if np.isnan(p[0]):
            continue
        if np.all(eq[:, :2] @ p + eq[:, 2] <= 1e-9):
            inside.append(c)
    if not inside:
        return None
    Qs = set(Q)
    return sum(1 for c in inside if c in Qs) / len(inside)


# ----------------------------------------------------------------------------
# xkcd loading
# ----------------------------------------------------------------------------
def load_xkcd_basic(db_path: str, rng: np.random.Generator,
                    n_users: int = XKCD_SAMPLE_USERS) -> tuple[pd.DataFrame, list[int]]:
    con = sqlite3.connect(db_path)
    names = ", ".join("'%s'" % t for t in BASIC_TERMS + ["grey"])
    basic = pd.read_sql_query(
        "select a.user_id, a.r, a.g, a.b, lower(trim(a.colorname)) as term "
        "from answers a join users u on a.user_id = u.id "
        "where lower(trim(a.colorname)) in (%s) and u.spamprob < %f and u.colorblind = 0"
        % (names, XKCD_MAX_SPAMPROB), con)
    con.close()
    basic["term"] = basic.term.replace({"grey": "gray"})
    counts = basic.groupby("user_id").size()
    eligible = counts[counts >= XKCD_MIN_BASIC].index.values
    chosen = rng.choice(eligible, size=min(n_users, len(eligible)), replace=False)
    sub = basic[basic.user_id.isin(chosen)].copy()
    lab = srgb_to_lab_d65(sub[["r", "g", "b"]].values)
    sub["L"], sub["a"], sub["b_"] = lab[:, 0], lab[:, 1], lab[:, 2]
    sub["tid"] = sub.term.map({t: i for i, t in enumerate(BASIC_TERMS)})
    # prototypes: per-term median CIELAB over ALL eligible users' basic answers
    allb = basic.copy()
    lab_all = srgb_to_lab_d65(allb[["r", "g", "b"]].values)
    allb["L"], allb["a"], allb["b_"] = lab_all[:, 0], lab_all[:, 1], lab_all[:, 2]
    protos = np.array([allb[allb.term == t][["L", "a", "b_"]].median().values for t in BASIC_TERMS])
    return sub, protos, len(eligible), int(len(counts))
