"""
make_figures.py -- regenerate every figure in the paper from the saved results.

Run after run_stages.py, run_nsweep.py, and run_extra.py, or with the shipped
results/ folder:

    python make_figures.py

The figures go to paper_figures/. The file names match the names that main.tex
includes. The worked example also needs the WCS raw files (python get_data.py --wcs).

Figure                         File                          Inputs
Figure 1 (worked example)      worked_example.png            results/worked_example_fits.npz, WCS raw files
Figure 3, left   (H1)          h1_hullfill.png               results/h1_convexity.csv
Figure 3, middle (H2)          h2_core_vs_kappa_right.png    results/h2_core.csv
Figure 3, right  (H3)          h3_patterns.png               results/h3_patterns.csv
Figure 4 (follow-up sampling)  nsweep.png                    results/h3_nsweep.csv, results/disagreement_volume.csv
Appendix C figure (H5)         h5_tilt.png                   results/h5_tilt.csv
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import config

RES = config.RESULTS
WCS_DIR = config.WCS_DIR
OUT = config.FIGURES
os.makedirs(OUT, exist_ok=True)
NS = np.array([2, 4, 8, 16, 24])


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)


def fig_h1():
    h1 = pd.read_csv(os.path.join(RES, "h1_convexity.csv"))
    t = h1[h1.testable == True]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.scatter(t.baseline_median, t.hull_fill, s=12, alpha=0.6)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("baseline median hull-fill (random chip sets)")
    ax.set_ylabel("hull-fill $H(Q)$ of common ground")
    save(fig, "h1_hullfill.png")


def fig_h2():
    h2 = pd.read_csv(os.path.join(RES, "h2_core.csv"))
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.scatter(h2.fixedn_kappa_max, h2.fixedn_relative_core, s=14)
    ax.set_xlabel(r"fitted anisotropy $\hat\kappa$ (five speakers)")
    ax.set_ylabel("median relative core (five speakers)")
    save(fig, "h2_core_vs_kappa_right.png")


def fig_h3():
    h3 = pd.read_csv(os.path.join(RES, "h3_patterns.csv"))
    fig, ax = plt.subplots(figsize=(5, 3.8))
    for _, g in h3.groupby("lang"):
        gg = g.groupby("n").patterns.mean()
        ax.plot(gg.index, gg.values, alpha=0.25, lw=1)
    ax.plot(NS, 30 * NS ** 3, "k--", label=r"$n^3$ reference")
    ax.plot(NS, 30 * NS ** 2, "k:", label=r"$n^2$ reference")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of speakers $n$")
    ax.set_ylabel(r"observed pattern count $|\Pi(X)|$")
    ax.legend()
    save(fig, "h3_patterns.png")


def fig_nsweep():
    df = pd.read_csv(os.path.join(RES, "h3_nsweep.csv"))
    dv = pd.read_csv(os.path.join(RES, "disagreement_volume.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.3))
    for N, mk in zip([2000, 20000, 100000], ["o", "s", "^"]):
        g = df[df.N == N].groupby("n").patterns.median()
        ax[0].plot(g.index, g.values, marker=mk, label="$N$ = %s" % format(N, ","))
    ax[0].plot(NS, 15 * NS ** 3, "k--", lw=1, label=r"$n^3$ reference")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("number of speakers $n$")
    ax[0].set_ylabel(r"observed pattern count $|\Pi(X)|$, median")
    ax[0].legend(fontsize=7)
    g = dv.groupby("n").nonunanimous.median()
    q1 = dv.groupby("n").nonunanimous.quantile(0.25)
    q3 = dv.groupby("n").nonunanimous.quantile(0.75)
    ax[1].plot(g.index, g.values, marker="o")
    ax[1].fill_between(g.index, q1.values, q3.values, alpha=0.2)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("number of speakers $n$")
    ax[1].set_ylabel(r"share of the box in $D_1$")
    ax[1].set_ylim(0, 1)
    save(fig, "nsweep.png")


def fig_h5():
    h5 = pd.read_csv(os.path.join(RES, "h5_tilt.csv"))
    v = h5.dropna(subset=["rho"])
    fig, ax = plt.subplots(figsize=(4.5, 3.3))
    ax.hist(v.rho, bins=25)
    ax.axvline(0, color="k")
    ax.set_xlabel(r"$r_{\mathrm{S}}$ between tilt and disagreement, per language")
    ax.set_ylabel("number of languages")
    save(fig, "h5_tilt.png")


def fig_worked(lang=87, ka=15, kb=9):
    """Worked example: WCS language 87, speakers with k = 15 and k = 9 (lowest and
    highest fitted anisotropy among speakers with accuracy >= 0.7)."""
    import cg_lib as cg
    w = cg.load_wcs(WCS_DIR)
    # fitted metrics of this language, stored as plain arrays by run_stages.py
    z = np.load(os.path.join(RES, "worked_example_fits.npz"))
    assert int(z["lang"]) == lang
    prototypes, labels, Ms, acc = z["prototypes"], z["labels"], list(z["M"]), z["acc"]
    n_speakers, m = labels.shape[0], prototypes.shape[0]
    Ma, Mb = Ms[ka], Ms[kb]
    acca, accb = acc[ka], acc[kb]
    ch = w.chips.set_index("cnum")

    def chart(labels):
        G = np.full((10, 41), np.nan)
        for c in range(1, 331):
            G[ord(ch.loc[c, "row"]) - ord("A"), int(ch.loc[c, "hue"])] = labels[c - 1]
        return G

    pa = cg.predict_labels(w.lab, Ma, prototypes)
    pb = cg.predict_labels(w.lab, Mb, prototypes)
    preds = np.stack([cg.predict_labels(w.lab, M, prototypes) for M in Ms])
    model_cg = (preds == preds[0]).all(axis=0)
    emp_cg = (labels == labels[0]).all(axis=0) & (labels[0] >= 0)
    cmap = ListedColormap(plt.get_cmap("tab10").colors[:m])
    proto_chips = [int(np.argmin(np.linalg.norm(w.lab - prototypes[i], axis=1))) + 1 for i in range(m)]
    px = [int(ch.loc[c, "hue"]) for c in proto_chips]
    py = [ord(ch.loc[c, "row"]) - ord("A") for c in proto_chips]

    fig, ax = plt.subplots(2, 2, figsize=(9.2, 5.0))

    def frame(a, title):
        a.set_title(title, fontsize=9)
        a.set_yticks(range(10))
        a.set_yticklabels(list("ABCDEFGHIJ"), fontsize=7)
        a.set_xticks([0, 10, 20, 30, 40])
        a.set_xticklabels([0, 10, 20, 30, 40], fontsize=7)
        a.set_xlabel("hue column", fontsize=8)
        a.tick_params(length=2)

    def gray_background(a):
        a.imshow(np.where(np.isnan(chart(pa)), np.nan, 0.85), cmap="gray", vmin=0, vmax=1, aspect="auto")

    for a, (p, M, acc, name, tag) in zip(ax[0], [(pa, Ma, acca, "Speaker 1", "a"), (pb, Mb, accb, "Speaker 2", "b")]):
        a.imshow(chart(p), cmap=cmap, vmin=-0.5, vmax=m - 0.5, aspect="auto", interpolation="nearest")
        a.scatter(px, py, marker="*", s=60, c="white", edgecolors="k", linewidths=0.6, zorder=3)
        frame(a, "(%s) %s, fitted diagram, $\\hat\\kappa$ = %.1f, accuracy %.2f" % (tag, name, cg.kappa_of(M), acc))
    G = chart(np.where(pa == pb, pa, -1).astype(float))
    G[G < 0] = np.nan
    gray_background(ax[1, 0])
    ax[1, 0].imshow(G, cmap=cmap, vmin=-0.5, vmax=m - 0.5, aspect="auto", interpolation="nearest")
    frame(ax[1, 0], "(c) Speakers 1 and 2: common ground; gray = disagreement")
    G = chart(np.where(model_cg, preds[0], -1).astype(float))
    G[G < 0] = np.nan
    gray_background(ax[1, 1])
    ax[1, 1].imshow(G, cmap=cmap, vmin=-0.5, vmax=m - 0.5, aspect="auto", interpolation="nearest")
    ys = [ord(ch.loc[c, "row"]) - ord("A") for c in range(1, 331) if emp_cg[c - 1]]
    xs = [int(ch.loc[c, "hue"]) for c in range(1, 331) if emp_cg[c - 1]]
    ax[1, 1].scatter(xs, ys, s=7, c="k", marker=".")
    frame(ax[1, 1], "(d) All %d speakers: model (color) and empirical (dots) common ground" % n_speakers)
    fig.tight_layout()
    path = os.path.join(OUT, "worked_example.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("wrote", path)
    with open(os.path.join(RES, "worked_example.txt"), "w") as f:
        f.write("lang %d speakers %d terms %d\n" % (lang, n_speakers, m))
        f.write("speaker 1 (k=%d): kappa_hat %.2f accuracy %.2f\n" % (ka, cg.kappa_of(Ma), acca))
        f.write("speaker 2 (k=%d): kappa_hat %.2f accuracy %.2f\n" % (kb, cg.kappa_of(Mb), accb))
        f.write("chips where speakers 1 and 2 disagree: %d of 330\n" % int((pa != pb).sum()))
        f.write("model common ground of all speakers: %d chips\n" % int(model_cg.sum()))
        f.write("empirical common ground of all speakers: %d chips\n" % int(emp_cg.sum()))
        f.write("empirical common ground inside model common ground: %d\n" % int((emp_cg & model_cg).sum()))


if __name__ == "__main__":
    fig_h1()
    fig_h2()
    fig_h3()
    fig_nsweep()
    fig_h5()
    if os.path.isdir(WCS_DIR):
        fig_worked()
    else:
        print("skipped worked_example.png: WCS raw files not found at", WCS_DIR)
