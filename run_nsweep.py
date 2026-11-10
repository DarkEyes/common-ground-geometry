"""
run_nsweep.py -- follow-up predictions on sampling: observed pattern counts on
N = 2,000, 20,000, and 100,000 points, and the share of the box in D_1.

    python run_nsweep.py                 # sweep, then summary
    python run_nsweep.py --summary-only  # summary from the saved CSV files
"""
import sys, numpy as np, pandas as pd
import config
import cg_lib as cg, analysis as an
w = cg.load_wcs(config.WCS_DIR)
langs, ltab = an.cached("langs", lambda: an.build_all(w))
fits = an.cached("fits", lambda: an.fit_all(w, langs))


def summarize():
    """Summary of the sample-size sweep, computed from the saved CSV files."""
    df=pd.read_csv("results/h3_nsweep.csv"); dv=pd.read_csv("results/disagreement_volume.csv")
    sizes=[2,4,8,16,24]; out=[]
    for N,g in df.groupby("N"):
        sl=[]
        for lang,gg in g.groupby("lang"):
            m=gg.groupby("n").patterns.mean()
            sl.append(np.polyfit(np.log(m.index.values),np.log(m.values),1)[0])
        out.append("N=%d: slope median %.2f IQR %.2f-%.2f; patterns median at n=2: %.0f, n=8: %.0f, n=24: %.0f"%(
            N,np.median(sl),np.percentile(sl,25),np.percentile(sl,75),
            g[g.n==2].patterns.median(),g[g.n==8].patterns.median(),g[g.n==24].patterns.median()))
    for N in sorted(df.N.unique()):
        g=df[df.N==N].groupby(["lang","n"]).patterns.mean().unstack()
        cols=[c for c in sizes if c in g.columns]
        loc=np.log(g[cols[1:]].values/g[cols[:-1]].values)/np.log(np.array(cols[1:])/np.array(cols[:-1]))
        out.append("N=%d local slopes %s: %s"%(N,["%d-%d"%(a,b) for a,b in zip(cols[:-1],cols[1:])],np.round(np.nanmedian(loc,axis=0),2)))
    out.append("share of the box in D_1 by n (median over languages): "+str(dv.groupby("n").nonunanimous.median().round(3).to_dict()))
    mterms={l:len(L.terms) for l,L in langs.items()}
    m=df.merge(dv.groupby(["lang","n"]).nonunanimous.mean().reset_index(),on=["lang","n"])
    m["ratio"]=m.patterns/(m.lang.map(mterms)+m["N"]*m.nonunanimous)
    out.append("max over rows of patterns/(m + N*share of D_1): %.3f"%m.ratio.max())
    out.append("median of that ratio at n=24 by N: "+str(m[m.n==24].groupby("N").ratio.median().round(3).to_dict()))
    out.append("languages: %d"%df.lang.nunique())
    open("results/nsweep_summary.md","w").write("\n".join(out)); print("\n".join(out))


def sweep():
    rng = an.stage_rng(21)
    rows=[]; vol=[]
    Ns=[2000,20000,100000]
    G_big = cg.random_grid(w.lab, max(Ns), rng)
    sizes=[2,4,8,16,24]
    for lang, L in langs.items():
        n=len(L.speakers)
        if n < 16: continue
        sub=fits[fits.lang==lang].sort_values("k"); Ms=list(sub.M_shared)
        base = cg.predict_labels(G_big, np.eye(3), L.prototypes)   # base (Euclidean) labeling
        preds = np.stack([cg.predict_labels(G_big, M, L.prototypes) for M in Ms])
        for s in [x for x in sizes if x<=n]:
            for rep in range(2 if s<n else 1):
                idx = rng.choice(n, size=s, replace=False)
                pr = preds[idx]
                for N in Ns:
                    p = pr[:, :N]
                    rows.append(dict(lang=lang, n=s, N=N, patterns=len(np.unique(p.T, axis=0))))
                # disagreement volume: fraction of the largest grid where not all agents agree
                vol.append(dict(lang=lang, n=s, nonunanimous=float((pr != pr[0]).any(axis=0).mean()),
                                differs_from_base=float((pr != base[None,:]).any(axis=0).mean())))
    df=pd.DataFrame(rows); dv=pd.DataFrame(vol)
    df.to_csv("results/h3_nsweep.csv",index=False); dv.to_csv("results/disagreement_volume.csv",index=False)


if __name__ == "__main__":
    if "--summary-only" not in sys.argv:
        sweep()
    summarize()
