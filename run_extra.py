"""
run_extra.py -- (1) per-term weights fitted with the metric held fixed,
(2) core and protected-region counts for the worked example (WCS language 87),
(3) tilt statistics and the most compressed CIELAB axis of the fitted metrics.
"""
import numpy as np, pandas as pd, cvxpy as cp
import config
import cg_lib as cg, analysis as an
w = cg.load_wcs(config.WCS_DIR)
langs, ltab = an.cached("langs", lambda: an.build_all(w)); fits = an.cached("fits", lambda: an.fit_all(w, langs))

def fit_weights_only(X, labels, P, M):
    Xs=X/cg.LAB_SCALE; Ps=P/cg.LAB_SCALE
    A, owner, pairs = cg._coef_rows(Xs, labels, Ps)
    m6=np.array([M[0,0],M[0,1],M[0,2],M[1,1],M[1,2],M[2,2]])
    base=2*(A@m6)
    labeled=np.where(labels>=0)[0]; pos={t:k for k,t in enumerate(labeled)}
    Sel=np.zeros((len(A),len(labeled)))
    for r,t in enumerate(owner): Sel[r,pos[t]]=1
    Wsel=np.zeros((len(A),P.shape[0]))
    for r,(i,j) in enumerate(pairs): Wsel[r,i]+=1; Wsel[r,j]-=1
    wv=cp.Variable(P.shape[0]); xi=cp.Variable(len(labeled),nonneg=True)
    prob=cp.Problem(cp.Minimize(cp.sum(xi)),[base+Wsel@wv>=cg.FIT_MARGIN-Sel@xi, cp.sum(wv)==0])
    prob.solve(solver="CLARABEL")
    W=wv.value*cg.LAB_SCALE**2
    pred=cg.predict_labels(X,M,P,W); ok=labels>=0
    return float((pred[ok]==labels[ok]).mean())

def stage():
    rows=[]
    for lang,L in langs.items():
        sub=fits[fits.lang==lang].sort_values("k")
        for k in range(len(L.speakers)):
            M=sub.M_shared.iloc[k]
            rows.append(dict(lang=lang,k=k,acc_weights_only=fit_weights_only(w.lab,L.labels[k],L.prototypes,M)))
    return pd.DataFrame(rows)
wo=an.cached("weights_only", stage)
print("weights-only (M fixed) mean accuracy %.3f median %.3f"%(wo.acc_weights_only.mean(), wo.acc_weights_only.median()))
print("for reference: euclid %.3f shared %.3f weighted(joint) %.3f"%(fits.acc_euclid.mean(),fits.acc_shared.mean(),fits.acc_weighted.mean()))
wo.to_csv("results/weights_only.csv",index=False)

# protected region counts on the worked example (lang 87)
lang=87; L=langs[lang]; sub=fits[fits.lang==lang].sort_values("k"); Ms=list(sub.M_shared); P=L.prototypes; m=len(P)
kap=max(cg.kappa_of(M) for M in Ms); q=(kap**2-1)/(kap**2+1)
preds=np.stack([cg.predict_labels(w.lab,M,P) for M in Ms])
out=[]
for i in range(m):
    dmin=min(np.linalg.norm(P[i]-P[j]) for j in range(m) if j!=i); rho=kap/(1+kap**2)*dmin
    ball=(np.linalg.norm(w.lab-P[i],axis=1)<=rho)
    prot=np.ones(330,bool)
    for j in range(m):
        if j==i: continue
        d=P[i]-P[j]; dh=d/np.linalg.norm(d); mid=(P[i]+P[j])/2; v=w.lab-mid
        prot&=(v@dh>=q*np.linalg.norm(v,axis=1)-1e-9)
    cg_model=(preds==i).all(axis=0)
    emp=(L.labels==i).all(axis=0)
    out.append((L.terms[i],int(ball.sum()),int(prot.sum()),int(cg_model.sum()),int(emp.sum()),bool((ball&~prot).sum()==0),bool((prot&~cg_model).sum()==0)))
print("lang 87 kappa_max %.2f q %.3f"%(kap,q))
print("term | chips in ball | in protected region | in model common ground | in empirical common ground | ball⊆prot | prot⊆model")
for r in out: print(r)
tot=[sum(r[i] for r in out) for i in range(1,5)]; print("totals",tot)
open("results/protected_region_lang87.txt","w").write("kappa_max %.2f q %.3f\n"%(kap,q)+"\n".join(map(str,out))+"\ntotals "+str(tot)+"\n")

# tilts of the fitted walls and the most compressed axis of each fitted metric
base_t=[]; pair_t=[]; axes=[]
for lang,L in langs.items():
    sub=fits[fits.lang==lang].sort_values("k"); Ms=list(sub.M_shared); P=L.prototypes; m=len(P)
    for M in Ms:
        ev,V=np.linalg.eigh(M); axes.append(int(np.argmax(np.abs(V[:,0]))))
        for i in range(m):
            for j in range(i+1,m):
                d=P[i]-P[j]; u=M@d
                base_t.append(np.degrees(np.arccos(np.clip(u@d/np.linalg.norm(u)/np.linalg.norm(d),-1,1))))
    for a in range(len(Ms)):
        for b in range(a+1,len(Ms)):
            for i in range(m):
                for j in range(i+1,m):
                    d=P[i]-P[j]; u=Ms[a]@d; v=Ms[b]@d
                    pair_t.append(np.degrees(np.arccos(np.clip(u@v/np.linalg.norm(u)/np.linalg.norm(v),-1,1))))
base_t=np.array(base_t); pair_t=np.array(pair_t); axes=np.array(axes)
pd.DataFrame([dict(tilt_vs_base_mean=base_t.mean(), tilt_vs_base_max=base_t.max(),
                   tilt_between_speakers_mean=pair_t.mean(), tilt_between_speakers_max=pair_t.max(),
                   compressed_axis_share_L=(axes==0).mean(), compressed_axis_share_a=(axes==1).mean(),
                   compressed_axis_share_b=(axes==2).mean())]).to_csv("results/tilts_and_axes.csv", index=False)
print("tilt from the base diagram: mean %.1f deg, max %.1f deg" % (base_t.mean(), base_t.max()))
print("tilt between two speakers: mean %.1f deg" % pair_t.mean())
print("most compressed axis: L %.2f, a %.2f, b %.2f" % tuple((axes==k).mean() for k in range(3)))
