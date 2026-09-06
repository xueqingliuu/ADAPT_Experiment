import numpy as np, glob, sys
Z=[np.load(f,allow_pickle=True) for f in sorted(glob.glob(sys.argv[1]))]
S=np.concatenate([z['states'] for z in Z],0)
cols=['uid','seed','k','d','t','dl','df','da','burden','wear7','pv7','antic_y','yest','p2h','inter7','act7','pu','cae','wknd','Ehat']
ix={c:i for i,c in enumerate(cols)}
uid=S[:,0]; dl=S[:,ix['dl']]
print("n rows",len(S),"users",len(np.unique(uid)))
print("Δ_myopic (normalized CAE units): mean %.2e sd %.2e; frac>0 %.3f"%(dl.mean(), dl.std(), (dl>0).mean()))
# decomposition: between-user vs within-user variance
um={u:dl[uid==u].mean() for u in np.unique(uid)}
between=np.var([um[u] for u in uid]); within=dl.var()-between
print("var between users %.2e  within-user %.2e  (within share %.2f)"%(between, within, within/dl.var()))
# fourSC part vs antic part contribution
th_f=S[:,ix['df']]; th_a=S[:,ix['da']]
print("corr(dl, fourSC part) and (dl, antic part) -- via per-row parts (weights applied inside dl):")
# regressions within user: features available to RLSVI adv block vs not
def ols(X,y):
    X=np.column_stack([np.ones(len(y)),X]); b=np.linalg.lstsq(X,y,rcond=None)[0]; r=y-X@b; return 1-r.var()/y.var(), b
adv_feats=['Ehat','cae','yest','p2h','act7','burden','inter7']   # ≈ RLSVI A×[E,b̂,C]
miss_feats=['wear7','pv7','antic_y','t','pu']                   # in env CATE but not in RLSVI advantage (pu latent; Ehat proxy)
all_feats=adv_feats+['wear7','pv7','antic_y','t']
r2_adv=[];r2_all=[];r2_pool=[]
for u in np.unique(uid):
    m=uid==u; y=dl[m]
    r2_adv.append(ols(S[m][:,[ix[c] for c in adv_feats]],y)[0])
    r2_all.append(ols(S[m][:,[ix[c] for c in all_feats]],y)[0])
print("per-user linear R2 of Δ on RLSVI-available adv features: mean %.3f (min %.3f max %.3f)"%(np.mean(r2_adv),min(r2_adv),max(r2_adv)))
print("per-user linear R2 of Δ on all env features incl wear7/pv7/antic_y/slot: mean %.3f"%np.mean(r2_all))
# pooled model (one β for all users) -- what V10/V12 can represent
r2p,b=ols(S[:,[ix[c] for c in adv_feats]],dl); print("pooled R2 (adv feats, common β, no user intercept): %.3f"%r2p)
Xu=np.column_stack([(uid==u).astype(float) for u in np.unique(uid)])
X=np.column_stack([Xu, S[:,[ix[c] for c in adv_feats]]]); bb=np.linalg.lstsq(X,dl,rcond=None)[0]; r=dl-X@bb
print("pooled R2 with user intercepts + common slopes: %.3f"%(1-r.var()/dl.var()))
# sign agreement: pooled-common-slope prediction vs true dl (what V12 could at best do)
pred=X@bb; print("sign agreement pooled(intercept+slopes) vs true Δ: %.3f ; common-slope-only: %.3f"%(((pred>0)==(dl>0)).mean(), ((np.column_stack([np.ones(len(dl)),S[:,[ix[c] for c in adv_feats]]])@b>0)==(dl>0)).mean()))
# which single features carry it (per-user corr averaged)
for c in ['burden','wear7','pv7','antic_y','yest','p2h','inter7','act7','pu','cae','Ehat','t','d','k']:
    cs=[np.corrcoef(S[uid==u][:,ix[c]],dl[uid==u])[0,1] for u in np.unique(uid) if np.std(S[uid==u][:,ix[c]])>0]
    print(f"  mean within-user corr(Δ,{c:8s}) = {np.nanmean(cs):+.3f}   mean |corr| = {np.nanmean(np.abs(cs)):.3f}")
# per-slot effect magnitude in raw CAE
print("sd of Δ within user (raw CAE units, x0.93): %.4f ; weekly CAE noise sd raw ~0.25"%(np.sqrt(within)*0.93))
print("mean |Δ| raw: %.4f"%(np.abs(dl).mean()*0.93))
