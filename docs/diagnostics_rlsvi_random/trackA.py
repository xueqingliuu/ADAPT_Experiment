import numpy as np, glob, os, sys
AG="results_vanilla_zero_v12/aggregated_20260905-121620"
files=sorted(glob.glob(f"{AG}/*_aggregated.npz"))
D={}
for f in files:
    n=os.path.basename(f).replace("_aggregated.npz","")
    z=np.load(f,allow_pickle=True); D[n]={k:z[k] for k in z.files}
print("keys:", list(D['random_send'].keys()))
for k,v in D['random_send'].items(): print(k, v.shape)
ref='random_send'
def paired(field, wk=slice(1,None)):
    R=D[ref][field][...,wk]
    out={}
    for n in D:
        X=D[n][field][...,wk]
        d=np.nanmean(X-R,axis=(1,2))   # per-seed paired delta (avg over slots, weeks)
        out[n]=(np.nanmean(X), d.mean(), d.std(ddof=1)/np.sqrt(len(d)), len(d))
    return out
for field in ['cae_noisy_runs','cae_latent_runs']:
    print(f"\n=== {field} weeks 1-36 paired vs random ===")
    for n,(m,d,se,N) in sorted(paired(field).items(), key=lambda x:-x[1][1]):
        print(f"{n:32s} mean={m:.4f} d={d:+.5f} se={se:.5f} z={d/se:+.2f} n={N}")
# week 0 identical?
for n in D:
    diff=np.nanmax(np.abs(D[n]['cae_noisy_runs'][...,0]-D[ref]['cae_noisy_runs'][...,0]))
    if diff>0: print("WEEK0 DIFFERS", n, diff)
print("\n=== week-stratified noisy paired Δ vs random ===")
for lo,hi in [(1,13),(13,25),(25,37),(1,7),(19,37)]:
    print(f"-- weeks {lo}-{hi-1}")
    for n in ['rl_v12_pooled_adv_m_g09','rl_v10_pooled_g09','rl_v1_base_g09','rl_v11_adv_m_g09','rl_v6_invariant_redistributed','always_send','never_send']:
        m,d,se,N=paired('cae_noisy_runs',slice(lo,hi))[n]
        print(f"   {n:32s} d={d:+.5f} se={se:.5f} z={d/se:+.2f}")
# CRN check: correlation of per-seed level means
print("\n=== CRN: corr of per-seed mean CAE with random_send; SE of unpaired diff ===")
Rm=np.nanmean(D[ref]['cae_noisy_runs'][...,1:],axis=(1,2))
for n in D:
    Xm=np.nanmean(D[n]['cae_noisy_runs'][...,1:],axis=(1,2))
    se_unp=np.sqrt(Xm.var(ddof=1)/len(Xm)+Rm.var(ddof=1)/len(Rm))
    print(f"{n:32s} corr={np.corrcoef(Xm,Rm)[0,1]:.4f} se_unpaired={se_unp:.5f}")
# pi stats
print("\n=== pi_A stats (weeks 1-35 idx) ===")
for n in D:
    p=D[n]['piA_runs']
    if p is None or p.size==0: continue
    p1=p[:,:,1:,:,:]
    fr=np.nanmean((p1<0.45)|(p1>0.55))
    print(f"{n:32s} mean={np.nanmean(p1):.4f} sd={np.nanstd(p1):.4f} frac_outside[.45,.55]={fr:.3f} frac==0.1={np.nanmean(p1<=0.1+1e-9):.3f} frac==0.9={np.nanmean(p1>=0.9-1e-9):.3f} wk18+={np.nanmean(p[:,:,18:]):.4f}")

