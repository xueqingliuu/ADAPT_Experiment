import numpy as np, glob, sys, json
files=sorted(glob.glob(sys.argv[1]))
Z=[np.load(f,allow_pickle=True) for f in files]
pols=list(Z[0]['pols']); uids=np.concatenate([z['uids'] for z in Z])
cae={p:np.concatenate([z[f'cae_{p}'] for z in Z],0) for p in pols}
lat={p:np.concatenate([z[f'lat_{p}'] for z in Z],0) for p in pols}
rate={p:np.concatenate([z[f'rate_{p}'] for z in Z],0) for p in pols}
shift,scale=5.132,0.93
def raw(x): return shift+scale*x
ref='bern0.5'
print(f"{'policy':10s} {'rawCAE':>8s} {'Δ vs bern.5':>12s} {'pairedSE':>9s} {'z':>6s} {'latentΔ':>9s} {'z_lat':>6s} {'rate':>6s}")
for p in pols:
    d=np.nanmean(raw(cae[p][...,1:])-raw(cae[ref][...,1:]),axis=(0,2))  # per seed, avg users & weeks
    dl=np.nanmean(raw(lat[p][...,1:])-raw(lat[ref][...,1:]),axis=(0,2))
    se=d.std(ddof=1)/np.sqrt(len(d)); sel=dl.std(ddof=1)/np.sqrt(len(dl))
    print(f"{p:10s} {np.nanmean(raw(cae[p][...,1:])):8.4f} {d.mean():+12.5f} {se:9.5f} {d.mean()/se:6.2f} {dl.mean():+9.5f} {dl.mean()/sel:6.2f} {np.nanmean(rate[p]):6.3f}")
print("\nPer-user latent Δ vs bern0.5 (raw scale), best constant p, and rank-oracle gain:")
best=[]
for i,u in enumerate(uids):
    row={p: np.nanmean(raw(lat[p][i,:,1:])-raw(lat[ref][i,:,1:])) for p in pols}
    bp=max(['bern0.0','bern0.25','bern0.5','bern0.75','bern1.0'], key=lambda p: row[p])
    best.append(row[bp])
    print(f"uid {u:4d}  never={row['bern0.0']:+.4f} p.25={row['bern0.25']:+.4f} p.75={row['bern0.75']:+.4f} always={row['bern1.0']:+.4f} | sign={row['sign']:+.4f} rank={row['rank']:+.4f} anti={row['antirank']:+.4f}  best_p={bp}")
print("\nmean over users of best-constant-p gain vs bern0.5 (user-level send-rate personalization oracle, optimistic/selection-biased):", np.mean(best))
