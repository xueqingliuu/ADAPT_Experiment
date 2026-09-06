import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))); os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import experiment as ex, algorithm_helpers as ah
from agents.micro_query import MicroQueryAgent
PD=sys.argv[1] if len(sys.argv)>1 else "env_para_vanilla"
S2=[]; TG=[]
_orig=MicroQueryAgent.update_rlsvi
def upd(self,k):
    _orig(self,k)
    if k>=1 and self._current_betas_rest is not None:
        S2.append((k,self.sigma2_rl))
        Phi,tg=ah.build_rl_training_data(k,self.dataset.A_hist,self.b_hat_hist,self.b_tilde_hist,self.betas_store[k-1],self.betas_target,self.gamma_dt,self.get_state)
        T=np.stack(tg); TG.append((k,np.abs(T).mean(),T.std(axis=0).mean(), Phi.shape))
MicroQueryAgent.update_rlsvi=upd
rows=[]
for uid in [269,106,13,22]:
    S2.clear(); TG.clear()
    seed=ex._episode_seed(0,0)
    r,oenv=ex.run_micro_query(uid,seed=seed,gamma_bar=0.9,params_dir=PD)
    bh=np.asarray(r["b_hat"]); cae=oenv.CAE_all[:len(bh)]; latent=oenv.CAE_mean_all[:len(bh)]
    # b_hat[k] should estimate CAE_all[k] (CAE of week k-1)
    m=np.isfinite(bh)&np.isfinite(cae)
    print(f"uid {uid}: corr(b_hat[k], CAE_all[k])={np.corrcoef(bh[m],cae[m])[0,1]:.3f}  corr(b_hat, latent)={np.corrcoef(bh[m],latent[m])[0,1]:.3f}  mean|b_hat-CAE|={np.abs(bh[m]-cae[m]).mean():.3f}  sd(CAE)={cae[m].std():.3f} sd(bhat)={bh[m].std():.3f}  J_w mean={np.nanmean(oenv.wp_all):.2f}")
    print("   b_hat[:8]", np.round(bh[:8],2)); print("   CAE   [:8]", np.round(cae[:8],2)); print("   latent[:8]", np.round(latent[:8],2))
    print("   sigma2_rl path (k, s2):", [(k,round(s,2)) for k,s in S2[::5]])
    print("   |target| mean, ens-sd of targets, Phi shape at k=5,15,35:", [(k,round(a,2),round(b,2),sh) for k,a,b,sh in TG if k in (5,15,35)])
    b=r['betas']; print("   betas shape", b.shape)
