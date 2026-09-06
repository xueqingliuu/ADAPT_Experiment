"""Does the V4 posterior on A×slot_pm move off its LOO prior? Run rl_v4 (design v2) for a few users on ste0.8 with LOO priors and log the ensemble-mean β on the two timing columns each week, plus π_PM−π_AM and the env truth."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))); os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["USE_ESTIMATED_PRIORS"]="1"
import experiment as ex
from agents.micro_query_reward_design import MicroQueryRewardDesignAgent
PD="env_para_ste0.8"
LOG=[]
_orig=MicroQueryRewardDesignAgent.update_rlsvi
def upd(self,k):
    _orig(self,k)
    b=self.betas_store.get(k)
    if b is not None:
        B=np.stack(b); d1=self.daily_eta_store.get(k) or {}; sc=np.asarray(d1.get("SC",[np.nan])).ravel()[-1]; aa=np.asarray(d1.get("AA",[np.nan])).ravel()[-1]; LOG.append([k, B[:,-2].mean(), B[:,-1].mean(), B[:,-1].std(), float(self.mu_0_rl[-1]), float(np.sqrt(self.Sigma_0_rl[-1,-1])), float(self.sigma2_rl), sc, aa])
MicroQueryRewardDesignAgent.update_rlsvi=upd
T=np.load("docs/diagnostics_rlsvi_random/proximal_ste08.npz",allow_pickle=True); truth=dict(zip(T['uids'].tolist(),T['truth_c'].tolist()))
for uid in [239,106,22,75]:
    ex._ensure_priors_configured(PD); ex.configure_leave_one_out_priors(uid, params_dir=PD)
    LOG.clear()
    r,oenv=ex.run_micro_query_reward_design(uid, seed=ex._episode_seed(0,0), reward_design="v2", gamma_bar=0.9, params_dir=PD)
    L=np.array(LOG); pi=np.asarray(r["pi_A"]) if "pi_A" in r else None
    gap = np.nanmean(pi[1:,:,1]-pi[1:,:,0]) if pi is not None else np.nan
    print(f"uid {uid} truth(AM−PM)={truth[uid]:+.4f}  prior mu(A×slot_pm)={L[0,4]:+.3f} sd={L[0,5]:.3f} | posterior mean A×slot_pm wk5={L[4,2]:+.3f} wk12={L[11,2]:+.3f} wk35={L[-1,2]:+.3f} (ens sd {L[-1,3]:.3f}) | A×wkend wk35={L[-1,1]:+.3f} | sigma2_Q wk35={L[-1,6]:.3f} | Stage1b β_t(SC) wk12={L[11,7]:+.3f} wk35={L[-1,7]:+.3f} | Stage1 γa−γm(AA) wk35={L[-1,8]:+.4f} | π_PM−π_AM={gap:+.3f}", flush=True)
