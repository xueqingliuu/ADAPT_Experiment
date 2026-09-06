"""Probe RLSVI internals on a few users: does the posterior advantage track the env's true myopic CATE?"""
import os, sys, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))); os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import experiment as ex, algorithm_helpers as ah
from agents.micro_query import MicroQueryAgent
from ewm_utils import gamma_from_n
PD="env_para_vanilla" if len(sys.argv)<2 else sys.argv[1]
OUT=os.path.expanduser(sys.argv[2]) if len(sys.argv)>2 else os.path.expanduser("~/diag/rl_probe_vanilla.npz")
def ewma_w(n):
    g=gamma_from_n(n); w=g**np.arange(n-1,-1,-1); return w/w.sum()
WF=ewma_w(14); WA=ewma_w(7)
ENV_LOG=[]; ACT_LOG=[]
_orig_step=ex.OnlineEnv.step_action
def step_action(self,k,d,t,action,I_w):
    s=self.s; env=self.env
    s["decisionTimeSlot"]=float(t); s["stepCountLast7DaysEma"]=self._stepCountLast7DaysEma_by_slot[int(t)]
    self._generate_prior2hour_for_slot(k,d,t,self._day_idx(k,d),self._step_idx(k,d,t))
    df=env.gen_fourSC_mean(s,1.0)-env.gen_fourSC_mean(s,0.0)
    da=(env.gen_antic_mean(s,1.0,0.0) if t==0 else env.gen_antic_mean(s,0.0,1.0))-env.gen_antic_mean(s,0.0,0.0)
    th=env.cfg.theta_CAE; dl=th[3]*WF[d*2+t]*df+th[4]*WA[d]*da
    ENV_LOG.append([k,d,t,action,dl,s["activitySuggestionsSentLast7Days"],s["caeAverageLastWeek"]])
    return _orig_step(self,k,d,t,action,I_w)
ex.OnlineEnv.step_action=step_action
_orig_act=MicroQueryAgent.act
def act(self,k,d,t,state):
    A,pi=_orig_act(self,k,d,t,state)
    if k==0: ACT_LOG.append([k,d,t,np.nan,np.nan,pi,self.b_hat_hist[k],np.nan]); return A,pi
    betas=self._current_betas_rest if d>=1 else self._current_betas_day1
    phi1=ah.build_phi_action(self.b_hat_hist[k],self.b_tilde_hist[k],state,d,t,1)
    phi0=ah.build_phi_action(self.b_hat_hist[k],self.b_tilde_hist[k],state,d,t,0)
    adv=np.stack(betas)@(phi1-phi0)
    ACT_LOG.append([k,d,t,adv.mean(),adv.std(),pi,self.b_hat_hist[k],self.sigma2_rl])
    return A,pi
MicroQueryAgent.act=act
UIDS=[106,239,339,13,269,22]; SEEDS=[0,1]
res={}
for uid in UIDS:
    for sd in SEEDS:
        for algo in ["v1","v11"]:
            ENV_LOG.clear(); ACT_LOG.clear()
            seed=ex._episode_seed(sd, UIDS.index(uid))
            t0=time.time()
            if algo=="v1": r,oenv=ex.run_micro_query(uid,seed=seed,gamma_bar=0.9,params_dir=PD)
            else: r,oenv=ex.run_micro_query_adv_m(uid,seed=seed,gamma_bar=0.9,params_dir=PD)
            E=np.array(ENV_LOG); Aa=np.array(ACT_LOG)
            assert len(E)==len(Aa) and np.allclose(E[:,:3],Aa[:,:3])
            res[f"{algo}_{uid}_{sd}"]=np.column_stack([E,Aa[:,3:], oenv.CAE_all[E[:,0].astype(int)+1], oenv.CAE_mean_all[E[:,0].astype(int)+1]])
            m=E[:,0]>=1
            c1=np.corrcoef(Aa[m,3],E[m,4])[0,1]; c2=np.corrcoef(Aa[m,5],E[m,4])[0,1]
            print(f"{algo} uid={uid} seed={sd} {time.time()-t0:.0f}s  corr(meanAdv,Δ)={c1:+.3f} corr(π,Δ)={c2:+.3f}  |meanAdv| median={np.median(np.abs(Aa[m,3])):.3e} advSD median={np.median(Aa[m,4]):.3e}  Δ sd={E[m,4].std():.2e}  π mean={Aa[m,5].mean():.3f} sd={Aa[m,5].std():.3f} sigma2_rl(last)={Aa[-1,7]:.3f}", flush=True)
            np.savez(OUT, **res)
