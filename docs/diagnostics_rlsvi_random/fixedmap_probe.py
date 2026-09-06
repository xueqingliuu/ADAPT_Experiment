import os, sys, numpy as np, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))); os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import experiment as ex
from vani_env import denormalize_CAE
PD="env_para_ste0.8"; MODE=sys.argv[1]; UIDS=[int(u) for u in sys.argv[2].split(",")]; SD=int(sys.argv[3]) if len(sys.argv)>3 else 0; SRC=sys.argv[4] if len(sys.argv)>4 else "ridge"
T=np.load("docs/diagnostics_rlsvi_random/proximal_ste08.npz",allow_pickle=True); truth=dict(zip(T['uids'].tolist(),T['truth_c'].tolist()))
for uid in UIDS:
    t0=time.time(); ex._ensure_priors_configured(PD); ex.configure_leave_one_out_priors(uid, params_dir=PD)
    seed=ex._episode_seed(SD,0)
    r,oenv=ex.run_micro_query_reward_design(uid, seed=seed, reward_design="v4", gamma_bar=0.9, params_dir=PD, stage2_mode=MODE, stage1_source=SRC)
    rr,orr=ex.run_random_send(uid, seed=seed, params_dir=PD)
    pr=r["timing_probe"]; pi=np.asarray(r["pi_A"])[1:]
    cae=denormalize_CAE(np.asarray(oenv.CAE_all),params_dir=PD)[1:]; cr=denormalize_CAE(np.asarray(orr.CAE_all),params_dir=PD)[1:]
    g=pr["stage1_gamma_pm_minus_am"]; so=pr["stage1_sign_ok"]; med=list(pr["stage1_mediators"])
    print(f"{MODE}/{SRC} uid {uid} truth(PM−AM)={-truth[uid]:+.4f} [{time.time()-t0:.0f}s] dCAE vs random={np.nanmean(cae-cr):+.4f} | π mean={pi.mean():.3f} sd={pi.std():.3f} PM−AM={np.mean(pi[:,:,1]-pi[:,:,0]):+.3f} | σ²_Q wk35={pr['sigma2_q'][35]:.3f}")
    print(f"    Stage1 γ(PM−AM) wk35: "+" ".join(f"{m}={g[35,i]:+.4f}" for i,m in enumerate(med))+f" | r_gap wk12/35={pr['stage2_r_pm_minus_am'][12]:+.5f}/{pr['stage2_r_pm_minus_am'][35]:+.5f} (cae {pr['stage2_r_gap_cae'][35]:+.5f}, shaping {pr['stage2_r_gap_shaping'][35]:+.5f}) | π_gap wk35={pr['pi_pm_minus_am'][35]:+.3f}", flush=True)
