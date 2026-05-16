# %%
import pandas as pd
# import numpy as np
from math import ceil

import matplotlib.pyplot as plt
import ssm
from ssm import LDS
import statsmodels.api as sm
from patsy import dmatrix
from pathlib import Path
import autograd.numpy as np  # Essential: ssm requires autograd.numpy
from autograd import hessian
from ssm.emissions import Emissions
from ssm.stats import bernoulli_logpdf
from ssm.preprocessing import interpolate_data, pca_with_imputation
from ssm.util import logistic, logit, ensure_args_are_lists

# Lower bound on Gaussian variances (cols 1–2) so exp(inv_eta) never underflows to 0.
_GAUSS_VAR_FLOOR = 1e-8

class MixedBernoulliGaussianEmissions(Emissions):
    """
    Emission mean depends on latent state only: E[y|x] = C x + d (no F u).
    Pass `inputs=u` to LDS.fit() as usual: those still drive **dynamics** (continuous
    latent AR process in ssm) but do not enter the observation likelihood here.
    """
    def __init__(self, N, K, D, M=0):
        super().__init__(N, K, D, M)
        self.Cs = np.random.randn(N, D)
        self.ds = np.random.randn(N)
        self.inv_etas = np.zeros(N)

    @property
    def params(self):
        return self.Cs, self.ds, self.inv_etas

    @params.setter
    def params(self, value):
        self.Cs, self.ds, self.inv_etas = value

    def invert(self, data, input=None, mask=None, tag=None):
        z_init = np.nanmean(data[:, 1:], axis=1, keepdims=True)
        return np.where(np.isnan(z_init), 0, z_init)


    def log_likelihoods(self, data, input, mask, tag, x):
        # 1. Get the linear prediction: Shape (T, 3)
        # Using np.dot(x, self.Cs.T) is safe for autograd
        prediction = np.dot(x, self.Cs.T) + self.ds
        
        # 2. Bernoulli — column 0 is week_present (binary). In this pipeline it is always observed,
        # so mask[:, 0] is all True and this multiply is a no-op; kept for ssm API consistency
        # and in case column 0 is ever missing.
        ll_b = bernoulli_logpdf(data[:, 0:1].astype(int), prediction[:, 0:1])
        ll_b = ll_b * mask[:, 0:1]
        
        # 3. Gaussian — Exp-tool-1 / Exp-tool-2 (cols 1–2); mask zeros missing survey weeks
        sigmas = np.maximum(np.exp(self.inv_etas[1:]), _GAUSS_VAR_FLOOR)
        ll_g = -0.5 * ((data[:, 1:] - prediction[:, 1:])**2 / sigmas + np.log(2 * np.pi * sigmas))
        ll_g = ll_g * mask[:, 1:]
        
        # Sum over emission dims; must match ssm shape (T, 1) for in-place add with dynamics log_likelihoods
        # (1D (T,) + (T,1) broadcasts to (T,T) and breaks laplace_em).
        ll = ll_b[:, 0] + ll_g[:, 0] + ll_g[:, 1]
        return ll[:, None]

    def neg_hessian_log_emissions_prob(self, data, input, mask, tag, x, Ez):
        """Return (T, D, D); base class squeeze breaks when D=1 and Laplace-EM needs 3D blocks."""
        T, D = x.shape
        obj = (
            lambda xt, datat, inputt, maskt: self.log_likelihoods(
                datat[None, :], inputt[None, :], maskt[None, :], tag, xt[None, :]
            )[0, 0]
        )
        hess_f = hessian(obj)
        blocks = []
        for xt, datat, inputt, maskt in zip(x, data, input, mask):
            h = np.asarray(hess_f(xt, datat, inputt, maskt)).reshape(D, D)
            blocks.append(h)
        return -np.stack(blocks, axis=0)

    def sample(self, z, input=None, tag=None):
        prediction = np.dot(z, self.Cs.T) + self.ds
        y_b = np.random.binomial(1, logistic(prediction[:, 0]))[:, None]
        sigmas = np.maximum(np.exp(self.inv_etas[1:]), _GAUSS_VAR_FLOOR)
        y_g = prediction[:, 1:] + np.sqrt(sigmas) * np.random.randn(len(z), 2)
        return np.column_stack([y_b, y_g])

# %%
# read data
PROJECT_ROOT = Path("/Users/xueqingliu/Harvard University Dropbox/Liu Xueqing/ADAPR-MRT-Testbed")
COMBINED_DIR = Path("/Users/xueqingliu/Harvard University Dropbox/Liu Xueqing/ADAPT_MRT/rawdata/_combined")
WORK_DIR = PROJECT_ROOT / "env_para_vanilla"
WORK_DIR.mkdir(parents=True, exist_ok=True)


# Aliases for later cells that use `folder` / `work_folder` (Path so `/` joins work)
folder = COMBINED_DIR
work_folder = Path(WORK_DIR)

df_fit = pd.read_csv(folder / 'df_fit.csv')

# %%
# user 141, 143 does not have week 0, change week 0 to week 1
# change all participants' who have more than 2 week 0 to week 1
for userid in df_fit['ParticipantIdentifier'].unique():
    vc = df_fit.loc[df_fit['ParticipantIdentifier'] == userid, 'week'].value_counts()
    if vc.get(0, 0) > 2:
        m = df_fit['ParticipantIdentifier'] == userid
        df_fit.loc[m, 'week'] = df_fit.loc[m, 'week'] + 1
# remove week 12
df_fit = df_fit[df_fit['week'] < 13]
# print(df_fit.week.unique())



# %%
# change it from 0-7 to 1-8
# df_fit['Exp-tool-1'] = df_fit['Exp-tool-1'] + 1
# df_fit['Exp-tool-2'] = df_fit['Exp-tool-2'] + 1

# Per week: sums over distinct days (daily_present; DayWearing duplicated across DecisionTime)
_weekly_daily = (
    df_fit.assign(_d=pd.to_datetime(df_fit['Date']))
    .groupby(['ParticipantIdentifier', 'week', '_d'], as_index=False)[['daily_present', 'morning_wearing', 'DailyPageviewCount']]
    .first()
    .groupby(['ParticipantIdentifier', 'week'])[['daily_present', 'morning_wearing', 'DailyPageviewCount']]
    .sum()
    .rename(columns={'daily_present': 'weekly_daily_present_sum', 'morning_wearing': 'weekly_daily_wearing_sum', 'DailyPageviewCount': 'weekly_daily_pageview_sum'})
    .reset_index()
)

df_fit = df_fit.merge(_weekly_daily, on=['ParticipantIdentifier', 'week'], how='left')

# min–max to [0, 1]: (x - min) / (max - min); constant column → 0
_pv = df_fit['weekly_daily_pageview_sum']
_lo, _hi = _pv.min(), _pv.max()
df_fit['weekly_daily_pageview_sum'] = (_pv - _lo) / (_hi - _lo) if _hi > _lo else 0.0


# %%
obs_dim = 3    # emission dimension
state_dim = 1   # latent state dimension
input_dim = 3    # input dimension
N_iters = 10

# Gaussian AR dynamics (`dynamics="gaussian"` = AutoRegressiveObservations).
#
# mu_init (K × D): mean of the predicted mean for the first `lags` rows in _compute_mus (anchoring
# the start of the latent trajectory).
#
# Sigmas_init (K × D × D): covariance used in log_likelihoods for the **initial segment only**:
#   - Code splits ll = ll_init(data[:lags]) with Sigmas_init, then ll_ar(data[lags:]) with Sigmas.
#   - With default lags=1, **only timestep t=0** uses Sigmas_init; t≥1 use process noise Sigmas.
#   - If lags>1, the block t=0..lags-1 is scored with Sigmas_init (still one shared matrix per k).
# EM/Laplace M-step updates Sigmas (transition noise) but not mu_init / Sigmas_init by default.
DYNAMICS_MU_INIT = np.ones((1, state_dim))
DYNAMICS_SIGMAS_INIT = 0.01 * np.eye(state_dim)[np.newaxis, :, :]  # (1, D, D), variance 0.01 on diag

# %%
# initialize parameters
np.random.seed(2026)
# input matrix u
weekly_daily_present = df_fit['weekly_daily_present_sum'] / 7
weekly_daily_wearing = df_fit['weekly_daily_wearing_sum'] / 7
weekly_daily_pageview = df_fit['weekly_daily_pageview_sum']

# fill nan with mean
# weekly_daily_present = weekly_daily_present.fillna(weekly_daily_present.mean())
# weekly_daily_wearing = weekly_daily_wearing.fillna(weekly_daily_wearing.mean())
# weekly_daily_pageview = weekly_daily_pageview.fillna(weekly_daily_pageview.mean())

# to weekly level
weekly_daily_present = weekly_daily_present.to_numpy().reshape(-1, 14)[:, 13]
weekly_daily_wearing = weekly_daily_wearing.to_numpy().reshape(-1, 14)[:, 13]
weekly_daily_pageview = weekly_daily_pageview.to_numpy().reshape(-1, 14)[:, 13]

u = np.stack([weekly_daily_present, weekly_daily_wearing, weekly_daily_pageview], axis=1)

# observation matrix y
weekly_present = df_fit['week_present']
weekly_exp1 = df_fit['Exp-tool-1_norm']
weekly_exp2 = df_fit['Exp-tool-2_norm']  

# fill nan with mean
# weekly_present = weekly_present.fillna(weekly_present.mean())
# weekly_exp1 = weekly_exp1.fillna(weekly_exp1.mean())
# weekly_exp2 = weekly_exp2.fillna(weekly_exp2.mean())

# to weekly level
weekly_present = weekly_present.to_numpy().reshape(-1, 14)[:, 13]
weekly_exp1 = weekly_exp1.to_numpy().reshape(-1, 14)[:, 13]
weekly_exp2 = weekly_exp2.to_numpy().reshape(-1, 14)[:, 13]

y = np.stack([weekly_present, weekly_exp1, weekly_exp2], axis=1)

mask = ~np.isnan(y)
y_filled = np.nan_to_num(y, nan=0.0)

mixed_emissions_init = MixedBernoulliGaussianEmissions(obs_dim, 0, state_dim, input_dim)

lds = ssm.LDS(
    N=obs_dim, D=state_dim, M=input_dim, 
    dynamics_kwargs={"l2_penalty_b": 1e8}, 
    transitions="inputdriven", 
    dynamics="gaussian", 
    emissions=mixed_emissions_init # <--- Custom class here
)
lds.dynamics.mu_init = DYNAMICS_MU_INIT
lds.dynamics.Sigmas_init = DYNAMICS_SIGMAS_INIT

lds_lps, lds_q = lds.fit(
    y_filled, 
    inputs=u, 
    masks=mask,
    method="laplace_em", 
    num_iters=N_iters, 
    initialize=True
)

plt.figure(figsize=(6, 4))
plt.plot(lds_lps, label="Laplace-EM")
plt.xlim(0, N_iters)
plt.xlabel("Iteration")
plt.ylabel("ELBO")
plt.title("Convergence of Laplace-EM Algorithm")
# plt.savefig(PROJECT_ROOT + 'LDS_initilization_EM_V'  + '.pdf')
plt.show()

# print(np.mean(lds_q.mean_continuous_states[0]))
# print(lds.params)
As_init, bs_init, Vs_init, sqrt_Sigmas_init = lds.dynamics.params
Cs_init, ds_init, inv_etas_init = lds.emissions.params


# %%
userid_all = df_fit['ParticipantIdentifier'].unique()
perceived_utility_by_user = {}
params_by_user = {}
for i, userid in enumerate(userid_all):
    dat_user = df_fit[df_fit['ParticipantIdentifier'] == userid].copy()
    dat_user = dat_user.sort_values(['Date', 'DecisionTime'], na_position='last').reset_index(drop=True)

    # Input matrix u
    weekly_daily_present = dat_user['weekly_daily_present_sum'] / 7
    weekly_daily_wearing = dat_user['weekly_daily_wearing_sum'] / 7
    weekly_daily_pageview = dat_user['weekly_daily_pageview_sum']

    # fill nan with mean
    # weekly_daily_present = weekly_daily_present.fillna(weekly_daily_present.mean())
    # weekly_daily_wearing = weekly_daily_wearing.fillna(weekly_daily_wearing.mean())
    # weekly_daily_pageview = weekly_daily_pageview.fillna(weekly_daily_pageview.mean())

    # to weekly level
    weekly_daily_present = weekly_daily_present.to_numpy().reshape(-1, 14)[:, 0]
    weekly_daily_wearing = weekly_daily_wearing.to_numpy().reshape(-1, 14)[:, 0]
    weekly_daily_pageview = weekly_daily_pageview.to_numpy().reshape(-1, 14)[:, 0]

    u = np.stack([weekly_daily_present, weekly_daily_wearing, weekly_daily_pageview], axis=1)

    # Observation matrix y
    weekly_present = dat_user['week_present']
    weekly_exp1 = dat_user['Exp-tool-1_norm']
    weekly_exp2 = dat_user['Exp-tool-2_norm']

    # fill nan with mean
    # weekly_present = weekly_present.fillna(weekly_present.mean())
    # weekly_exp1 = weekly_exp1.fillna(weekly_exp1.mean())
    # weekly_exp2 = weekly_exp2.fillna(weekly_exp2.mean())

    # to weekly level
    weekly_present = weekly_present.to_numpy().reshape(-1, 14)[:, 0]
    weekly_exp1 = weekly_exp1.to_numpy().reshape(-1, 14)[:, 0]
    weekly_exp2 = weekly_exp2.to_numpy().reshape(-1, 14)[:, 0]

    y = np.stack([weekly_present, weekly_exp1, weekly_exp2], axis=1)

    mask = ~np.isnan(y)

    y_filled = np.nan_to_num(y, nan=0.0)

    user_emissions = MixedBernoulliGaussianEmissions(obs_dim, 0, state_dim, input_dim)

    # use the ssm package to fit the data
    lds = ssm.LDS(
        N=obs_dim, D=state_dim, M=input_dim,
        transitions="inputdriven", dynamics="gaussian", 
        emissions=user_emissions # <--- Custom class here
    )

    lds.dynamics._As = As_init
    lds.dynamics.bs = bs_init
    lds.dynamics.Vs = Vs_init
    lds.dynamics._sqrt_Sigmas = sqrt_Sigmas_init
    lds.dynamics.mu_init = DYNAMICS_MU_INIT
    lds.dynamics.Sigmas_init = DYNAMICS_SIGMAS_INIT
    lds.emissions.Cs = Cs_init
    lds.emissions.ds = ds_init
    lds.emissions.inv_etas = inv_etas_init
    lds_lps, lds_q = lds.fit(
        y_filled, 
        inputs=u, 
        masks=mask,
        method="laplace_em", 
        num_iters=N_iters, 
        initialize=False
    )


    # perceived utility (one value per LDS timestep; map to df rows with np.repeat)
    perceived_utility = lds_q.mean_continuous_states[0].reshape(-1)
    perceived_utility_by_user[userid] = np.asarray(perceived_utility, dtype=float)

    As_u, bs_u, Vs_u, sqrt_S_u = lds.dynamics.params
    Cs_u, ds_u, inv_etas_u = lds.emissions.params
    params_by_user[userid] = {
        "As": np.asarray(As_u).ravel(),
        "bs": np.asarray(bs_u).ravel(),
        "Vs": np.asarray(Vs_u).ravel(),
        "sqrt_Sigmas": np.asarray(sqrt_S_u).ravel(),
        "Cs": np.asarray(Cs_u).ravel(),
        "ds": np.asarray(ds_u).ravel(),
        "inv_etas": np.asarray(inv_etas_u).ravel(),
    }
    print(perceived_utility)


# %%
# Plots: match model notation (3-month MRT slide):
#   E_w = a_0 + a_1 E_{w-1} + a_2 PV + a_3 FW + a_4 PJ + ε,  ε ~ N(0,σ^2)
#   J_w^week | E_w ~ Bernoulli,  logit p = b_0 + b_1 E_w
#   U_{w,1} = c_0 + c_1 E_w + v^{U_1},   U_{w,2} = d_0 + d_1 E_w + v^{U_2}
# Inputs u are stacked [PJ, FW, PV] -> Vs[0]=a_4 (PJ), Vs[1]=a_3 (FW), Vs[2]=a_2 (PV).
# One PNG per coefficient *family*: e.g. params_group_a.png has five subplots ($a_0\ldots a_4$).
_plot_dir = WORK_DIR / "plots"
_plot_dir.mkdir(parents=True, exist_ok=True)

_users = list(params_by_user.keys())
if len(_users) > 0:
    _n = len(_users)
    _n_cols = min(4, max(1, _n))
    _n_rows_ew = max(1, ceil(_n / _n_cols))
    _x = np.arange(len(_users))

    def _bar_users_one_series(_ax, ylabel, title, vals):
        _ax.bar(_x, vals, width=0.75)
        _ax.set_xticks(_x)
        _ax.set_xticklabels([str(u) for u in _users], rotation=45, ha="right")
        _ax.set_ylabel(ylabel)
        _ax.set_title(title, fontsize=10)

    # --- $E_w$ trajectories (one PNG) ---
    _fig_ew, _axes_ew = plt.subplots(_n_rows_ew, _n_cols, figsize=(3.6 * _n_cols, 2.3 * _n_rows_ew), squeeze=False)
    for _ax in np.ravel(_axes_ew):
        _ax.set_visible(False)
    for _idx, _uid in enumerate(_users):
        _r, _c = divmod(_idx, _n_cols)
        _ax = _axes_ew[_r][_c]
        _ax.set_visible(True)
        _pu = perceived_utility_by_user[_uid]
        _ax.plot(np.arange(len(_pu)), _pu, marker="o", ms=2, lw=1)
        _ax.set_title(f"Participant {_uid}", fontsize=9)
        _ax.set_xlabel(r"$w$", fontsize=8)
        _ax.set_ylabel(r"$E_w$", fontsize=8)
    _fig_ew.suptitle(r"Posterior mean $E_w$ by participant", fontsize=11)
    _fig_ew.tight_layout()
    _fig_ew.savefig(_plot_dir / "Ew_posterior_mean_by_user.png", dpi=150, bbox_inches="tight")
    plt.show()

    # --- All $a_i$: one PNG, five separate subplots ---
    _fig_a, _axes_a = plt.subplots(1, 5, figsize=(16, 3.2), squeeze=False)
    _a_specs = [
        (r"$a_0$", lambda _p: float(_p["bs"][0])),
        (r"$a_1$", lambda _p: float(_p["As"][0])),
        (r"$a_2\,(\mathrm{PV})$", lambda _p: float(_p["Vs"][2])),
        (r"$a_3\,(\mathrm{FW})$", lambda _p: float(_p["Vs"][1])),
        (r"$a_4\,(\mathrm{PJ})$", lambda _p: float(_p["Vs"][0])),
    ]
    for _k, (_tit, _fn) in enumerate(_a_specs):
        _vals = [_fn(params_by_user[_u]) for _u in _users]
        _bar_users_one_series(_axes_a[0][_k], "Estimate", _tit, _vals)
    _fig_a.suptitle(
        r"State dynamics: $E_w = a_0 + a_1 E_{w-1} + a_2 \mathrm{PV} + a_3 \mathrm{FW} + a_4 \mathrm{PJ} + \epsilon_w$",
        fontsize=11,
        y=1.08,
    )
    _fig_a.tight_layout()
    _fig_a.savefig(_plot_dir / "params_group_a.png", dpi=150, bbox_inches="tight")
    plt.show()

    # --- Innovation / $\sigma$ (one PNG, one subplot) ---
    _fig_s, _ax_s = plt.subplots(1, 1, figsize=(max(7.0, len(_users) * 0.35), 3.2))
    _vals_s = [float(params_by_user[_u]["sqrt_Sigmas"][0]) for _u in _users]
    _bar_users_one_series(
        _ax_s,
        "Estimate",
        r"$\mathrm{chol}(\Sigma)$ — $\epsilon_w \sim \mathcal{N}(0,\sigma^2)$",
        _vals_s,
    )
    _fig_s.tight_layout()
    _fig_s.savefig(_plot_dir / "params_group_sigma.png", dpi=150, bbox_inches="tight")
    plt.show()

    # --- $b_0$, $b_1$: one PNG, two subplots ---
    _fig_b, _axes_b = plt.subplots(1, 2, figsize=(10, 3.2), squeeze=False)
    _bar_users_one_series(
        _axes_b[0][0],
        "Estimate",
        r"$b_0$ — intercept in $\mathrm{logit}(p_w)= b_0 + b_1 E_w$",
        [float(params_by_user[_u]["ds"][0]) for _u in _users],
    )
    _bar_users_one_series(
        _axes_b[0][1],
        "Estimate",
        r"$b_1$ — coefficient on $E_w$ for $J_w^{\mathrm{week}}$",
        [float(params_by_user[_u]["Cs"][0]) for _u in _users],
    )
    _fig_b.suptitle(r"Bernoulli emission ($J_w^{\mathrm{week}}$)", fontsize=11, y=1.05)
    _fig_b.tight_layout()
    _fig_b.savefig(_plot_dir / "params_group_b.png", dpi=150, bbox_inches="tight")
    plt.show()

    # --- $c_0$, $c_1$: one PNG, two subplots ---
    _fig_c, _axes_c = plt.subplots(1, 2, figsize=(10, 3.2), squeeze=False)
    _bar_users_one_series(
        _axes_c[0][0],
        "Estimate",
        r"$c_0$ — $U_{w,1}$ (helpfulness) intercept",
        [float(params_by_user[_u]["ds"][1]) for _u in _users],
    )
    _bar_users_one_series(
        _axes_c[0][1],
        "Estimate",
        r"$c_1$ — coefficient on $E_w$ for $U_{w,1}$",
        [float(params_by_user[_u]["Cs"][1]) for _u in _users],
    )
    _fig_c.suptitle(r"Gaussian $U_{w,1}$ (helpfulness)", fontsize=11, y=1.05)
    _fig_c.tight_layout()
    _fig_c.savefig(_plot_dir / "params_group_c.png", dpi=150, bbox_inches="tight")
    plt.show()

    # --- $d_0$, $d_1$: one PNG, two subplots ---
    _fig_d, _axes_d = plt.subplots(1, 2, figsize=(10, 3.2), squeeze=False)
    _bar_users_one_series(
        _axes_d[0][0],
        "Estimate",
        r"$d_0$ — $U_{w,2}$ (pleasantness) intercept",
        [float(params_by_user[_u]["ds"][2]) for _u in _users],
    )
    _bar_users_one_series(
        _axes_d[0][1],
        "Estimate",
        r"$d_1$ — coefficient on $E_w$ for $U_{w,2}$",
        [float(params_by_user[_u]["Cs"][2]) for _u in _users],
    )
    _fig_d.suptitle(r"Gaussian $U_{w,2}$ (pleasantness)", fontsize=11, y=1.05)
    _fig_d.tight_layout()
    _fig_d.savefig(_plot_dir / "params_group_d.png", dpi=150, bbox_inches="tight")
    plt.show()

    # --- $\log(\sigma^{U_i})^2$: one PNG, two subplots ---
    _fig_n, _axes_n = plt.subplots(1, 2, figsize=(10, 3.2), squeeze=False)
    _bar_users_one_series(
        _axes_n[0][0],
        "Estimate",
        r"$\log(\sigma^{U_1})^2$ — noise for $U_{w,1}$",
        [float(params_by_user[_u]["inv_etas"][1]) for _u in _users],
    )
    _bar_users_one_series(
        _axes_n[0][1],
        "Estimate",
        r"$\log(\sigma^{U_2})^2$ — noise for $U_{w,2}$",
        [float(params_by_user[_u]["inv_etas"][2]) for _u in _users],
    )
    _fig_n.suptitle(r"Gaussian emission noise", fontsize=11, y=1.05)
    _fig_n.tight_layout()
    _fig_n.savefig(_plot_dir / "params_group_log_sigma2.png", dpi=150, bbox_inches="tight")
    plt.show()


# %%

# %%
# Lag-1 within each participant (global [:-1] misaligns rows and crosses participants)

# add perceived utility for each user (from LDS cell: perceived_utility_by_user)
for userid in df_fit['ParticipantIdentifier'].unique():
    pu = perceived_utility_by_user[userid]
    m = df_fit['ParticipantIdentifier'] == userid
    n = int(m.sum())
    reps = n // len(pu)
    if n % len(pu) != 0:
        raise ValueError(
            f"Participant {userid}: {n} rows not divisible by LDS length {len(pu)}"
        )
    df_fit.loc[m, 'perceived_utility'] = np.repeat(pu, reps)

# add last week's perceived utility
df_fit['perceived_utility_lastweek'] = df_fit['perceived_utility'].shift(1)

# remove week 0 and week 1 data
df_fit = df_fit[df_fit['week'] > 1]

# turn week 2 into week 1
df_fit['week'] = df_fit['week'] - 1


df_fit.to_csv(folder / "df_fit_11week.csv")


