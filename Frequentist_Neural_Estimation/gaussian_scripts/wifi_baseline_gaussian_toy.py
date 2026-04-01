import torch
import numpy as np
import sys
from scipy.optimize import minimize
import torch.nn as nn
import glob

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CUT = 2


def ensemble_ratios(x, ensemble, wifi_weights):
    """Compute exp(w @ logits) density ratio. Returns shape (n_samples,)"""
    all_llrs = []
    for model in ensemble:
        model.eval()
        with torch.no_grad():
            llrs = model(x).double()
            all_llrs.append(llrs)
    ones = torch.ones(all_llrs[0].shape[0], device=device).double()
    all_llrs.append(ones)
    return (wifi_weights @ torch.stack(all_llrs, dim=0)).exp()


class BinaryClassifier(nn.Module):
    """Binary classifier for signal vs background."""
    def __init__(self, units=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, units),
            nn.LeakyReLU(0.2),
            nn.Linear(units, units),
            nn.LeakyReLU(0.2),
            nn.Linear(units, units),
            nn.LeakyReLU(0.2),
            nn.Linear(units, 1)
        )
        self.input_mean = 0.
        self.input_std = 1.

    def forward(self, x):
        x = (x - self.input_mean) / self.input_std
        return self.net(x).squeeze()


# === Job indexing ===
units = 64
raw_job_idx = int(sys.argv[1])
msds_idx = raw_job_idx - 1
print("Using MSD Config", msds_idx, flush=True)

# === Paths ===
this_dir = './'

# === Load saved MSD parameters ===
msd_signal_means = torch.load(this_dir + f'varying_msds/signal_baseline_means_final_job_{msds_idx+1}.pt')
msd_signal_covs = torch.load(this_dir + f'varying_msds/signal_baseline_covs_final_job_{msds_idx+1}.pt')
msd_background_means = torch.load(this_dir + f'varying_msds/background_baseline_means_final_job_{msds_idx+1}.pt')
msd_background_covs = torch.load(this_dir + f'varying_msds/background_baseline_covs_final_job_{msds_idx+1}.pt')
print("Loaded MSD parameters", flush=True)

# === Load ensemble ===
ensemble = []
model_list = sorted(glob.glob(this_dir + f'varying_msds/baseline_model_final_{msds_idx+1}_ens_*.pt'))
for model_path in model_list:
    model_ens = BinaryClassifier(units=units).to(device)
    model_ens.load_state_dict(torch.load(model_path, map_location=device))
    model_ens.eval()
    ensemble.append(model_ens)
print(f"Loaded {len(ensemble)} ensemble members", flush=True)

wifi_weights = torch.load(this_dir + f'wifi_weights/wifi_weights_final_baseline_msd_{msds_idx+1}.pt',
                          map_location=device)
w_wifi_full = wifi_weights['weights'].double().to(device)  # (M+1,)

# === Experiment parameters ===
TRUE_MU = 0.1
TRUE_SIGNAL_FRAC = TRUE_MU / (1 + TRUE_MU)

Nb_obs = 50000
Ns_obs = int(TRUE_MU * Nb_obs)
n_ref = Nb_obs + Ns_obs

print(f"TRUE μ = {TRUE_MU}", flush=True)
print(f"N_signal = {Ns_obs}, N_background = {Nb_obs}", flush=True)

td_signal_mean = [0, 0]
td_signal_cov = [[1, 0], [0, 1]]
td_background_mean = [0, 1]
td_background_cov = [[3, 0.4], [0.4, 3]]


# === MLC Loss Function (1 parameter: f_s) ===
def mlc_loss(params):
    f_s = params[0]

    # Data term: -sum(log(f_s * r(x) + (1 - f_s)))
    mixture_data = f_s * data_ratios + (1 - f_s)
    data_term = -mixture_data.log().sum()

    # Reference term: sum(f_s * r(x) + (1 - f_s) - 1) = sum(f_s * (r(x) - 1))
    mixture_ref = f_s * ref_ratios + (1 - f_s)
    ref_term = (mixture_ref - 1).sum()

    return data_term + ref_term


# === Wrappers for scipy ===
def loss_np(x):
    params = torch.tensor(x, dtype=torch.float64, device=device)
    return mlc_loss(params).item()


def grad_np(x):
    params = torch.tensor(x, dtype=torch.float64, device=device)
    return torch.func.grad(mlc_loss)(params).cpu().numpy()


def hess_np(x):
    params = torch.tensor(x, dtype=torch.float64, device=device)
    return torch.func.hessian(mlc_loss)(params).cpu().numpy()


# === Optimize ===
params_0 = np.array([0.5])  # Initial guess for f_s


z_profiles = []
f_preds = []
f_uncs = []

for job_idx in range(300):
    print(f"\n=== Pseudoexperiment {job_idx+1}/300 ===", flush=True)

    try:
        # Overgenerate and apply feature cut [-CUT, CUT] on both dimensions
        Nb0 = 1000000
        signal_msd_raw = torch.tensor(np.random.multivariate_normal(
            msd_signal_means[0].numpy(), msd_signal_covs[0].numpy(), Nb0)).float()
        signal_msd = signal_msd_raw[(signal_msd_raw[:, 0] >= -CUT) & (signal_msd_raw[:, 0] <= CUT) & (signal_msd_raw[:, 1] >= -CUT) & (signal_msd_raw[:, 1] <= CUT)]
        signal_msd_eff = signal_msd.shape[0]/signal_msd_raw.shape[0]

        background_msd_raw = torch.tensor(np.random.multivariate_normal(
            msd_background_means[0].numpy(), msd_background_covs[0].numpy(), Nb0)).float()
        background_msd = background_msd_raw[(background_msd_raw[:, 0] >= -CUT) & (background_msd_raw[:, 0] <= CUT) & (background_msd_raw[:, 1] >= -CUT) & (background_msd_raw[:, 1] <= CUT)]
        background_msd_eff = background_msd.shape[0]/background_msd_raw.shape[0]
        efficiency_ratio = background_msd_eff/signal_msd_eff

        obs_signal_raw = torch.tensor(np.random.multivariate_normal(td_signal_mean, td_signal_cov, 10 * Ns_obs)).float()
        obs_signal_raw = obs_signal_raw[(obs_signal_raw[:, 0] >= -CUT) & (obs_signal_raw[:, 0] <= CUT) & (obs_signal_raw[:, 1] >= -CUT) & (obs_signal_raw[:, 1] <= CUT)]
        obs_signal = obs_signal_raw[:Ns_obs]

        obs_background_raw = torch.tensor(np.random.multivariate_normal(td_background_mean, td_background_cov, 10 * Nb_obs)).float()
        obs_background_raw = obs_background_raw[(obs_background_raw[:, 0] >= -CUT) & (obs_background_raw[:, 0] <= CUT) & (obs_background_raw[:, 1] >= -CUT) & (obs_background_raw[:, 1] <= CUT)]
        obs_background = obs_background_raw[:Nb_obs]

        obs_data = torch.cat([obs_signal, obs_background], dim=0)
        obs_data = obs_data[torch.randperm(obs_data.shape[0])].to(device)

        # === Reference background samples from MSD ===
        perm_bkg_ref = torch.randperm(background_msd.shape[0])
        ref_data = background_msd[perm_bkg_ref[:n_ref]].to(device)

        # === Compute ratios ===
        data_ratios = ensemble_ratios(obs_data, ensemble, w_wifi_full) * efficiency_ratio
        ref_ratios = ensemble_ratios(ref_data, ensemble, w_wifi_full) * efficiency_ratio

        n_data = len(data_ratios)

        res = minimize(
            loss_np,
            x0=params_0,
            method='trust-exact',
            jac=grad_np,
            hess=hess_np,
            options={'initial_trust_radius': 0.01, 'max_trust_radius': 0.05}
        )

        f_s_fit = res.x[0]

        # === Sandwich Estimator for Uncertainty ===
        # Compute mixture at fitted value
        mixture_data_fit = f_s_fit * data_ratios + (1 - f_s_fit)

        # Per-sample gradient contributions
        # d/d(f_s) of -log(f_s * r + (1-f_s)) = -(r - 1) / (f_s * r + (1-f_s))
        grad_data = -(data_ratios - 1) / mixture_data_fit  # shape (n_data,)
        # d/d(f_s) of (f_s * r + (1-f_s) - 1) = (r - 1)
        grad_ref = ref_ratios - 1  # shape (n_ref,)

        # Hessian A (second derivative of loss)
        # d²/d(f_s)² of -log(f_s * r + (1-f_s)) = (r - 1)² / (f_s * r + (1-f_s))²
        A = ((data_ratios - 1)**2 / mixture_data_fit**2).sum()

        # Score variance B (variance of gradient contributions, scaled by n)
        B_data = n_data * grad_data.var()
        B_ref = n_ref * grad_ref.var()
        B = B_data + B_ref

        # Sandwich formula: var(f_s) = B / A²
        try:
            var_f_s = B / (A**2)
            std_f_s = var_f_s.sqrt().item()

            # Profiled likelihood ratio test (no nuisance params in baseline)
            params_fit_t = torch.tensor([f_s_fit], dtype=torch.float64, device=device)
            params_true_t = torch.tensor([TRUE_SIGNAL_FRAC], dtype=torch.float64, device=device)
            loss_at_fit = mlc_loss(params_fit_t).item()
            loss_at_true = mlc_loss(params_true_t).item()
            profile_stat = 2 * (loss_at_true - loss_at_fit)

            # Correction: A_schur = A (no nuisance), V_tt = var_f_s (from sandwich)
            lambda_corr = (A * var_f_s).item()
            z_profile = np.sign(f_s_fit - TRUE_SIGNAL_FRAC) * np.sqrt(max(profile_stat / lambda_corr, 0))
        except Exception:
            std_f_s = np.nan
            z_profile = np.nan

        print(f"kappa estimate: {f_s_fit:.6f} +/- {std_f_s:.6f} | profile z: {z_profile:.6f}", flush=True)
        z_profiles.append(z_profile)
        f_preds.append(f_s_fit)
        f_uncs.append(std_f_s)

    except Exception as e:
        print(f"Pseudoexperiment {job_idx+1} failed: {e}", flush=True)
        z_profiles.append(np.nan)
        f_preds.append(np.nan)
        f_uncs.append(np.nan)

# Save all results for this MSD config in single files
torch.save(torch.tensor(z_profiles), this_dir + f'../results_dir/wifi_baseline_gaussian_toy_final_cut{CUT}_z_profile_{msds_idx+1}.pt')
torch.save(torch.tensor(f_preds), this_dir + f'../results_dir/wifi_baseline_gaussian_toy_final_cut{CUT}_kappa_pred_{msds_idx+1}.pt')
torch.save(torch.tensor(f_uncs), this_dir + f'../results_dir/wifi_baseline_gaussian_toy_final_cut{CUT}_kappa_unc_{msds_idx+1}.pt')
print(f"\nSaved combined results for MSD config {msds_idx+1}", flush=True)
