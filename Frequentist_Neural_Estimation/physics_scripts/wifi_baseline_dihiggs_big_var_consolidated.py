import torch
import numpy as np
import sys
from scipy.optimize import minimize
import torch.nn as nn
import re
import ast
import glob

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def ensemble_ratios(x, ensemble, wifi_weights):
    """Average of exp(logits) across ensemble members. Returns shape (n_samples,)"""
    all_ratios = []
    for model in ensemble:
        model.eval()
        with torch.no_grad():
            llrs = model(x).double()
            ratios = llrs
            all_ratios.append(ratios)
    ones = torch.ones(all_ratios[0].shape[0], device=device).double()
    all_ratios.append(ones)
    return torch.exp(wifi_weights @ torch.stack(all_ratios, dim=0))


def extract_msds(filepath):
    with open(filepath, 'r') as file:
        content = file.read()
    match = re.search(r'Using MSDs: (\[[\d,\s]+\])', content)
    msds = ast.literal_eval(match.group(1)) if match else []
    return msds


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
        self.input_mean = 125.
        self.input_std = 13.

    def forward(self, x):
        x = (x - self.input_mean) / self.input_std
        return self.net(x).squeeze()


# === Job indexing ===
units = 64
msds_idx = int(sys.argv[1]) - 1
print("Using MSD Config", msds_idx, flush=True)

# === Paths ===
this_dir = './'
msd_data_dir = '../msd_data_dir/'

# === Load the MSD index used for training ===
msd_numbers = extract_msds(this_dir + f'outputs/train_dihiggs_msd_network_baseline_big_var_{msds_idx+1}.out')
print("Using MSDs:", msd_numbers, flush=True)
msd_idx = msd_numbers[0]  # Use the single MSD

# === Load ensemble ===
ensemble = []
model_list = glob.glob(this_dir + f'varying_msds_big_var/baseline_model_{msds_idx+1}_ens_*.pt')
for model_path in model_list:
    model_ens = BinaryClassifier(units=units).to(device)
    model_ens.load_state_dict(torch.load(model_path, map_location=device))
    model_ens.eval()
    ensemble.append(model_ens)
print(f"Loaded {len(ensemble)} ensemble members", flush=True)

wifi_weights = torch.load(this_dir + f'wifi_weights_big_var/wifi_weights_baseline_msd_{msds_idx+1}.pt',
                          map_location=device)
w_wifi_full = wifi_weights['weights'].double().to(device)  # (M+1, num_classes)

# === Load MSD data (for reference samples) ===
signal_msd = torch.tensor(np.loadtxt(msd_data_dir + f'signal_ssds_big_var/processed/dihiggs_ssd_big_var_{msd_idx}.dat')).float()
background_msd = torch.tensor(np.loadtxt(msd_data_dir + f'background_ssds_big_var/processed/bbbb_ssd_big_var_{msd_idx}.dat')).float()

# === Apply 110–140 GeV mass window cut to MSD data ===
sig_mask = (signal_msd[:, 0] >= 110) & (signal_msd[:, 0] <= 140) \
         & (signal_msd[:, 1] >= 110) & (signal_msd[:, 1] <= 140)
signal_msd_eff = sig_mask.float().mean().item()
signal_msd = signal_msd[sig_mask]

bkg_mask = (background_msd[:, 0] >= 110) & (background_msd[:, 0] <= 140) \
         & (background_msd[:, 1] >= 110) & (background_msd[:, 1] <= 140)
background_msd_eff = bkg_mask.float().mean().item()
background_msd = background_msd[bkg_mask]

efficiency_ratio = background_msd_eff / signal_msd_eff
print(f"MSD signal efficiency: {signal_msd_eff:.4f}, MSD background efficiency: {background_msd_eff:.4f}", flush=True)
print(f"Efficiency ratio (bkg/sig): {efficiency_ratio:.4f}", flush=True)
print(f"MSD signal: {signal_msd.shape[0]} events (after cut)", flush=True)
print(f"MSD background: {background_msd.shape[0]} events (after cut)", flush=True)

# === Experiment parameters ===
TRUE_MU = 0.1
TRUE_SIGNAL_FRAC = TRUE_MU / (1 + TRUE_MU)

Nb_obs = 50000
Ns_obs = int(TRUE_MU * Nb_obs)
n_ref = Nb_obs + Ns_obs

print(f"TRUE μ = {TRUE_MU}", flush=True)
print(f"N_signal = {Ns_obs}, N_background = {Nb_obs}", flush=True)

td_data_dir = '../td_data_dir/'
td_signal = torch.tensor(np.loadtxt(td_data_dir + 'dihiggs_sd_processed.dat')).float()
td_background = torch.tensor(np.loadtxt(td_data_dir + 'bbbb_sd_combined_processed_feb_12.dat')).float()

# === Apply 110–140 GeV mass window cut to TD data ===
td_signal = td_signal[(td_signal[:, 0] >= 110) & (td_signal[:, 0] <= 140)
                      & (td_signal[:, 1] >= 110) & (td_signal[:, 1] <= 140)]
td_background = td_background[(td_background[:, 0] >= 110) & (td_background[:, 0] <= 140)
                              & (td_background[:, 1] >= 110) & (td_background[:, 1] <= 140)]
print(f"TD signal: {td_signal.shape[0]} events (after cut)", flush=True)
print(f"TD background: {td_background.shape[0]} events (after cut)", flush=True)

# === Collect results over 300 pseudoexperiments ===
z_profiles = []
f_preds = []
f_uncs = []

for job_idx in range(300):
    print(f"Job {job_idx+1}/300", flush=True)

    # Resample observed data
    perm_sig = torch.randperm(td_signal.shape[0])
    perm_bkg = torch.randperm(td_background.shape[0])
    obs_signal = td_signal[perm_sig[:Ns_obs]]
    obs_background = td_background[perm_bkg[:Nb_obs]]
    obs_data = torch.cat([obs_signal, obs_background], dim=0)
    obs_data = obs_data[torch.randperm(obs_data.shape[0])].to(device)

    # Resample reference data from MSD background
    perm_bkg_ref = torch.randperm(background_msd.shape[0])
    ref_data = background_msd[perm_bkg_ref[:n_ref]].to(device)

    # Compute ratios (corrected for mass window cut efficiencies)
    data_ratios = ensemble_ratios(obs_data, ensemble, w_wifi_full) * efficiency_ratio
    ref_ratios = ensemble_ratios(ref_data, ensemble, w_wifi_full) * efficiency_ratio

    n_data = len(data_ratios)

    # === MLC Loss Function (1 parameter: f_s) ===
    def mlc_loss(params):
        f_s = params[0]

        mixture_data = f_s * data_ratios + (1 - f_s)
        data_term = -mixture_data.log().sum()

        mixture_ref = f_s * ref_ratios + (1 - f_s)
        ref_term = (mixture_ref - 1).sum()

        return data_term + ref_term

    def loss_np(x):
        params = torch.tensor(x, dtype=torch.float64, device=device)
        return mlc_loss(params).item()

    def grad_np(x):
        params = torch.tensor(x, dtype=torch.float64, device=device)
        return torch.func.grad(mlc_loss)(params).cpu().numpy()

    def hess_np(x):
        params = torch.tensor(x, dtype=torch.float64, device=device)
        return torch.func.hessian(mlc_loss)(params).cpu().numpy()

    params_0 = np.array([0.5])

    res = minimize(
        loss_np,
        x0=params_0,
        method='trust-exact',
        jac=grad_np,
        hess=hess_np,
        options={'initial_trust_radius': 0.1, 'max_trust_radius': 1.0}
    )

    f_s_fit = res.x[0]

    # === Sandwich Estimator ===
    mixture_data_fit = f_s_fit * data_ratios + (1 - f_s_fit)

    grad_data = -(data_ratios - 1) / mixture_data_fit
    grad_ref = ref_ratios - 1

    A = ((data_ratios - 1)**2 / mixture_data_fit**2).sum()

    B_data = n_data * grad_data.var()
    B_ref = n_ref * grad_ref.var()
    B = B_data + B_ref

    try:
        var_f_s = B / (A**2)
        std_f_s = var_f_s.sqrt().item()

        # === Profiled z-score ===
        # No nuisance params: evaluate loss at true f_s directly (no inner optimization)
        params_fit = torch.tensor([f_s_fit], dtype=torch.float64, device=device)
        params_at_true = torch.tensor([TRUE_SIGNAL_FRAC], dtype=torch.float64, device=device)
        loss_at_fit = mlc_loss(params_fit).item()
        loss_at_true = mlc_loss(params_at_true).item()
        profile_stat = 2 * (loss_at_true - loss_at_fit)

        # Schur complement = A_p[0,0] (no nuisance block to eliminate)
        A_p = torch.func.hessian(mlc_loss)(params_fit)
        A_schur = A_p[0, 0]
        V_tt = var_f_s
        lambda_corr = (A_schur * V_tt).item()
        z_profile = np.sign(f_s_fit - TRUE_SIGNAL_FRAC) * np.sqrt(max(profile_stat / lambda_corr, 0))
    except Exception:
        std_f_s = np.nan
        z_profile = np.nan

    print(f"kappa estimate: {f_s_fit:.6f} +/- {std_f_s} | profile z: {z_profile}", flush=True)
    z_profiles.append(z_profile)
    f_preds.append(f_s_fit)
    f_uncs.append(std_f_s)

print("All jobs done. Saving results.", flush=True)
torch.save(torch.tensor(z_profiles), f'../results_dir/wifi_baseline_dihiggs_z_profile_{msds_idx+1}.pt')
torch.save(torch.tensor(f_preds), f'../results_dir/wifi_baseline_dihiggs_kappa_pred_{msds_idx+1}.pt')
torch.save(torch.tensor(f_uncs), f'../results_dir/wifi_baseline_dihiggs_kappa_unc_{msds_idx+1}.pt')
print("Results saved.", flush=True)
