import torch
import numpy as np
import torch.nn.functional as F
import argparse
from scipy.optimize import minimize
import torch.nn as nn
import re
import ast
import glob

parser = argparse.ArgumentParser()
parser.add_argument('job_idx', type=int)
parser.add_argument('--lam', type=float, default=1.0)
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
lam = torch.tensor(args.lam, dtype=torch.float64, device=device)
sig_dists = []
bkg_dists = []


def normalize_prob(x, epsilon=0.0001):
    x = x + epsilon
    x = np.asarray(x, dtype=np.float64)
    s = x.sum(dtype=np.float64)
    y = x / s
    i = y.argmax()                     # adjust the largest to keep all positive
    y[i] += 1.0 - y.sum(dtype=np.float64)
    return y


def flat_histogramdd(arr, bins):
    edges = [np.asarray(b) for b in bins]
    return np.histogramdd(arr, bins=edges)[0].ravel()


def flat_histogramdd_weighted(arr, bins, weights):
    edges = [np.asarray(b) for b in bins]
    return np.histogramdd(arr, bins=edges, weights=weights)[0].ravel()


def hellinger_distance(p, q):
    return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))


def ensemble_ratios_wifi(x, ensemble, w_sig, w_bkg):
    """
    Compute WiFi-weighted ensemble ratios.
    w_sig: (M+1, num_sig) including f_0 term
    w_bkg: (M+1, num_bkg) including f_0 term

    Returns: ratios with shape (n_samples, num_classes) where num_classes = num_sig + num_bkg
    """
    llrs_list = []
    for model in ensemble:
        model.eval()
        with torch.no_grad():
            llrs = model.llrs_all(x).double()  # (n_samples, num_classes)
            llrs = llrs.t()
            llrs_list.append(llrs)

    llrs_all = torch.stack(llrs_list, dim=0)

    n_samples = llrs_all.shape[2]
    num_classes = llrs_all.shape[1]
    f0 = torch.ones(1, num_classes, n_samples, dtype=torch.float64, device=device)
    llrs_with_f0 = torch.cat([llrs_all, f0], dim=0)  # (M+1, num_classes, n_samples)

    num_sig = w_sig.shape[1]
    llrs_sig = llrs_with_f0[:, :num_sig, :]  # (M+1, num_sig, n_samples)
    llrs_bkg = llrs_with_f0[:, num_sig:, :]  # (M+1, num_bkg, n_samples)

    log_r_sig = (llrs_sig * w_sig.unsqueeze(2)).sum(dim=0)  # (num_sig, n_samples)
    log_r_bkg = (llrs_bkg * w_bkg.unsqueeze(2)).sum(dim=0)  # (num_bkg, n_samples)

    log_ratios = torch.cat([log_r_sig, log_r_bkg], dim=0).t()  # (n_samples, num_classes)
    return log_ratios.exp()


def get_weights(params):
    w_sig_free = params[3:3+num_sig-1]
    w_sig_last = (1 - w_sig_free.sum()).unsqueeze(0)
    w_sig = torch.cat([w_sig_free, w_sig_last])

    w_bkg_free = params[3+num_sig-1:]
    w_bkg_last = (1 - w_bkg_free.sum()).unsqueeze(0)
    w_bkg = torch.cat([w_bkg_free, w_bkg_last])

    return w_sig, w_bkg


def extract_msds(filepath):
    with open(filepath, 'r') as file:
        content = file.read()

    match = re.search(r'Using MSDs: (\[[\d,\s]+\])', content)
    msds = ast.literal_eval(match.group(1)) if match else []
    return msds


class MultiClassifier(nn.Module):
    def __init__(self, n_classes=8, units=64):
        super().__init__()
        self.layer1 = nn.Linear(2, units)
        self.layer2 = nn.Linear(units, units)
        self.layer3 = nn.Linear(units, units)
        self.layer4 = nn.Linear(units, n_classes)

    def forward(self, x):
        x = (x - 125.) / 13.
        x = F.leaky_relu(self.layer1(x), negative_slope=0.2)
        x = F.leaky_relu(self.layer2(x), negative_slope=0.2)
        x = F.leaky_relu(self.layer3(x), negative_slope=0.2)
        x = self.layer4(x)
        return x

    def llrs_all(self, x):
        logits = self.forward(x)
        ref = logits.exp().mean(dim=1, keepdim=True).log()
        return logits - ref


units = 64
msds_idx = args.job_idx - 1
print("Using MSD Config", msds_idx, flush=True)

num_td = 8
num_sig = num_td
num_bkg = num_td
num_classes = 2 * num_td

this_dir = './'
msd_data_dir = '../msd_data_dir/'
msd_numbers = extract_msds(this_dir + f'outputs/train_dihiggs_msd_network_{num_td}_big_var_{msds_idx+1}.out')
print("Using MSDs:", msd_numbers, flush=True)

ensemble = []
model_list = glob.glob(this_dir + f'varying_msds_big_var/{num_td}_msd_model_{msds_idx+1}_ens_*.pt')
for i, model_path in enumerate(model_list):
    model_ens = MultiClassifier(n_classes=num_classes).to(device)
    model_ens.load_state_dict(torch.load(model_path, map_location=device))
    ensemble.append(model_ens)

# Load WiFi weights
wifi_weights = torch.load(this_dir + f'wifi_weights_big_var/wifi_weights_{num_td}_msd_{msds_idx+1}.pt',
                          map_location=device)
w_wifi_full = wifi_weights['weights'].double().to(device)  # (M+1, num_classes)
w_sig_wifi = w_wifi_full[:, :num_sig]  # (M+1, num_sig)
w_bkg_wifi = w_wifi_full[:, num_sig:]  # (M+1, num_bkg)
print(f"Loaded WiFi weights with {wifi_weights['M']} ensemble members", flush=True)
print(f"w_sig_wifi shape: {w_sig_wifi.shape}, w_bkg_wifi shape: {w_bkg_wifi.shape}", flush=True)

signal_msds = []
background_msds = []

for i, msd_idx in enumerate(msd_numbers):
    signal_msds.append(torch.tensor(np.loadtxt(msd_data_dir + f'signal_ssds_big_var/processed/dihiggs_ssd_big_var_{msd_idx}.dat')).float())
    background_msds.append(torch.tensor(np.loadtxt(msd_data_dir + f'background_ssds_big_var/processed/bbbb_ssd_big_var_{msd_idx}.dat')).float())

# === Compute per-class acceptance efficiencies from full pre-cut MSD data ===
# Efficiencies are computed before truncation for maximum statistical precision.
eff_sig = []
for s in signal_msds:
    mask = (s[:, 0] >= 110) & (s[:, 0] <= 140) & (s[:, 1] >= 110) & (s[:, 1] <= 140)
    eff_sig.append(mask.float().mean().item())

eff_bkg = []
for b in background_msds:
    mask = (b[:, 0] >= 110) & (b[:, 0] <= 140) & (b[:, 1] >= 110) & (b[:, 1] <= 140)
    eff_bkg.append(mask.float().mean().item())

eff_all = torch.tensor(eff_sig + eff_bkg, dtype=torch.float64, device=device)
eff_ref = eff_all.mean()
eff_scale = eff_ref / eff_all  # shape (num_classes,): multiply r_k by this
print(f"Signal MSD efficiencies: {eff_sig}", flush=True)
print(f"Background MSD efficiencies: {eff_bkg}", flush=True)
print(f"eff_scale: {eff_scale.tolist()}", flush=True)

n_samples = min([s.shape[0] for s in signal_msds] + [b.shape[0] for b in background_msds])
signal_msds = [s[:n_samples] for s in signal_msds]
background_msds = [b[:n_samples] for b in background_msds]

# Setup parameters
TRUE_MU = 0.1
TRUE_SIGNAL_FRAC = TRUE_MU / (1 + TRUE_MU)

Nb_obs = 50000
Ns_obs = int(TRUE_MU * Nb_obs)

n_ref_total = num_td * Nb_obs  # num_td times bigger than TD dataset

n_params = 3 + (num_sig - 1) + (num_bkg - 1)

td_data_dir = '../td_data_dir/'
td_signal = torch.tensor(np.loadtxt(td_data_dir + 'dihiggs_sd_processed.dat')).float()
td_background = torch.tensor(np.loadtxt(td_data_dir + 'bbbb_sd_combined_processed_feb_12.dat')).float()

# === Apply 110–140 GeV mass window cut to TD observed data ===
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
    obs_signal_p = td_signal[perm_sig[:Ns_obs]].to(device)
    obs_background_p = td_background[perm_bkg[:Nb_obs]].to(device)
    obs_signal_big_p = td_signal.to(device)
    obs_background_big_p = td_background.to(device)
    obs_data_p = torch.cat([obs_signal_p, obs_background_p], dim=0)
    obs_data_p = obs_data_p[torch.randperm(obs_data_p.shape[0])]
    mybinning = [np.linspace(110, 140, 10), np.linspace(110, 140, 10)]
    binned_signal_true = normalize_prob(flat_histogramdd(obs_signal_big_p.cpu().numpy(), mybinning))
    binned_background_true = normalize_prob(flat_histogramdd(obs_background_big_p.cpu().numpy(), mybinning))

    all_msd = torch.cat(signal_msds + background_msds, dim=0)

    ref_pre = all_msd[torch.randperm(len(all_msd))]
    ref_cut_mask = (ref_pre[:, 0] >= 110) & (ref_pre[:, 0] <= 140) \
        & (ref_pre[:, 1] >= 110) & (ref_pre[:, 1] <= 140)
    ref_data_p = ref_pre[ref_cut_mask][:n_ref_total].to(device)

    pen_pre = all_msd[torch.randperm(len(all_msd))]
    pen_cut_mask = (pen_pre[:, 0] >= 110) & (pen_pre[:, 0] <= 140) \
        & (pen_pre[:, 1] >= 110) & (pen_pre[:, 1] <= 140)
    pen_data_p = pen_pre[pen_cut_mask][:n_ref_total].to(device)

    if job_idx == 0:
        pool_size = ref_cut_mask.sum().item()
        print(f"Post-cut reference pool size: {pool_size}, fraction used: {n_ref_total / pool_size:.3f}", flush=True)

    # Compute ratios
    data_ratios_p = ensemble_ratios_wifi(obs_data_p, ensemble, w_sig_wifi, w_bkg_wifi)
    ref_ratios_p = ensemble_ratios_wifi(ref_data_p, ensemble, w_sig_wifi, w_bkg_wifi)
    pen_ratios_p = ensemble_ratios_wifi(pen_data_p, ensemble, w_sig_wifi, w_bkg_wifi)

    # Apply per-class efficiency correction: r_k_corrected = r_k * eff_ref / eff_k
    data_ratios_p = data_ratios_p * eff_scale.unsqueeze(0)
    ref_ratios_p = ref_ratios_p * eff_scale.unsqueeze(0)
    pen_ratios_p = pen_ratios_p * eff_scale.unsqueeze(0)

    n_data_p, n_ref_p, n_pen_p = len(data_ratios_p), len(ref_ratios_p), len(pen_ratios_p)
    c_ref = n_data_p / n_ref_p  # rescales reference term so both data and ref terms are on the same scale
    c_pen = n_data_p / n_pen_p

    def mlc_loss(params):
        f_s, c_s, c_b = params[0], params[1], params[2]
        w_sig, w_bkg = get_weights(params)

        r_sig_data = (data_ratios_p[:, :num_sig].log() * w_sig).sum(dim=1).exp()
        r_bkg_data = (data_ratios_p[:, num_sig:].log() * w_bkg).sum(dim=1).exp()
        ratio_data = (f_s * c_s * r_sig_data + (1 - f_s) * c_b * r_bkg_data).clamp(min=1e-8)

        r_sig_ref = (ref_ratios_p[:, :num_sig].log() * w_sig).sum(dim=1).exp()
        r_bkg_ref = (ref_ratios_p[:, num_sig:].log() * w_bkg).sum(dim=1).exp()
        ratio_ref = f_s * c_s * r_sig_ref + (1 - f_s) * c_b * r_bkg_ref

        r_sig_pen = (pen_ratios_p[:, :num_sig].log() * w_sig).sum(dim=1).exp()
        r_bkg_pen = (pen_ratios_p[:, num_sig:].log() * w_bkg).sum(dim=1).exp()
        pen_val = c_b * r_bkg_pen.mean() - c_s * r_sig_pen.mean()

        sig_pen = 0.5*lam*((w_sig - 1.0/num_sig)**2).sum()
        more_bkg_pen = 0.5*lam*((w_bkg - 1.0/num_bkg)**2).sum()

        return -ratio_data.log().sum() + c_ref * (ratio_ref - 1).sum() + 0.5 * lam * n_data_p * pen_val**2 + sig_pen + more_bkg_pen

    def signal(params, data_ratios):
        w_sig, w_bkg = get_weights(params)

        r_sig_data = (data_ratios[:, :num_sig].log() * w_sig).sum(dim=1).exp()

        return r_sig_data  # reweights reference to signal

    def background(params, data_ratios):
        w_sig, w_bkg = get_weights(params)

        r_bkg_data = (data_ratios[:, num_sig:].log() * w_bkg).sum(dim=1).exp()

        return r_bkg_data  # reweights reference to background

    def loss_np(x):
        params = torch.tensor(x, dtype=torch.float64, device=device)
        return mlc_loss(params).item()

    def grad_np(x):
        return torch.func.grad(mlc_loss)(torch.tensor(x, dtype=torch.float64, device=device)).cpu().numpy()

    def hess_np(x):
        return torch.func.hessian(mlc_loss)(torch.tensor(x, dtype=torch.float64, device=device)).cpu().numpy()

    def loss_profiled_np(nuisance_params, f_s_fixed):
        full_params = torch.zeros(n_params, dtype=torch.float64, device=device)
        full_params[0] = f_s_fixed
        full_params[1:] = torch.tensor(nuisance_params, dtype=torch.float64, device=device)
        return mlc_loss(full_params).item()

    def grad_profiled_np(nuisance_params, f_s_fixed):
        full_params = torch.zeros(n_params, dtype=torch.float64, device=device)
        full_params[0] = f_s_fixed
        full_params[1:] = torch.tensor(nuisance_params, dtype=torch.float64, device=device)
        return torch.func.grad(mlc_loss)(full_params)[1:].cpu().numpy()

    def hess_profiled_np(nuisance_params, f_s_fixed):
        full_params = torch.zeros(n_params, dtype=torch.float64, device=device)
        full_params[0] = f_s_fixed
        full_params[1:] = torch.tensor(nuisance_params, dtype=torch.float64, device=device)
        return torch.func.hessian(mlc_loss)(full_params)[1:, 1:].cpu().numpy()
    params_0 = np.zeros(n_params)
    params_0[0] = 0.1
    params_0[1] = 1.0
    params_0[2] = 1.0
    for i in range(3, 3 + num_sig - 1):
        params_0[i] = 1.0 / num_sig
    for i in range(3 + num_sig - 1, n_params):
        params_0[i] = 1.0 / num_bkg

    res_profiled = minimize(
            lambda x: loss_profiled_np(x, TRUE_SIGNAL_FRAC),
            x0=params_0[1:],
            method='trust-exact',
            jac=lambda x: grad_profiled_np(x, TRUE_SIGNAL_FRAC),
            hess=lambda x: hess_profiled_np(x, TRUE_SIGNAL_FRAC),
            options={'initial_trust_radius': 0.01, 'max_trust_radius': 0.1}
        )
    params_0 = np.concatenate([np.array([0.1]), res_profiled.x])
    res = minimize(loss_np, x0=params_0, method='trust-exact', jac=grad_np, hess=hess_np, options={'initial_trust_radius': 0.1, 'max_trust_radius': 10.0})

    ref_signal_weights = signal(torch.tensor(res.x, dtype=torch.float64, device=device), ref_ratios_p).cpu().numpy()
    ref_background_weights = background(torch.tensor(res.x, dtype=torch.float64, device=device), ref_ratios_p).cpu().numpy()
    binned_ref_signal_weighted = normalize_prob(flat_histogramdd_weighted(ref_data_p.cpu().numpy(), mybinning, ref_signal_weights))
    binned_ref_background_weighted = normalize_prob(flat_histogramdd_weighted(ref_data_p.cpu().numpy(), mybinning, ref_background_weights))
    sig_dist = hellinger_distance(binned_ref_signal_weighted, binned_signal_true)
    bkg_dist = hellinger_distance(binned_ref_background_weighted, binned_background_true)
    sig_dists.append(sig_dist)
    bkg_dists.append(bkg_dist)
    if job_idx % 100 == 0:
        print(f"Hellinger distances to truth: signal {sig_dist:.4f}, background {bkg_dist:.4f}", flush=True)

    f_s_fit, c_s_fit, c_b_fit = res.x[0], res.x[1], res.x[2]
    params_fit = torch.tensor(res.x, dtype=torch.float64, device=device)
    w_sig_fit, w_bkg_fit = get_weights(params_fit)

    A_p = torch.func.hessian(mlc_loss)(params_fit)
    print("Hessian eigenvalues are", torch.linalg.eigvalsh(A_p), flush=True)
    r_sig_data = (data_ratios_p[:, :num_sig].log() * w_sig_fit).sum(dim=1).exp()
    r_bkg_data = (data_ratios_p[:, num_sig:].log() * w_bkg_fit).sum(dim=1).exp()
    r_sig_ref = (ref_ratios_p[:, :num_sig].log() * w_sig_fit).sum(dim=1).exp()
    r_bkg_ref = (ref_ratios_p[:, num_sig:].log() * w_bkg_fit).sum(dim=1).exp()
    r_sig_pen = (pen_ratios_p[:, :num_sig].log() * w_sig_fit).sum(dim=1).exp()
    r_bkg_pen = (pen_ratios_p[:, num_sig:].log() * w_bkg_fit).sum(dim=1).exp()

    ratio_data_fit = f_s_fit * c_s_fit * r_sig_data + (1 - f_s_fit) * c_b_fit * r_bkg_data

    def compute_gradients(ratios, r_sig, r_bkg):
        grads = [
            c_s_fit * r_sig - c_b_fit * r_bkg,
            f_s_fit * r_sig,
            (1 - f_s_fit) * r_bkg,
        ]
        for i in range(num_sig - 1):
            grads.append(f_s_fit * c_s_fit * r_sig * (ratios[:, i].log() - ratios[:, num_sig - 1].log()))
        for i in range(num_bkg - 1):
            grads.append((1 - f_s_fit) * c_b_fit * r_bkg * (ratios[:, num_sig + i].log() - ratios[:, num_sig + num_bkg - 1].log()))
        return torch.stack(grads, dim=1)

    dratio_data = compute_gradients(data_ratios_p, r_sig_data, r_bkg_data)
    dratio_ref = compute_gradients(ref_ratios_p, r_sig_ref, r_bkg_ref)

    grad_data = -dratio_data / ratio_data_fit.unsqueeze(1)
    grad_ref = dratio_ref

    grad_pen = torch.zeros(n_pen_p, n_params, dtype=torch.float64, device=device)
    grad_pen[:, 1] = -r_sig_pen  # c_s
    grad_pen[:, 2] = r_bkg_pen   # c_b
    for i in range(num_sig - 1):
        grad_pen[:, 3 + i] = -c_s_fit * r_sig_pen * (pen_ratios_p[:, i].log() - pen_ratios_p[:, num_sig - 1].log())
    for i in range(num_bkg - 1):
        grad_pen[:, 3 + num_sig - 1 + i] = c_b_fit * r_bkg_pen * (pen_ratios_p[:, num_sig + i].log() - pen_ratios_p[:, num_sig + num_bkg - 1].log())

    g_pen = c_b_fit * r_bkg_pen - c_s_fit * r_sig_pen
    var_g = (g_pen**2).mean() - g_pen.mean()**2

    B_data = n_data_p * ((grad_data.T @ grad_data) / n_data_p - torch.outer(grad_data.mean(0), grad_data.mean(0)))
    B_ref = c_ref**2 * n_ref_p * ((grad_ref.T @ grad_ref) / n_ref_p - torch.outer(grad_ref.mean(0), grad_ref.mean(0)))
    B_pen = lam**2 * c_pen**2 * n_pen_p * var_g * torch.outer(grad_pen.mean(0), grad_pen.mean(0))
    B_p = B_data + B_ref + B_pen

    try:
        A_inv = torch.linalg.inv(A_p)
        cov_theta = A_inv @ B_p @ A_inv
        if job_idx % 100 == 0:
            cov_data = (A_inv @ B_data @ A_inv)[0, 0].item()
            cov_ref = (A_inv @ B_ref  @ A_inv)[0, 0].item()
            cov_pen = (A_inv @ B_pen  @ A_inv)[0, 0].item()
            vfs = cov_theta[0, 0].item()
            print(f"  var(f_s) relative contributions: B_data={cov_data/vfs:.3f}, B_ref={cov_ref/vfs:.3f}, B_pen={cov_pen/vfs:.8f} (var_fs={vfs:.4e})", flush=True)
        std_f_s = cov_theta[0, 0].sqrt().item()

        loss_at_fit = mlc_loss(params_fit).item()
        params_at_true = torch.zeros(n_params, dtype=torch.float64, device=device)
        params_at_true[0] = TRUE_SIGNAL_FRAC
        params_at_true[1:] = torch.tensor(res_profiled.x, dtype=torch.float64, device=device)
        loss_at_true = mlc_loss(params_at_true).item()
        profile_stat = 2 * (loss_at_true - loss_at_fit)

        A_nn_inv = torch.linalg.inv(A_p[1:, 1:])
        A_schur = A_p[0, 0] - A_p[0, 1:] @ A_nn_inv @ A_p[1:, 0]

        V_tt = cov_theta[0, 0]
        lambda_corr = (A_schur * V_tt).item()
        z_profile = np.sign(f_s_fit - TRUE_SIGNAL_FRAC) * np.sqrt(max(profile_stat / lambda_corr, 0))
        if profile_stat < 0:
            print("Profile stat is", profile_stat, flush=True)
            print("lambda_corr is", lambda_corr, flush=True)
            print(res, flush=True)
            print(res_profiled, flush=True)
    except Exception:
        std_f_s = np.nan
        z_profile = np.nan

    print(f"kappa estimate: {f_s_fit:.6f} +/- {std_f_s} | profile z: {z_profile}", flush=True)

    z_profiles.append(z_profile)
    f_preds.append(f_s_fit)
    f_uncs.append(std_f_s)

print("All jobs done. Saving results.", flush=True)
lam_tag = str(args.lam).replace('.', 'p')
torch.save(torch.tensor(z_profiles), f'../results_dir/wifi_exponential_param_dihiggs_{num_td}_lam{lam_tag}_z_profile_{msds_idx+1}.pt')
torch.save(torch.tensor(f_preds), f'../results_dir/wifi_exponential_param_dihiggs_{num_td}_lam{lam_tag}_kappa_pred_{msds_idx+1}.pt')
torch.save(torch.tensor(f_uncs), f'../results_dir/wifi_exponential_param_dihiggs_{num_td}_lam{lam_tag}_kappa_unc_{msds_idx+1}.pt')
torch.save(torch.tensor(sig_dists), f'../results_dir/wifi_exponential_param_dihiggs_{num_td}_lam{lam_tag}_sig_dist_{msds_idx+1}.pt')
torch.save(torch.tensor(bkg_dists), f'../results_dir/wifi_exponential_param_dihiggs_{num_td}_lam{lam_tag}_bkg_dist_{msds_idx+1}.pt')
print("Results saved.", flush=True)
