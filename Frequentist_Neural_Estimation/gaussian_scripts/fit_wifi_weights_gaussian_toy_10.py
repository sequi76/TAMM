import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import glob
import sys
from scipy.optimize import minimize
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class MultiClassifier(nn.Module):
    def __init__(self, n_classes=8, units=64):
        super().__init__()
        self.layer1 = nn.Linear(2, units)
        self.layer2 = nn.Linear(units, units)
        self.layer3 = nn.Linear(units, units)
        self.layer4 = nn.Linear(units, n_classes)

    def forward(self, x):
        x = (x - 0.) / 1.
        x = F.leaky_relu(self.layer1(x), negative_slope=0.2)
        x = F.leaky_relu(self.layer2(x), negative_slope=0.2)
        x = F.leaky_relu(self.layer3(x), negative_slope=0.2)
        x = self.layer4(x)
        return x

    def llrs_all(self, x):
        logits = self.forward(x)
        ref = logits.exp().mean(dim=1, keepdim=True).log()
        return logits - ref


def matrix_from_params(means, covs, n_samples):
    msd_matrix = []
    for i in range(len(means)):
        mean = means[i]
        cov = covs[i]
        msd_matrix.append(torch.tensor(np.random.multivariate_normal(mean.numpy(), cov.numpy(), n_samples)).float())
    return msd_matrix


# Parse arguments
msds_idx = int(sys.argv[1]) - 1 if len(sys.argv) > 1 else 0
print(f"Fitting wifi weights for MSD-to-reference density ratios (MSD config {msds_idx})", flush=True)

num_td = 10
num_classes = 2 * num_td  # One class per MSD (both signal and background MSDs)

this_dir = './'

Nb0 = 1000000

# Load saved MSD parameters (written by training script)
msd_signal_means = torch.load(this_dir + f'varying_msds/signal_{num_td}_msd_means_final_job_{msds_idx+1}.pt')
msd_signal_covs = torch.load(this_dir + f'varying_msds/signal_{num_td}_msd_covs_final_job_{msds_idx+1}.pt')
msd_background_means = torch.load(this_dir + f'varying_msds/background_{num_td}_msd_means_final_job_{msds_idx+1}.pt')
msd_background_covs = torch.load(this_dir + f'varying_msds/background_{num_td}_msd_covs_final_job_{msds_idx+1}.pt')
print("Loaded SMD parameters", flush=True)

# Load ensemble
ensemble = []
model_list = sorted(glob.glob(this_dir + f'varying_msds/{num_td}_msd_model_final_{msds_idx+1}_ens_*.pt'))
for model_path in model_list:
    model_ens = MultiClassifier(n_classes=num_classes).to(device)
    model_ens.load_state_dict(torch.load(model_path, map_location=device))
    ensemble.append(model_ens)

M = len(ensemble)
print(f"Loaded ensemble with {M} members", flush=True)

# Generate MSD data from parameters
signal_msds = matrix_from_params(msd_signal_means, msd_signal_covs, Nb0)
background_msds = matrix_from_params(msd_background_means, msd_background_covs, Nb0)

# Use same number of samples from each MSD
n_samples = min([s.shape[0] for s in signal_msds] + [b.shape[0] for b in background_msds])

signal_fit = [s[:n_samples] for s in signal_msds]
background_fit = [b[:n_samples] for b in background_msds]

# Keep individual MSD data separate for density ratio learning
signal_fit_data_list = [s.to(device) for s in signal_fit]
background_fit_data_list = [b.to(device) for b in background_fit]

# Create reference data by pooling ALL MSDs (signal + background)
reference_data = torch.cat(signal_fit + background_fit, dim=0).to(device)
# Subsample reference data to cap memory usage
max_ref_samples = 2000000  # or even 1M — tune as needed
if reference_data.shape[0] > max_ref_samples:
    idx = torch.randperm(reference_data.shape[0])[:max_ref_samples]
    reference_data = reference_data[idx]
    print(f"Subsampled reference data to {max_ref_samples} samples", flush=True)

print(f"Number of signal MSDs: {len(signal_fit_data_list)}", flush=True)
print(f"Number of background MSDs: {len(background_fit_data_list)}", flush=True)
print(f"Samples per MSD: {n_samples}", flush=True)
print(f"Reference data shape (all MSDs pooled): {reference_data.shape}", flush=True)

# Compute LLRs for all ensemble members
# For each MSD, compute log r(x) = log p_MSD(x) / p_reference(x)
# Shape: (M, num_msds, n_samples)
llrs_msd_list = []

for model in ensemble:
    model.eval()
    with torch.no_grad():
        # Compute LLRs for each individual MSD
        llrs_per_msd = []

        # Process signal MSDs (first num_sd classes)
        for i, sig_data in enumerate(signal_fit_data_list):
            llrs_all = model.llrs_all(sig_data).double()
            llrs_msd_class = llrs_all[:, i]  # (n_samples,) - extract i-th class
            llrs_per_msd.append(llrs_msd_class)

        # Process background MSDs (last num_sd classes)
        for i, bkg_data in enumerate(background_fit_data_list):
            llrs_all = model.llrs_all(bkg_data).double()
            llrs_msd_class = llrs_all[:, num_td + i]  # (n_samples,) - extract (num_sd + i)-th class
            llrs_per_msd.append(llrs_msd_class)

        # Stack: (num_classes, n_samples)
        llrs_msd = torch.stack(llrs_per_msd, dim=0)
        llrs_msd_list.append(llrs_msd)

# Stack across ensemble members: (M, num_classes, n_samples)
llrs_all_msds = torch.stack(llrs_msd_list, dim=0)

# Add f_0 = 1 term (constant in log space)
# Shape: (M+1, num_classes, n_samples)
f0 = torch.ones(1, num_classes, n_samples, dtype=torch.float64, device=device)

llrs_with_f0 = torch.cat([llrs_all_msds, f0], dim=0)

print(f"LLRs with f0 shape: {llrs_with_f0.shape} (M+1={M+1}, num_classes={num_classes}, n_samples={n_samples})", flush=True)

# Precompute reference LLRs once (to avoid recomputing in loss functions)
print("Precomputing reference LLRs...", flush=True)
llrs_ref_list = []
for model in ensemble:
    model.eval()
    with torch.no_grad():
        llrs_ref = model.llrs_all(reference_data).double()  # (n_ref_samples, num_classes)
        # Transpose to (num_classes, n_ref_samples) for consistent indexing with MSD LLRs
        llrs_ref = llrs_ref.t()  # Now (num_classes, n_ref_samples)
        llrs_ref_list.append(llrs_ref)

# Stack: (M, num_classes, n_ref_samples) - consistent with MSD LLRs indexing
llrs_ref_all = torch.stack(llrs_ref_list, dim=0)

# Add f0 = 1 term (constant in log space) for reference LLRs too
# Shape: (M+1, num_classes, n_ref_samples)
n_ref_samples = reference_data.shape[0]
f0_ref = torch.ones(1, num_classes, n_ref_samples, dtype=torch.float64, device=device)
llrs_ref_with_f0 = torch.cat([llrs_ref_all, f0_ref], dim=0)

print(f"Reference LLRs with f0 shape: {llrs_ref_with_f0.shape} (M+1={M+1}, num_classes={num_classes}, n_ref_samples={n_ref_samples})", flush=True)


# WiFi loss function for MSD-to-reference density ratios
def wifi_loss(w):
    """
    MLC loss for WiFi weight fitting of MSD-to-reference density ratios.
    w: weights of shape (M+1) * num_classes

    For each MSD k, we learn r_k(x) = p_k(x) / p_ref(x)
    where p_ref is the pooled distribution across all MSDs.

    The loss ensures E_{p_k}[1/r_k(x)] ≈ 1 and E_{p_ref}[r_k(x)] ≈ 1 for all k.
    """
    M_plus_1 = M + 1
    w_t = torch.tensor(w.reshape(M_plus_1, num_classes), dtype=torch.float64, device=device)

    total_loss = 0.0

    # Data terms: for each MSD k, evaluate on samples from p_k
    for k in range(num_classes):
        # Get LLRs for MSD k: (M+1, n_samples)
        llrs_msd_k = llrs_with_f0[:, k, :]
        # Get weights for MSD k: (M+1,)
        w_msd_k = w_t[:, k]

        # Compute weighted log ratio on MSD k's data: (n_samples,)
        log_r_k_on_msd = (llrs_msd_k * w_msd_k.unsqueeze(1)).sum(dim=0)

        # Data term: -E_{p_k}[log r_k] + E_{p_k}[exp(-log r_k) - 1]
        data_term = -torch.log(log_r_k_on_msd.exp() / (1 + log_r_k_on_msd.exp())).mean()
        total_loss += data_term

    # Reference terms: for each MSD k, evaluate on samples from p_ref
    # Use precomputed reference LLRs with consistent indexing
    for k in range(num_classes):
        # Get LLRs for MSD k on reference: (M+1, n_ref_samples)
        llrs_k_on_ref = llrs_ref_with_f0[:, k, :]

        # Get WiFi weights (including f_0): (M+1,)
        w_msd_k = w_t[:, k]

        # Compute weighted log ratio: (n_ref_samples,)
        log_r_k_on_ref = (llrs_k_on_ref * w_msd_k.unsqueeze(1)).sum(dim=0)
        ref_term = -torch.log(1 / (1 + log_r_k_on_ref.exp())).mean()

        # Reference term: E_{p_ref}[log r_k] + E_{p_ref}[exp(log r_k) - 1]
        total_loss += ref_term

    return total_loss.item()


def wifi_loss_grad(w):
    """Gradient of WiFi loss using autograd."""
    M_plus_1 = M + 1
    w_t = torch.tensor(w.reshape(M_plus_1, num_classes), dtype=torch.float64, device=device, requires_grad=True)

    total_loss = 0.0

    # Data terms: for each MSD k, evaluate on samples from p_k
    for k in range(num_classes):
        llrs_msd_k = llrs_with_f0[:, k, :]
        w_msd_k = w_t[:, k]
        log_r_k_on_msd = (llrs_msd_k * w_msd_k.unsqueeze(1)).sum(dim=0)
        data_term = -torch.log(log_r_k_on_msd.exp() / (1 + log_r_k_on_msd.exp())).mean()
        total_loss = total_loss + data_term

    # Reference terms: for each MSD k, evaluate on samples from p_ref
    # Use precomputed reference LLRs with consistent indexing
    for k in range(num_classes):
        llrs_k_on_ref = llrs_ref_with_f0[:, k, :]
        w_msd_k = w_t[:, k]
        log_r_k_on_ref = (llrs_k_on_ref * w_msd_k.unsqueeze(1)).sum(dim=0)
        ref_term = -torch.log(1 / (1 + log_r_k_on_ref.exp())).mean()
        total_loss = total_loss + ref_term

    total_loss.backward()

    return w_t.grad.cpu().numpy().flatten()


# Initialize weights: uniform for ensemble members, 0.0 for f_0 (log space)
w0 = np.ones((M + 1, num_classes)) / M
w0[-1, :] = 0.0  # f_0 term: log(1) = 0

w0 = w0.flatten()

print(f"Initial loss: {wifi_loss(w0):.6f}", flush=True)

# Optimize using minimize
print("Optimizing WiFi weights...", flush=True)
result = minimize(wifi_loss, w0, method='BFGS', jac=wifi_loss_grad, options={'disp': True})

if result.success:
    print("Optimization succeeded!", flush=True)
else:
    print(f"Optimization warning: {result.message}", flush=True)

w_opt = result.x
print(f"Final loss: {wifi_loss(w_opt):.6f}", flush=True)

# Extract optimized weights
M_plus_1 = M + 1
w_opt_reshaped = w_opt.reshape(M_plus_1, num_classes)

print("\nOptimized weights (shape: M+1 x num_classes):", flush=True)
print(w_opt_reshaped, flush=True)

# Save weights
output_dir = this_dir + 'wifi_weights/'
os.makedirs(output_dir, exist_ok=True)

weights_dict = {
    'weights': torch.tensor(w_opt_reshaped),  # (M+1, num_classes)
    'M': M,
    'num_classes': num_classes,
    'num_td': num_td,
    'msds_idx': msds_idx,
}

output_path = output_dir + f'wifi_weights_final_{num_td}_msd_{msds_idx+1}.pt'
torch.save(weights_dict, output_path)
print(f"\nSaved WiFi weights to {output_path}", flush=True)