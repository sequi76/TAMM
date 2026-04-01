import torch
import numpy as np
import torch.nn as nn
import re
import ast
import glob
import sys
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def extract_msds(filepath):
    with open(filepath, 'r') as file:
        content = file.read()
    match = re.search(r'Using MSDs: (\[[\d,\s]+\])', content)
    msds = ast.literal_eval(match.group(1)) if match else []
    return msds


class BinaryClassifier(nn.Module):
    """
    Binary classifier for signal vs background.
    Will be used as a basis function fᵢ(x) in the wifi ensemble.
    """
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
        """Returns the basis function value fᵢ(x) (the logit)"""
        x = (x - self.input_mean) / self.input_std
        return self.net(x).squeeze()


# Parse arguments
msds_idx = int(sys.argv[1]) - 1 if len(sys.argv) > 1 else 0
print("Fitting WiFi weights for MSD-to-reference density ratios (MSD config baseline)", flush=True)

this_dir = './'
msd_data_dir = '../msd_data_dir/'

# Load MSD configuration
msd_numbers = extract_msds(this_dir + f'outputs/train_dihiggs_msd_network_baseline_big_var_{msds_idx+1}.out')
print("Using MSDs:", msd_numbers, flush=True)

# Load ensemble
ensemble = []
model_list = glob.glob(this_dir + f'varying_msds_big_var/baseline_model_{msds_idx+1}_ens_*.pt')
for model_path in model_list:
    model_ens = BinaryClassifier(units=64).to(device)
    model_ens.load_state_dict(torch.load(model_path, map_location=device))
    ensemble.append(model_ens)

M = len(ensemble)
print(f"Loaded ensemble with {M} members", flush=True)

# Load ALL MSD data
signal_msds = []
background_msds = []

for msd_idx in msd_numbers:
    sig_data = torch.tensor(np.loadtxt(msd_data_dir + f'signal_ssds_big_var/processed/dihiggs_ssd_big_var_{msd_idx}.dat')).float()
    bkg_data = torch.tensor(np.loadtxt(msd_data_dir + f'background_ssds_big_var/processed/bbbb_ssd_big_var_{msd_idx}.dat')).float()
    signal_msds.append(sig_data)
    background_msds.append(bkg_data)

# Use same number of samples from each MSD
n_samples = min([s.shape[0] for s in signal_msds] + [b.shape[0] for b in background_msds])

signal_fit = [s[:n_samples] for s in signal_msds]
background_fit = [b[:n_samples] for b in background_msds]

signal_data = signal_fit[0].to(device)
background_data = background_fit[0].to(device)

print(f"Samples per MSD: {n_samples}", flush=True)

# Compute LLRs for all ensemble members
# For each ensemble member, compute log r(x) = log p_sig(x) / p_bkg(x)
# Shape: (M, n_samples)
llrs_sig_list = []
llrs_bkg_list = []

for model in ensemble:
    model.eval()
    with torch.no_grad():
        llrs_sig_list.append(model(signal_data).double())
        llrs_bkg_list.append(model(background_data).double())
llrs_sig_list.append(torch.ones(n_samples, dtype=torch.float64, device=device))
llrs_bkg_list.append(torch.ones(n_samples, dtype=torch.float64, device=device))

llrs_sig = torch.stack(llrs_sig_list, dim=0)  # (M+1, n_samples)
llrs_bkg = torch.stack(llrs_bkg_list, dim=0)  # (M+1, n_samples)


def wifi_loss(w):
    w_t = torch.tensor(w, dtype=torch.float64, device=device)
    sig_llrs = w_t @ llrs_sig
    bkg_llrs = w_t @ llrs_bkg
    term1 = -sig_llrs.mean() + (bkg_llrs.exp().mean() - 1)
    term2 = bkg_llrs.mean() + ((-sig_llrs).exp().mean() - 1)
    return (term1 + term2).item()


def wifi_loss_grad(w):
    w_t = torch.tensor(w, dtype=torch.float64, device=device, requires_grad=True)
    sig_llrs = w_t @ llrs_sig
    bkg_llrs = w_t @ llrs_bkg
    term1 = -sig_llrs.mean() + (bkg_llrs.exp().mean() - 1)
    term2 = bkg_llrs.mean() + ((-sig_llrs).exp().mean() - 1)
    loss = term1 + term2
    loss.backward()
    return w_t.grad.cpu().numpy()


# Initialize weights: uniform for ensemble members, 0.0 for f_0 (log space)
w0 = np.ones((M + 1,)) / M
w0[-1] = 0.0  # f_0 term

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

print("\nOptimized weights:", flush=True)
print(w_opt, flush=True)

# Save weights
output_dir = this_dir + 'wifi_weights_big_var/'
os.makedirs(output_dir, exist_ok=True)

weights_dict = {
    'weights': torch.tensor(w_opt),
    'M': M,
    'msds_idx': msds_idx,
    'msd_numbers': msd_numbers
}

output_path = output_dir + f'wifi_weights_baseline_msd_{msds_idx+1}.pt'
torch.save(weights_dict, output_path)
print(f"\nSaved WiFi weights to {output_path}", flush=True)