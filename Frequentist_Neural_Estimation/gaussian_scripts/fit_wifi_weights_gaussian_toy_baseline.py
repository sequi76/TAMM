import torch
import numpy as np
import torch.nn as nn
import glob
import sys
from scipy.optimize import minimize
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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
        self.input_mean = 0.
        self.input_std = 1.

    def forward(self, x):
        """Returns the basis function value fᵢ(x) (the logit)"""
        x = (x - self.input_mean) / self.input_std
        return self.net(x).squeeze()


# Parse arguments
msds_idx = int(sys.argv[1]) - 1 if len(sys.argv) > 1 else 0
print("Fitting WiFi weights for MSD-to-reference density ratios (MSD config baseline)", flush=True)

this_dir = './'

Nb0 = 1000000

# Load saved MSD parameters (written by training script)
signal_means = torch.load(this_dir + f'varying_msds/signal_baseline_means_final_job_{msds_idx+1}.pt')
signal_covs = torch.load(this_dir + f'varying_msds/signal_baseline_covs_final_job_{msds_idx+1}.pt')
background_means = torch.load(this_dir + f'varying_msds/background_baseline_means_final_job_{msds_idx+1}.pt')
background_covs = torch.load(this_dir + f'varying_msds/background_baseline_covs_final_job_{msds_idx+1}.pt')
print("Loaded MSD parameters", flush=True)

# Load ensemble
ensemble = []
model_list = sorted(glob.glob(this_dir + f'varying_msds/baseline_model_final_{msds_idx+1}_ens_*.pt'))
for model_path in model_list:
    model_ens = BinaryClassifier(units=64).to(device)
    model_ens.load_state_dict(torch.load(model_path, map_location=device))
    ensemble.append(model_ens)

M = len(ensemble)
print(f"Loaded ensemble with {M} members", flush=True)

# Generate MSD data from parameters
signal_msds = []
background_msds = []
for i in range(len(signal_means)):
    signal_msds.append(torch.tensor(np.random.multivariate_normal(
        signal_means[i].numpy(), signal_covs[i].numpy(), Nb0)).float())
    background_msds.append(torch.tensor(np.random.multivariate_normal(
        background_means[i].numpy(), background_covs[i].numpy(), Nb0)).float())

# Use same number of samples from each MSD
n_samples = min([s.shape[0] for s in signal_msds] + [b.shape[0] for b in background_msds])

signal_data = signal_msds[0][:n_samples].to(device)
background_data = background_msds[0][:n_samples].to(device)

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
    term1 = -torch.log(sig_llrs.exp()/(1+sig_llrs.exp())).mean()
    term2 = -torch.log(1/(1+bkg_llrs.exp())).mean()
    print((term1+term2).item(), flush=True)
    return (term1 + term2).item()


def wifi_loss_grad(w):
    w_t = torch.tensor(w, dtype=torch.float64, device=device, requires_grad=True)
    sig_llrs = w_t @ llrs_sig
    bkg_llrs = w_t @ llrs_bkg
    term1 = -torch.log(sig_llrs.exp()/(1+sig_llrs.exp())).mean()
    term2 = -torch.log(1/(1+bkg_llrs.exp())).mean()
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
output_dir = this_dir + 'wifi_weights/'
os.makedirs(output_dir, exist_ok=True)

weights_dict = {
    'weights': torch.tensor(w_opt),
    'M': M,
    'msds_idx': msds_idx
}

output_path = output_dir + f'wifi_weights_final_baseline_msd_{msds_idx+1}.pt'
torch.save(weights_dict, output_path)
print(f"\nSaved WiFi weights to {output_path}", flush=True)