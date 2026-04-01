import torch
import numpy as np
import sys
import torch.nn as nn


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
units = 64
mult = 1

num_td = 1
job_idx = str(sys.argv[1])
print("Job number", job_idx, flush=True)

Nb0 = 1000000


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


def matrix_from_params(means, covs):
    msd_background_matrix = []
    for i in range(len(means)):
        mean = means[i]
        cov = covs[i]
        msd_background_matrix.append(torch.tensor(np.random.multivariate_normal(mean.numpy(), cov.numpy(), Nb0)).float())
    return msd_background_matrix


this_dir = './'

msd_signal_means = torch.tensor(torch.load(this_dir + 'signal_means_final.pt'))
msd_signal_covs = torch.tensor(torch.load(this_dir + 'signal_covs_final.pt'))
msd_background_means = torch.tensor(torch.load(this_dir + 'background_means_final.pt'))
msd_background_covs = torch.tensor(torch.load(this_dir + 'background_covs_final.pt'))

msd_numbers = torch.randint(0, high=msd_signal_means.shape[0], size=(num_td,))
msd_signal_means = msd_signal_means[msd_numbers]
msd_signal_covs = msd_signal_covs[msd_numbers]
msd_background_means = msd_background_means[msd_numbers]
msd_background_covs = msd_background_covs[msd_numbers]
torch.save(msd_signal_means, this_dir + f'varying_msds/signal_baseline_means_final_job_{job_idx}.pt')
torch.save(msd_signal_covs, this_dir + f'varying_msds/signal_baseline_covs_final_job_{job_idx}.pt')
torch.save(msd_background_means, this_dir + f'varying_msds/background_baseline_means_final_job_{job_idx}.pt')
torch.save(msd_background_covs, this_dir + f'varying_msds/background_baseline_covs_final_job_{job_idx}.pt')

signal_msds = matrix_from_params(msd_signal_means, msd_signal_covs)[:num_td]
background_msds = matrix_from_params(msd_background_means, msd_background_covs)[:num_td]

msd_means = torch.cat((msd_signal_means[:num_td], msd_background_means[:num_td]), dim=0)
msd_covs = torch.cat((msd_signal_covs[:num_td], msd_background_covs[:num_td]), dim=0)

print("Using MSDs:", msd_numbers.tolist(), flush=True)

num_sig = 1
num_bkg = 1
num_classes = num_sig + num_bkg

n_samples = min([s.shape[0] for s in signal_msds] + [b.shape[0] for b in background_msds])
signal_msds = [s[:n_samples] for s in signal_msds]
background_msds = [b[:n_samples] for b in background_msds]

print(f"Loaded {n_samples} samples per MSD", flush=True)

# Prepare full dataset
x_all = torch.cat(background_msds + signal_msds, dim=0)
y_all = torch.cat([i * torch.ones(n_samples) for i in range(num_classes)], dim=0)

shuf = torch.randperm(num_classes * n_samples)
x_all = x_all[shuf]
y_all = y_all[shuf]

train_frac = 0.9
n_train = int(train_frac * num_classes * n_samples)
x_train, y_train = x_all[:n_train], y_all[:n_train]
x_val, y_val = x_all[n_train:].to(device), y_all[n_train:].to(device)

n_ensemble = 4
ensemble = []

print(f"\n{'='*60}", flush=True)
print(f"Training ensemble of {n_ensemble} classifiers ({num_classes} classes)", flush=True)
print(f"{'='*60}", flush=True)


for ens_idx in range(n_ensemble):
    print(f"\n--- Ensemble member {ens_idx+1}/{n_ensemble} ---", flush=True)

    # Bootstrap resample
    bootstrap_idx = torch.randint(0, n_train, (n_train,))
    x_boot = x_train[bootstrap_idx].to(device)
    y_boot = y_train[bootstrap_idx].to(device)

    val_bootstrap_idx = torch.randint(0, len(x_val), (len(x_val),))
    x_val_boot = x_val[val_bootstrap_idx]
    y_val_boot = y_val[val_bootstrap_idx]

    model = BinaryClassifier(units=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    trainset = torch.utils.data.TensorDataset(x_boot, y_boot.unsqueeze(1))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4096, shuffle=True)

    n_epochs = 100
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    patience = 10
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.
        for batch_x, batch_y in trainloader:
            optimizer.zero_grad()
            logits = model(batch_x).unsqueeze(1)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(trainloader)

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val_boot).unsqueeze(1)
            val_loss = criterion(val_logits, y_val_boot.unsqueeze(1)).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch}", flush=True)
            break

        print(f"  Epoch {epoch:3d}: train_loss={epoch_loss:.4f}, val_loss={val_loss:.4f}", flush=True)

    # Load best model and freeze
    model.load_state_dict(best_state)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    ensemble.append(model)

print(f"\nEnsemble training complete. ln({num_classes}) = {np.log(num_classes):.4f}", flush=True)

for ii, model in enumerate(ensemble):
    torch.save(model.state_dict(), this_dir + f'varying_msds/baseline_model_final_{job_idx}_ens_{ii}.pt')
