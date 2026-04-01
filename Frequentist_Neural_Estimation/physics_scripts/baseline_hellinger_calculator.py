import torch
import numpy as np
import glob


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


def hellinger_distance(p, q):
    return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

msd_data_dir = '../msd_data_dir/'
msd_signal_list = glob.glob(msd_data_dir + 'signal_ssds_big_var/processed/dihiggs_ssd_big_var_*.dat')
msd_background_list = glob.glob(msd_data_dir + 'background_ssds_big_var/processed/bbbb_ssd_big_var_*.dat')
good_msds = []
for i in range(1, 501):
    if f'dihiggs_ssd_big_var_{i}.dat' in [file.split('/')[-1] for file in msd_signal_list]:
        if f'bbbb_ssd_big_var_{i}.dat' in [file.split('/')[-1] for file in msd_background_list]:
            good_msds.append(i)


results_dir = '../results_dir/'
td_data_dir = '../td_data_dir/'
td_signal = torch.tensor(np.loadtxt(td_data_dir + 'dihiggs_sd_processed.dat')).float()
td_background = torch.tensor(np.loadtxt(td_data_dir + 'bbbb_sd_processed_full.dat')).float()

# === Apply 110–140 GeV mass window cut to SD observed data ===
td_signal = td_signal[(td_signal[:, 0] >= 110) & (td_signal[:, 0] <= 140)
                     & (td_signal[:, 1] >= 110) & (td_signal[:, 1] <= 140)]
td_background = td_background[(td_background[:, 0] >= 110) & (td_background[:, 0] <= 140)
                              & (td_background[:, 1] >= 110) & (td_background[:, 1] <= 140)]

mybinning = [np.linspace(110, 140, 10), np.linspace(110, 140, 10)]
binned_signal_true = normalize_prob(flat_histogramdd(td_signal[:5000].cpu().numpy(), mybinning))
binned_background_true = normalize_prob(flat_histogramdd(td_background[:50000].cpu().numpy(), mybinning))

signal_hds = []
background_hds = []

for i, msd_idx in enumerate(good_msds):
    signal_msd = torch.tensor(np.loadtxt(msd_data_dir + f'signal_ssds_big_var/processed/dihiggs_ssd_big_var_{msd_idx}.dat')).float()
    signal_msd = signal_msd[(signal_msd[:, 0] >= 110) & (signal_msd[:, 0] <= 140)
                             & (signal_msd[:, 1] >= 110) & (signal_msd[:, 1] <= 140)]
    signal_msd = signal_msd[:200000]
    background_msd = torch.tensor(np.loadtxt(msd_data_dir + f'background_ssds_big_var/processed/bbbb_ssd_big_var_{msd_idx}.dat')).float()
    background_msd = background_msd[(background_msd[:, 0] >= 110) & (background_msd[:, 0] <= 140)
                                   & (background_msd[:, 1] >= 110) & (background_msd[:, 1] <= 140)]
    background_msd = background_msd[:200000]
    binned_signal_msd = normalize_prob(flat_histogramdd(signal_msd.cpu().numpy(), mybinning))
    binned_background_msd = normalize_prob(flat_histogramdd(background_msd.cpu().numpy(), mybinning))

    signal_hds.append(hellinger_distance(binned_signal_true, binned_signal_msd))
    background_hds.append(hellinger_distance(binned_background_true, binned_background_msd))
    if i % 10 == 0:
        print(f'Processed {i}/{len(good_msds)} MSDs', flush=True)
        print(f'Current average signal Hellinger distance: {np.mean(signal_hds):.4f}', flush=True)
        print(f'Current average background Hellinger distance: {np.mean(background_hds):.4f}', flush=True)

torch.save(signal_hds, results_dir + 'signal_hellinger_distances.pt')
torch.save(background_hds, results_dir + 'background_hellinger_distances.pt')
