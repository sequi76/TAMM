# Template-Adapted Mixture Model: Frequentist Neural Estimation

### Instructions to run the Gaussian analysis

The analysis scripts are located in `gaussian_scripts`, and we include the scripts to run the inference pipeline with the baseline and $K = 10$ FNE TAMM for brevity. The results for $K = 2, 4, 6, 8$ can be obtained simply by changing the variable num_td in the $K = 10$ network training, $w_i f_i$ fitting, and pseudo-experiment scripts.

<b>Note</b>: The structure of the code assumes that an output file has been written to an `outputs/` directory by the network training script, as this is used to determine which MSDs should be used for subsequent scripts using those trained networks. If you do not use the attached `.sbatch` files to run the analysis scripts on a cluster as suggested, you will need to make adjustments to the scripts to correctly propagate the MSDs used.

# $K = 10$ recommended workflow to reproduce paper results
1) Run the script `train_gaussian_toy_msd_network_10.py` using sbatch --array=1-30 train_gaussian_toy_msd_network_10.sbatch to train 30 ensembles of basis functions (you may first need to create the directory `varying_msds` within `gaussian_scripts`)
2) Run the script `fit_wifi_weights_gaussian_toy_10.py` using sbatch --array=1-30 fit_wifi_weights_gaussian_toy_10.sbatch to fit $w_i f_i$ weights for these 30 ensembles (you may first need to create the directory `wifi_weights` within `gaussian_scripts`)
3) Run the script `wifi_exponential_param_gaussian_toy_10.py` using sbatch --array=1-30 wifi_exponential_param_gaussian_toy_10.sbatch to obtain predictions and uncertainties, as well as the binned Hellinger distance between the best-fit models and the TD samples. 

# Baseline recommended workflow to reproduce paper results
1) Run the script `train_gaussian_toy_msd_network_baseline.py` using sbatch --array=1-30 train_gaussian_toy_msd_network_baseline.sbatch to train 30 ensembles of basis functions (you may first need to create the directory `varying_msds` within `gaussian_scripts`)
2) Run the script `fit_wifi_weights_gaussian_toy_baseline.py` using sbatch --array=1-30 fit_wifi_weights_gaussian_toy_baseline.sbatch to fit $w_i f_i$ weights for these 30 ensembles (you may first need to create the directory `wifi_weights` within `gaussian_scripts`)
3) Run the script `wifi_baseline_gaussian_toy.py` using sbatch --array=1-30 wifi_baseline_gaussian_toy.sbatch to obtain predictions and uncertainties.

### Instructions to run the di-Higgs analysis

<b>Important</b>: This analysis involves downloading large files with LHC simulations of pp>hh>bbbb and pp>bbbb. With all archives uncompressed, the total size of the data files is tens of GB, so use with care.

First, download the data files from the Zenodo link: https://zenodo.org/records/19341120

Place the TD files `bbbb_sd_combined_processed_feb_12.dat` and `dihiggs_sd_processed.dat` into a directory `td_data_dir` located at the same level as this readme. If you wish to run both Frequentist Neural Estimation and Bayesian Topic Modeling without downloading the large datasets twice, either create a symlink to the relevant location, or change the values of the msd_data_dir and td_data_dir variables in these scripts to the appropriate location.

Then, create another directory `msd_data_dir` at the same level as this readme. Create two subdirectories, `signal_ssds_big_var` and `background_ssds_big_var`. Untar `signal_ssds_processed_big_var.tar.gz` into the former and `background_ssds_processed_big_var.tar.gz` into the latter, so that `msd_data_dir` contains two directories, each containing a directory called `processed`, and each `processed` directory has 500 MSD datasets.

The analysis scripts are located in `physics_scripts`, and we include the scripts to run the inference pipeline with the baseline and $K = 8$ FNE TAMM for brevity. The results for $K = 2, 4, 6$ can be obtained simply by changing the variable num_td in the $K = 8$ network training, $w_i f_i$ fitting, and pseudo-experiment scripts.

<b>Note</b>: The structure of the code assumes that an output file has been written to an `outputs/` directory by the network training script, as this is used to determine which MSDs should be used for subsequent scripts using those trained networks. If you do not use the attached `.sbatch` files to run the analysis scripts on a cluster as suggested, you will need to make adjustments to the scripts to correctly propagate the MSDs used.

# $K = 8$ recommended workflow to reproduce paper results
1) Run the script `train_dihiggs_msd_network_8_big_var.py` using sbatch --array=1-30 train_dihiggs_msd_network_8_big_var.sbatch to train 30 ensembles of basis functions (you may first need to create the directory `varying_msds_big_var` within `physics_scripts`)
2) Run the script `fit_wifi_weights_dihiggs_8_big_var.py` using sbatch --array=1-30 fit_wifi_weights_dihiggs_8_big_var.sbatch to fit $w_i f_i$ weights for these 30 ensembles (you may first need to create the directory `wifi_weights_big_var` within `physics_scripts`)
3) Run the script `wifi_exponential_param_dihiggs_8.py` using sbatch --array=1-30 wifi_exponential_param_dihiggs_8.sbatch to obtain predictions and uncertainties, as well as the binned Hellinger distance between the best-fit models and the TD samples. 
4) Run `baseline_hellinger_calculator.py` to obtain the distances between the MSDs and the TD, for comparison with the results of Step 3.

# Baseline recommended workflow to reproduce paper results
1) Run the script `train_dihiggs_msd_network_baseline_big_var.py` using sbatch --array=1-30 train_dihiggs_msd_network_baseline_big_var.sbatch to train 30 ensembles of basis functions (you may first need to create the directory `varying_msds_big_var` within `physics_scripts`)
2) Run the script `fit_wifi_weights_dihiggs_baseline_big_var.py` using sbatch --array=1-30 fit_wifi_weights_dihiggs_baseline_big_var.sbatch to fit $w_i f_i$ weights for these 30 ensembles (you may first need to create the directory `wifi_weights_big_var` within `physics_scripts`)
3) Run the script `wifi_baseline_dihiggs_big_var_consolidated.py` using sbatch --array=1-30 wifi_baseline_dihiggs_big_var_consolidated.sbatch to obtain predictions and uncertainties.


### Create papers' plots

Run `gaussian_plotter.ipynb` to generate the Gaussian case study plots and `physics_plotter.ipynb` to generate the physics case study plots. Results files are prepopulated in `results_dir`, so the plots can be reproduced from the repository without running the whole analysis.