# Template-Adapted Mixture Model: Frequentist Neural Estimation

### Instructions to run the Gaussian analysis

### Instructions to run the di-Higgs analysis

<b>Important</b>: This analysis involves downloading large files with LHC simulations of pp>hh>bbbb and pp>bbbb. With all archives uncompressed, the total size of the data files is tens of GB, so use with care.

First, download the data files from the Zenodo link: https://zenodo.org/records/19341120

Place the TD files `bbbb_sd_combined_processed_feb_12.dat` and `dihiggs_sd_processed.dat` into a directory `td_data_dir` located at the same level as this readme.

Then, create another directory `msd_data_dir` at the same level as this readme. Create two subdirectories, `signal_ssds_big_var` and `background_ssds_big_var`. Untar `signal_ssds_processed_big_var.tar.gz` into the former and `background_ssds_processed_big_var.tar.gz` into the latter, so that `msd_data_dir` contains two directories, each containing a directory called `processed`, and each `processed` directory has 500 MSD datasets.

The analysis scripts are located in `physics_scripts`, and we include the scripts to run the inference pipeline with the baseline and $K = 8$ FNE TAMM for brevity. The results for $K = 2, 4, 6$ can be obtained simply by changing the variable num_td in the $K = 8$ network training, $w_i f_i$ fitting, and pseudo-experiment scripts.

<b>Note</b>: The structure of the code assumes that an output file has been written to an `outputs/` directory by the network training script, as this is used to determine which MSDs should be used for subsequent scripts using those trained networks. If you do not use the attached `.sbatch` files to run the analysis scripts on a cluster as suggested, you will need to make adjustments to the scripts to correctly propagate the MSDs used.

# $K = 8$ recommended workflow to reproduce paper results
1) Run the script `train_dihiggs_msd_network_8_big_var.py` using sbatch --array=1-30 train_dihiggs_msd_network_8_big_var.sbatch to train 30 ensembles of basis functions (you may first need to create the directory `varying_msds_big_var` within `physics_scripts`)
2) Run the script `fit_wifi_weights_dihiggs_8_big_var.py` using sbatch --array=1-30 fit_wifi_weights_dihiggs_8_big_var.sbatch to fit $w_i f_i$ weights for these 30 ensembles (you may first need to create the directory `wifi_weights_big_var` within `physics_scripts`)
3) Run the script `wifi_exponential_param_dihiggs_8.py` using sbatch --array=1-30 wifi_exponential_param_dihiggs_8.sbatch to obtain predictions and uncertainties for the parameter $\mu \equiv \frac{\kappa}{1-\kappa}$ in terms of the signal fraction $\kappa$, defined for convenience so that the true $\mu = 0.1$, as well as the binned Hellinger distance between the best-fit models and the TD samples (you may first need to create the directory `results_dir` at the same level as the directory `physics_scripts`)

### Create papers' plots

Run `gaussian_plotter.ipynb` to generate the Gaussian case study results and `physics_plotter.ipynb` to generate the physics case study results.