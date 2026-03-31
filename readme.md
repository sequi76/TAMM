# Many wrongs make a right

### Main notebooks 

- `toy-model.sk-learn-varying-topics-seed.ipynb`
- `dihiggs-model.sk-learn-varying-topics-seed.ipynb`

The above notebooks use Variational Inference to compute the MAP of the topics model.  Hence one cannot sample from different topics sets.

<b>Important</b>: The directory `data/` contains large files with LHC simulations of pp>hh>bbbb and pp>bbbb.  Because of storage limits, it should be downloaded separately into your local from Zenodo link:  https://zenodo.org/records/19341120

Read carefully how to bring the .dat files from Zenodo.  Once downloaded all files from Zenodo to your local (where you've cloned this repo!), place the files `bbbb_sd_combined_processed_feb_12.dat` and `dihiggs_sd_processed.dat` into the `data/` folder.   Whereas the two huge .tar.gz files shuold be untared using 'tar -xzf file.tar.gz' from CLI. Each one of them will create a folder named 'processed' with many .dat files inside (be careful, when untaring both create a folder with the same name...).  You need to create inside the `data/` folder the folders `signal_ssds_big-var_processed` and `background_ssds_big-var_processed` and put inside them the corresponding .dat files that come out of untarring the .tar.gz files.

## The notebooks below are not needed in a first pass

### Auxiliary notebooks

Analyze when the true signal fraction is sampled from the prior instead of being fixed:
- `toy-model.sk-learn-varying-topics-seed-unbiased_parameter_estimation.ipynb`
- `dihiggs-model.sk-learn-varying-topics-seed-unbiased_parameter_estimation.ipynb`

### Full Bayesian Inference 

- `toy-model.ipynb`
- `dihiggs-model.ipynb`

The above notebooks do full Bayesian Inference on the topics.  They are put just as reference if one whishes to have a posterior over the topics.  We've found that does not contribute too much and takes about 100x more time.

## Create papers' plots

- run `plots-paper.ipynb`
