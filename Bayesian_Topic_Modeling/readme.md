# Template-Adapted Mixture Model

### Main notebooks 

- `toy-model.sk-learn-varying-topics-seed.ipynb`
- `dihiggs-model.sk-learn-varying-topics-seed.ipynb`

The above notebooks use Variational Inference to compute the MAP of the topics model.  Hence one cannot sample from different topics sets.

<b>Important</b>: The directory `data/` contains large files with LHC simulations of pp>hh>bbbb and pp>bbbb.  Because of storage limits, it should be downloaded separately into your local from Zenodo link:  https://zenodo.org/records/19341120  .  Read instructions inside `data/` folder on how to download from Zenodo.

### Create papers' plots

- run `plots-paper.ipynb`


<br>
<br>
<br>
<br>
<br>

## The notebooks below are not needed in a first pass

## Auxiliary notebooks

(move them to this folder to run them in agreement with their relative path)

Analyze when the true signal fraction is sampled from the prior instead of being fixed:
- `toy-model.sk-learn-varying-topics-seed-unbiased_parameter_estimation.ipynb`
- `dihiggs-model.sk-learn-varying-topics-seed-unbiased_parameter_estimation.ipynb`

### Full Bayesian Inference 

- `toy-model.ipynb`
- `dihiggs-model.ipynb`

The above notebooks do full Bayesian Inference on the topics.  They are put just as reference if one whishes to have a posterior over the topics.  We've found that does not contribute too much and takes about 100x more time.


