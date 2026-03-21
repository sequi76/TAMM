## Many wrongs make a right

### Main notebooks 

- toy-model.sk-learn-varying-topics-seed.ipynb
- dihiggs-model.sk-learn-varying-topics-seed.ipynb

The above notebooks use Variational Inference to compute the MAP of the topics model.  Hence one cannot sample from different topics sets.

<b>Important</b>: The directory "data" contains large files with LHC simulations of pp>hh>bbbb and pp>bbbb.  Because of storage limits, it should be downloaded 
separately into your local through XXXXXXXX

### Auxiliary notebooks

Analyze when the true signal fraction is sampled from the prior instead of being fixed:
- toy-model.sk-learn-varying-topics-seed-unbiased_parameter_estimation.ipynb
- dihiggs-model.sk-learn-varying-topics-seed-unbiased_parameter_estimation.ipynb

### Full Bayesian Inference 

- toy-model.ipynb
- dihiggs-model.ipynb

The above notebooks do full Bayesian Inference on the topics.

## Create papers's plots

- run (`plots-paper.ipynb`)
