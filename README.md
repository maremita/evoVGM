# EvoVGM: a Deep Variational Generative Model for Evolutionary Parameter Estimation

### Disclaimer
**EvoVGM** is actively in the development phase. The names of the different package entities and the default values are subject to change.
Please, feel free to contact me if you want to refactor, add, or discuss a feature.

## Overview
**EvoVGM** is a deep variational generative model that simultaneously estimates local evolutionary parameters and generates nucleotide sequence data. Like phylogenetic inference,  we explicitly integrate a continuous-time Markov chain substitution model into the generative model like phylogenetic inference. 

## Dependencies
The `evoVGM` package depends on Python packages that could be installed with `pip`. The main packages that `EvoVGM` uses are `pytorch`, `numpy` and `biopython`. 
You can find the complete list of dependencies in the `requirements.txt` file. This file is used automatically by `setup.py` to install  `evoVGM` using `pip` or `conda`.

## Installation guide
`evoVGM` is developed in Python3 and can be easily installed using `pip`. I recommend installing the package in a separate virtual environment (`virtualenv`, `venv`, `conda env`, ect.). I haven't tested the installation yet using `conda`.

Once the virtual environment is created, `evoVGM` can be installed from the git repository directly through `pip`:
```
python -m pip install git+https://github.com/maremita/evoVGM.git
```
or by cloning the repository  and installing with `pip`:
```
git clone https://github.com/maremita/evoVGM.git
cd evoVGM
pip install .
```

## Usage
### Using `evovgm.py`, the main program
`evovgm.py` is the main program that uses the **EvoVGM** model with different substitution models (JC69, K80 and GTR).
The model can be trained for `nb_replicates` times. 
The program can use a sequence alignment from a `FASTA` file or simulates a new sequence alignment using evolutionary parameters defined in the configuration file.
A new  configuration file can be copied and customized from `evovgm_conf_template.ini`.
`evovgm.py` program could be found in the `scripts/` directory. However, once the `evoVGM` package is installed, it can be called from anywhere using this command line:

```
evovgm.py evovgm_conf_template.ini
```

### Using `evoVGM` package
Classes and functions implemented in the `evoVGM` package can be called and used in Python scripts and notebooks.
This is an example of how to use `evoVGM` to fit an  `EvoVGM_GTR` to estimate ancestral states, branch lengths, substitution rates and relative frequencies, and generate new sequences using these parameters.

```python
from evoVGM.data import build_msa_categorical
from evoVGM.models import EvoVGM_GTR
import torch

# Construct a MSA categorical-embedded data
motifs_cats = build_msa_categorical(sequences) # sequences is a list of strings or a SeqCollection containing the list of the aligned sequences.
X = torch.from_numpy(motifs_cats.data)
X_val = X.clone().detach() # will be used in generation step
X, X_counts = X.unique(dim=0, return_counts=True) # collapse X to unique patterns

# Instantiate an EvoVGM model with GTR substitution model
evoModel = EvoVGM_GTR(
        x_dim=4, # Size of the one-hot encoding of an observable nucleotide
        a_dim=4, # Size of one-hot encoding of an ancestral nucleotide
        h_dim=32, # Size of the hidden layers
        m_dim=len(sequences), # Number of sequences
        ancestor_prior_hp=torch.ones(4)/4, # Hyper-param of ancestral prior dist.
        branch_prior_hp=torch.tensor([0.1, 1.]), # Hyper-param of branch prior dist.
        rates_prior_hp=torch.ones(6), # Hyper-param of rates prior dist.
        freqs_prior_hp=torch.ones(4)) # Hyper-param of frequencies prior dist.

# Fit the model to estimates the evolutionary parameters
evoModel.fit(X, X_counts, 
             latent_sample_size=100, # Size of sampling 
             sample_temp=0.1, # Temperature for sampling from categorical distribution
             alpha_kl=0.001, # $\alpha_{KL}$ hyper-parameter
             max_iter=5000, # Number of iterations
             optim="adam", # sgd or adam
             optim_learning_rate=0.005,
             optim_weight_decay=0.)

# Once the model is fitted, we can generate new sequences using the inferred evolutionary parameters. 
results = evoModel.generate(X_val, None,
                            latent_sample_size=100,
                            sample_temp=0.1,
                            alpha_kl=0.001)
```

`scripts/train_evogtr.py`  is an example script that uses the `evoVGM` package to train `EvoVGM_GTR` model using simulated sequences.

## Experiments
The [`evoVGM_Exp`](https://github.com/maremita/evoVGM_Exp "evoVGM_Exp") project contains scripts that run `evovgm.py` on different grid schemas (evolutionary models, data types and sizes, and hyper-parameters) locally and on slurm clusters.
There are also scripts that help sum up and plot the results of the experiments.

To reproduce the experiments and the results of the [EvoVGM article](#how-to-cite), you can use the configuration files found in `evoVGM_Exp/exp_2022_bcb/`.

## How to cite
```
@article{remita2022evovgm,
  title={EvoVGM: a Deep Variational Generative Model for Evolutionary Parameter Estimation},
  author={Remita, Amine M and Diallo, Abdoulaye Banir{\'e}},
  journal={arXiv preprint arXiv:2205.13034},
  year={2022}
}
```
## License
The EvoVGM package including the modules and the scripts is distributed under the **MIT License**.

## Contact
If you have any questions, please do not hesitate to contact:
- Amine Remita <remita.amine@courrier.uqam.ca>

