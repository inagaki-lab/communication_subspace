# Communication subspace
Implementation of the
[communication subspace analysis](https://doi.org/10.1016/j.neuron.2019.01.026) using scikit-learn.

Loosely following the
[MATLAB implementation](https://github.com/joao-semedo/communication-subspace/tree/master)

# How to use this repo
For more information on the structure of this repo, 
see this [template repo](https://github.com/inagaki-lab/template_data_pipelines).

## Analysis pipelines
The Jupyter notebooks in [`notebooks/`](./notebooks/) are workflows for the individual steps in the analysis pipeline.

|notebook file|use case|
|---|---|
|[`example.ipynb`](notebooks/example.ipynb)| detailed explanation on data handling and regression models|
|[`batch_mode.ipynb`](notebooks/batch_analysis.ipynb)| convenience workflow to analyze many recordings |

## Installation
```bash
# get source code
git clone https://github.com/inagaki-lab/communication_subspace
cd communication_subspace

# create conda environment with necessary dependencies
conda env create -n communication_subspace -f environment.yml
conda activate communication_subspace

# install code as local local python module
pip install -e .

```

## Update
```bash
# pull from github
cd communication_subspace
git pull origin main
```

Note that will result in an error, if you have modified any file tracked by version control.
To revert any changes, use `git status` to see which files have been modified and then `git reset --hard` to revert the changes.
Then run the `git pull` command again.
