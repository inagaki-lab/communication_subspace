# Communication subspace
Implementation of the
[communication subspace analysis](https://doi.org/10.1016/j.neuron.2019.01.026) using scikit-learn.

Loosely following the
[MATLAB implementation](https://github.com/joao-semedo/communication-subspace/tree/master)

# How to use this repo
For more information on the structure of this repo, 
see this [template repo](https://github.com/inagaki-lab/template_data_pipelines).

## Analysis pipelines
The script files `scripts/*.py` are workflows for the individual steps in the analysis pipeline.

|script file|use case|
|---|---|
|[`example.py`](scripts/example.py)| detailed explanation on data handling and regression models|
|[`batch_mode.py`](scripts/batch_analysis.py)| convenience workflow to analyze many recordings |

During the installation a notebook file is created in the `notebooks` folder 
for each script file in the `scripts` folder.

## Installation
```bash
# get source code
git clone https://github.com/bidaye-lab/communication_subspace
cd communication_subspace

# create conda environment with necessary dependencies
conda env create -n communication_subspace -f environment.yml
conda activate communication_subspace

# install code as local local python module
pip install -e .

# convert scripts to notebooks
jupytext --sync scripts/*.py
```

