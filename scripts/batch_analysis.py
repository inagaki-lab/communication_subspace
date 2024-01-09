# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: ml
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

from pathlib import Path
from src import batch_helpers as bh


# %% [markdown]
# This workflow explains how to efficiently process multiple recordings.
#
# We assume that each recording is stored in a separate folder.
# Each folder contains two matlab files that contain data simultaneously recorded from probes in different brain regions.
#
# For each recording, we want to calculate the following interactions:
# - region A -> region B
# - region B -> region A
# - within region A
# - within region B

# %% [markdown]
# # Select folders to analyze
# Here, we define a list of folders that we want to analyze
# using the `glob` function.
# You can also define a list of folder names manually.

# %%
# select all subfolder
p_root = Path(r'C:\temp\dual_ephys\ALM_STR')
p_dirs = [ p for p in p_root.glob('*/') if p.is_dir() ]
p_dirs

# %% [markdown]
# # Define brain regions
# To make the output more readable and to avoid confusion,
# we define short name for each probe/matlab file using the `probe_names` dictionary.

# %%
probe_names = {
    'MK22_20230301_2H2_g0_JRC_units_probe1.mat' : 'ALM1',
    'MK22_20230301_2H2_g0_JRC_units_probe2.mat' : 'ALM2',
    'MK22_20230303_2H2_g0_JRC_units_probe1.mat' : 'ALM1',
    'MK22_20230303_2H2_g0_JRC_units_probe2.mat' : 'ALM2',
    'MK25_20230314_2H2_g0_JRC_units_probe1.mat' : 'ALM1',
    'MK25_20230314_2H2_g0_JRC_units_probe2.mat' : 'ALM2',
    'ZY78_20211015NP_g0_imec0_JRC_units.mat'    : 'STR',
    'ZY78_20211015NP_g0_JRC_units.mat'          : 'ALM',
    'ZY82_20211028NP_g0_imec0_JRC_units.mat'    : 'STR',
    'ZY82_20211028NP_g0_JRC_units.mat'          : 'ALM',
    'ZY83_20211108NP_g0_imec0_JRC_units.mat'    : 'STR',
    'ZY83_20211108NP_g0_JRC_units.mat'          : 'ALM',
    'ZY113_20220617_NPH2_g0_imec0_JRC_units.mat': 'THA',
    'ZY113_20220617_NPH2_g0_JRC_units.mat'      : 'ALM',
    'ZY113_20220618_NPH2_g0_imec0_JRC_units.mat': 'THA',
    'ZY113_20220618_NPH2_g0_JRC_units.mat'      : 'ALM',
    'ZY113_20220620_NPH2_g0_imec0_JRC_units.mat': 'THA',
    'ZY113_20220620_NPH2_g0_JRC_units.mat'      : 'ALM',
}

# %% [markdown]
# # Run analysis
#
# We finally run the analysis using the `analyze_interactions` function.
# We define an output folder with `out_dir` where the results will be stored.
# This way, we try different parameter sets for the same data.

# %%
# define parameters (see example notebook for more details)
params = {
    'bin_size': 0.2,
    'rate_src': (1, None), 
    'rate_trg': (1, None),
    'spike_width_src': (None, None), 
    'spike_width_trg': (  .5, None),
    'perc_trials': 0.9,             
    'first_lick' : (None, None),
    'type_incl': [ 'l_n', ],
    'scoring': 'r2',
    'subtract_baseline': True,
    'min_units_src': 5,
}
epochs = {
    'all'       : (None, None, 'cue'), 
    'pre_cue'   : (-.6,  .0,   'cue'),
    'post_cue1' : ( .0,  .6,   'cue'),
    'post_cue2' : ( .6, 1.2,   'cue'),
    'pre_lick'  : (-.6,  .0,  'lick'),
    'post_lick' : ( .0,  .6,  'lick'),
}

# run analysis
bh.analyze_interactions(p_dirs, params, epochs, probe_names, out_dir='analysis/some_parameters')

# %% [markdown]
# The output folder contains four folders containing the different predictions of activity:
# - `regionA_regionB`: neurons in region A -> region B
# - `regionB_regionA`: neurons in region B -> region A
# - `regionA`: subset of neurons in region A ->  region A
# - `regionB`: subset of neurons in region B ->  region B
#
# Each of these folders contains the following files:
# - `params.json`: the `params` dictionary used for the analysis
# - `epochs.json`: the `epochs` dictionary used for the analysis
# - in a separate subfolder for each epoch:
#     - `reg_ridge.png`: hyperparameter selection for ridge regression
#     - `reg_rrr.png`: hyperparameter selection for reduced-rank regression
#     - `pred_ridge.png`: PSTHs for all target neurons (actual data and ridge regression prediction)
#     - `pred_ridge_scores.csv`: scores for all units in `pred_ridge.png`
#     - `*.parquet`: raw data for CV folds in linear, ridge, and reduced-rank regression
#
# Note that analysis will be skipped if `params.json` is already present,
# so you can rerun the analysis by deleting this file.
#
# See `src.batch_helpers.processing_wrapper` for more details how the analysis is run.
#
# # Different parameter sets
# We can easily run the analysis with different parameters and 
# save the results in different folders.
#
# Here, we load two parameter sets that we defined in the file `batch_parameter_sets.yml`.
# Note that the syntax in the YML file is slightly different from the python syntax.
# The two sets only differ in the `subtract_baseline` parameter, therefore,
# we can investigate the effect of baseline subtraction by comparing the results.

# %%
# load parameter sets
param_sets = bh.load_yml('./batch_parameter_sets.yml')

# loop over parameter sets
for name, params in param_sets.items():
    print(f'>>>> now running parameter set {name}')
    bh.analyze_interactions(p_dirs, params, epochs, probe_names, out_dir=f'analysis/{name}')
