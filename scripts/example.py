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
#     display_name: comm_sub
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2
import numpy as np
from pathlib import Path

from src.recording import Recording
from src import utils as utl

# %% [markdown]
# # User settings
# The user can define settings via the `params`, `epochs`, and `trial_types` dictionaries.
#
#
# The `params` dictionary controls the following:
#
# - `bin_size : float` \
# Bin size in seconds
# - `rate_src : (float, float)`\
# Rate range filter in Hz applied to the source (src) population
# - `rate_trg : (float, float)`\
# Rate range filter in Hz applied to the target (trg) population
# - `spike_width_src : (float, float)`\
# Spike width filter in ms applied to the source (src) population
# - `spike_width_trg : (float, float)`\
# Spike width filter in ms applied to the target (trg) population
# - `perc_trials : float`\
# Percentage of trials filter applied to neurons.
# Since the valid trial range for individual neurons may differ greatly, 
# we choose a fraction of trials that we want to keep (0.9 = 90 %).
# Then, we drop neurons until the remaining neurons cover at least
# this fraction of the maximum available trial range.
# - `first_lick : (float, float)`\
# Lick time filter in seconds applied to trials.
# This is the time of the first lick relative to cue onset.
# - `type_incl: list of str`\
# Trial type filter applied to trials.
# Strings are matched with beginning of `unit(1).Behavior.stim_type_name` strings.
# Set empty list to include all trials.
# - `scoring : str`\
# Score reported in the output.
# Click
# [here](https://scikit-learn.org/stable/modules/model_evaluation.html) to see available scorers.
# Note that the definition of the score does not affect
# definition of the loss function, but only the evaluation of the cross-validation. 
# - `subtract_baseline : bool`\
# whether or not to subtract baseline.
# This subtracts the average firing rate during pre cue period per trial
#
# Note that when defining intervals `(float, float)`, set one of the values to `None`
# for no upper/lower limit.
#
# Note that for within-region analysis, only source thresholds apply.

# %%
params = {
    'bin_size'        : 0.2,
    'rate_src': (1, None), 
    'rate_trg': (1, None),
    'spike_width_src': (None, None), 
    'spike_width_trg': (  .5, None),
    'perc_trials': 0.9,             
    'first_lick' : (None, None),
    'type_incl': [ 'l_n', ],
    'scoring': 'r2',
    'subtract_baseline': True
}

# %% [markdown]
# The `epochs` dictionary is used to define the time intervals of interest.
# The keys are the names of the epochs and the values `(start, end, alignment)`
# define the half-open intervals `[start, end)` in seconds
# and their alignment to the event of interest, such as cue or first lick.
#
#
# Default epoch `all` defined as `['cue' - 2 s, 'lick' + 2 s)`
# Note that settings no limit for either start or finish via `None` 
# only works with cue-aligned epochs.
#
#
# Note that only complete bins are kept.
# For example, if bin_size = 0.2 and first_lick = 1.81,
# then the last bin in this trial is defined as `[1.98, 2.00)`.
#
# Note that lick times may fall between bins, because bins are aligned to cue
#

# %%
epochs = {
    'all'       : (None, None, 'cue'), 
    'pre_cue'   : (-.6,  .0,   'cue'),
    'post_cue1' : ( .0,  .6,   'cue'),
    'post_cue2' : ( .6, 1.2,   'cue'),
    'pre_lick'  : (-.6,  .0,  'lick'),
    'post_lick' : ( .0,  .6,  'lick'),
}

# %% [markdown]
# Trials can be classified using the `trial_types` dictionary.
# TODO

# %%
# define sets of trials 
trial_groups = {
    'lick_0.6': (0.6, None, 'lick'), # e.g. all trials with lick times relative to cue > 0.6 s
    'lick_1.2': (1.2, None, 'lick'),
}

# %% [markdown]
# # Data handling
#
# ## Loading data 
# Data from matlab files is loaded as `rec = Recording('somefolder/matlabfile.mat')`.
#
# The data is stored in the following pandas DataFrames:
# - `rec.df_trl`: trial information
# - `rec.df_unt`: unit/neuron information
# - `rec.df_spk`: spike times
#
# The `Recording` class saves intermediate resutls in a temporary folder,
# which is `somefolder/tmp` by default. This can be controlled via the `tmp_dir` argument.
# The intermediate results are reused when the same data is loaded again.
# To recalculate the intermediate results, either delete the temporary folder or set `force_overwrite=True`.

# %%
data_root = Path(r'C:\temp\dual_ephys')

# ALM-Str (imec0: STR)
rec2 = Recording(data_root / 'ALM_STR/ZY78_20211015/ZY78_20211015NP_g0_JRC_units.mat')
rec1 = Recording(data_root / 'ALM_STR/ZY78_20211015/ZY78_20211015NP_g0_imec0_JRC_units.mat')

# %% [markdown]
# ## Selecting data
#
# For the communication subspace analysis, we need to select a source and a target population, which is done via the `select_data` function:
# - To study the interaction between two regions, 
# we call `X, Y = select_data(rec1, rec2=rec2, params=params)`.
# This applies the filters in the `params` dictionary and returns DataFrames `X` and `Y` with the 
# data corresponding to recordings `rec1` and `rec2`, respectively.
# - To study the interaction within a single region,
# we can call `X, Y = select_data(rec1, rec2=None, params=params)`,
# which will also apply the filters in the `params` dictionary,
# but the data in `X` and `Y` now contains randomly selected neurons from only `rec1`.
#
# Optionally, we can
# - subract the pre-cue baseline firing rate from the data with `subtract_baseline`
# - select a time interval of interest defined in the `epoch` dictionary with `select_epoch`

# %%
# select units and trials, and bin data
dfx_bin, dfy_bin = utl.select_data(rec1, rec2=rec2, params=params)

# subtract baseline
dfx_bin0 = utl.subtract_baseline(dfx_bin, rec1.df_spk)
dfy_bin0 = utl.subtract_baseline(dfy_bin, rec1.df_spk if rec2 is None else rec2.df_spk)

# optional: filter some epoch
dfx_bin0_epo = utl.select_epoch(dfx_bin0, epochs['pre_lick'], rec1.df_trl)
dfy_bin0_epo = utl.select_epoch(dfy_bin0, epochs['pre_lick'], rec1.df_trl if rec2 is None else rec2.df_trl)

# %% [markdown]
# # Model fitting
# Now we use the selected data to fit some models.
# The models predict the activity of the target population based on the activity of the source population.
# Each neuron in the target population is fitted independently and has therefore its own score.
#
# In this example, we use the baseline-subtracted firing rates `dfx_bin0` and `dfy_bin0` to fit the models.
#
# ## Linear model
# The linear model is called via `ridge_regression` while setting the regularization parameter to zero.
#
# ## Ridge regression
# We can do a hyperparameter search for the regularization parameter by passing, a list of values 
# as the `alpha` argument to `ridge_regression`.
#
# `plot_gridsearch` plots the results of the hyperparameter search in blue
# and the score of the linear model in orange.
# Error bars are the standard deviation of the cross-validation scores.
#
# ## Reduced-rank regression
# In reduced-rank regression, TODO explanation
#
# `plot_gridsearch` now compares the linear model and the ridge regression with all 
# ranks of the reduced-rank regression.

# %%
# linear regression (= ridge with alpha=0)
lin_mods = utl.ridge_regression(dfx_bin0, dfy_bin0, scoring=params['scoring'], alphas=[0])
lin_mod = lin_mods.best_estimator_

# ridge
ridge_mods = utl.ridge_regression(dfx_bin0, dfy_bin0, scoring=params['scoring'], alphas=np.logspace(-13, 13, 27))
ridge_mod = ridge_mods.best_estimator_
utl.plot_gridsearch(ridge_mods, 'ridge', other_mods={'linear': lin_mods}, logscale=True)

# %%
# RRR
rr_mods = utl.reduced_rank_regression(dfx_bin0, dfy_bin0, scoring=params['scoring'])
rr_mod = rr_mods.best_estimator_
utl.plot_gridsearch(rr_mods, 'reduced rank', other_mods={'linear': lin_mods, 'ridge': ridge_mods}, logscale=False)
