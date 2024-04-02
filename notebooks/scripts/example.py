# %%
# %load_ext autoreload
# %autoreload 2
import numpy as np
import pandas as pd 
from pathlib import Path

from src.recording import Recording
from src.probe import ZProbe, YProbe
from src import (
    visualization as vis,
    regression_models as reg,
)

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
# - `trial_overlap : float`\
# Betweeen 0 and 1.
# Filter units and trials based on `Trial_info.Trial_range_to_analyze`. 
# Choose first and last trial where `trial_overlap` units are active and discard units that do not cover this range.
# Also drop trials outside this range.
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
# - `baseline_period : (float, float)`\
# Defines the baseline period in seconds relative to cue onset. Must be defined if `subtract_baseline` is `True`.
# - `min_units_src : int`\
# Minimum number of units in source population.
# This will skip the analysis in batch mode.
# - `trial_groups: (str, list of float)`\
# The first element (`str`) defines the column name in the `trial_info` table,
# the second element (`list of float`) defines the time intervals used to group trials.
#
# Note that when defining intervals `(float, float)`, set one of the values to `None`
# for no upper/lower limit.
#
# ## settings specific to ZProbe
# For the `Z` unit structure matlab file, the following settings are available:
# - `first_lick : (float, float)`\
# Lick time filter in seconds applied to trials.
# This is the time of the first lick relative to cue onset.
#
# ## Settings specific to YProbe
# For the `Y` unit structure matlab file, the following settings are available:
# - `only_good`: bool\
# Whether to include only good units (`GoodUnits == 1`).
# - `area_code_A: int or set of int`\
# Aread code(s) to include one population.
# - `area_code_B: int or set of int`\
# Aread code(s) to include in the other population.
#
# Note to compare a subset of units within the same area code(s), set 
# `area_code_A` and `area_code_B` to the same value(s).
#

# %%
# parameters for all unit structures
params_global = {
    'bin_size': 0.2,
    'rate_src': (1, None), 
    'rate_trg': (1, None),
    'spike_width_src': (None, None), 
    'spike_width_trg': (  .5, None),
    'trial_overlap': 0.7,             
    'type_incl': [ 'l_n', ],
    'scoring': 'r2',
    'subtract_baseline': True,
    'baseline_period': (-2, 0), 
    'min_units_src': 5,
}

# parameters for Z unit structure
params_z = params_global.copy()
params_z.update({
    'first_lick' : (None, None),
    'trial_groups': ('dt_lck', [ 0, 0.6, 1.2, 2.4, 4.8 ]),
})

# parameters for Y unit structure
params_y = params_global.copy()
params_y.update({
    'only_good' : True,
    'area_code_A': {7, 8},
    'area_code_B': {3, 17},
    'water_ratio': (None, 3),
    'reward_delay': (.4, None),
    'trial_groups': ('dt_rew', [ 0, 1.0, 2.0 ]),
})

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
# # Data handling
#
# ## Loading data 
# Data from each synchronously recorded  matlab file is loaded into a separate `Probe` objects.
#
# The data is stored in the following pandas DataFrames:
# - `rec.df_trl`: trial information
# - `rec.df_unt`: unit/neuron information
# - `rec.df_spk`: spike times
# - `rec.df_bin`: spike times binned by `params['bin_size']`
#
# These `Probe` attributes are saved in a temporary folder,
# so they can be reused when testing different parameters.
# The name of the temporary folder can be controlled via the `tmp_dir` argument.
# To recalculate the intermediate results, either delete the temporary folder or set `force_overwrite=True`.
#
# ## Loading data from `Z` unit structure
# Currently, the `Z` and `Y` unit structure matlab files are supported.
# To load `Z` matlab files, make sure use `ZProbe` class and `params_z` dictionary.
#
#

# %%
# choose ZProbe
Probe = ZProbe
params = params_z

# load individual probes
data_root = Path(r'C:\temp\dual_ephys')
probe1 = Probe(data_root / 'ALM_STR/ZY78_20211015/ZY78_20211015NP_g0_JRC_units.mat',
               trial_groups=params['trial_groups'], bin_size=params['bin_size'], force_overwrite=False)
probe2 = Probe(data_root / 'ALM_STR/ZY78_20211015/ZY78_20211015NP_g0_imec0_JRC_units.mat',
               trial_groups=params['trial_groups'], bin_size=params['bin_size'], force_overwrite=False)

# combine probes into a recording with descriptive names
probes = {
    'ALM': probe1,
    'STR': probe2,
}
rec = Recording(probes)

# display trial information for ZProbe
vis.plot_trial_infos(rec.df_trl)

# %% [markdown]
# ### Selecting units based on probes
# For the communication subspace analysis, we need to select a source and a target population. 
# The following example shows how set source and target populations based on which probe the 
# neurons were recorded from. This is done via the `rec` object we just created:\
# `rec.select_data_probes(probes_src, probes_trg, params)`
#
# Here, `probes_src` and `probes_trg` are either strings or lists of strings 
# that must match the probe names we defined above.
# We can: 
# - set `probes_src != probes_trg`  to study the interaction between two or more probes
# - set `probes_src == probes_trg`  to study the interaction between a random subsample of units within the same probe(s)
#
# Note when `probes_src != probes_trg`, the same probe name must not appear in both lists.
#
# At the same time, the `params` dictionary is used to filter the data based on the settings defined above.
#
# Finally, we can select a time interval of interest defined in the `epoch` dictionary with `select_epoch`.
# Note that epochs relative to the first lick are only available for `Z` probe data.

# %%
# select units and trials, and bin data
X, Y = rec.select_data_probes('STR', 'ALM', params)

# check if enough units are left is source recording (will skip calculation in batch mode)
if len(X.columns) < params['min_units_src']:
    print('WARNING: Too few units in source recording!')

# # optional: filter some epoch
# X = rec.select_epoch(X, epochs['pre_lick'])
# Y = rec.select_epoch(Y, epochs['pre_lick'])

# %% [markdown]
# ### Loading data from `Y` unit structure
# Make sure to use `YProbe` class and `params_y` dictionary to load `Y` matlab files.

# %%
# choose YProbe
Probe = YProbe
params = params_y

data_root = Path(r'C:\temp\trip_ephys')

probe1 = Probe(data_root / 'J44-20221008_g0_imec0_JRC_units.mat',
               trial_groups=params['trial_groups'], bin_size=params['bin_size'], force_overwrite=False)
probe2 = Probe(data_root / 'J44-20221008_g0_imec1_JRC_units.mat',
               trial_groups=params['trial_groups'], bin_size=params['bin_size'], force_overwrite=False)
probe3 = Probe(data_root / 'J44-20221008_g0_imec2_JRC_units.mat',
               trial_groups=params['trial_groups'], bin_size=params['bin_size'], force_overwrite=False)

probes = {
    'imec0': probe1,
    'imec1': probe2,
    'imec2': probe3,
}
rec = Recording(probes)

# display trial information for YProbe
vis.plot_trial_infos(rec.df_trl)

# %% [markdown]
# ### Selecting units based on area codes
# The selection is analogous to the probe-based selection, but we use:\
# `rec.select_data_area_codes(area_code_A, area_code_B, params)`

# %%
# select units and trials, and bin data
X, Y = rec.select_data_area_code(
    area_code_src=params['area_code_A'], 
    area_code_trg=params['area_code_B'], 
    params=params)

# check if enough units are left is source recording (will skip calculation in batch mode)
if len(X.columns) < params['min_units_src']:
    print('WARNING: Too few units in source recording!')

# # optional: filter some epoch
# X = rec.select_epoch(X, epochs['pre_cue'])
# Y = rec.select_epoch(Y, epochs['pre_cue'])

# %% [markdown]
# # Model fitting
# Now we use the `X` and `Y` data frames to fit some models; from now on we do not have to worry about the data handling anymore.
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
# We can investigate how well the regression model predicts the activity of individual target neurons
# by first calculating the predictions with the `get_ypred` and then plotting the actual and predicted activity
# using the `plot_mean_response` function.

# %%
# linear regression (= ridge with alpha=0)
lin_mods = reg.ridge_regression(X, Y, scoring=params['scoring'], alphas=[0])
lin_mod = lin_mods.best_estimator_

# ridge
ridge_mods = reg.ridge_regression(X, Y, scoring=params['scoring'], alphas=np.logspace(-13, 13, 27))
ridge_mod = ridge_mods.best_estimator_
vis.plot_gridsearch(ridge_mods, 'ridge', other_mods={'linear': lin_mods}, logscale=True)

# %%
# calculate and plot predictions
Y_pred, scores = reg.get_ypred(X, Y, ridge_mod, scoring=params['scoring'])
vis.plot_mean_response(Y, Y_pred, scores)

# %% [markdown]
# ## Reduced-rank regression
# In reduced-rank regression,
# we first calculate the least-squares solution and then project the weight matrix onto 
# the first `rank` principal components. For more details, see `src.regression_models.RRRegressor`.
#
# `plot_gridsearch` now compares the linear model and the ridge regression with all 
# ranks of the reduced-rank regression.

# %%
# RRR
rr_mods = reg.reduced_rank_regression(X, Y, scoring=params['scoring'])
rr_mod = rr_mods.best_estimator_
vis.plot_gridsearch(rr_mods, 'reduced rank', other_mods={'linear': lin_mods, 'ridge': ridge_mods}, logscale=False)

# %% [markdown]
# The optimal rank is defined as the lowest rank that is within one SEM of the CV restuls for the best scoring rank.
# The best model is the one with the highest mean score.

# %%
rrr = reg.analyze_rrr(pd.DataFrame(rr_mods.cv_results_))
rrr

# %% [markdown]
# # Activity modes

# %%
# choose parameters to filter data for mode calculation
params_mode = {
    'bin_size': 0.2,
    'type_incl': [ 'l_n', ],
    'subtract_baseline': True,
    'baseline_period': (-2, 0),
}

# define modes via their intervals
modes = {
    'ramp': {
        'interval_1': ('dt_cue', -0.2, 0),
        'interval_2': ('dt_rew', -0.2, 0),
    },
    'cue': {
        'interval_1': ('dt_cue', -0.2, 0),
        'interval_2': ('dt_cue', 0, 0.2),
    },
    'reward': {
        'interval_1': ('dt_rew', -0.2, 0),
        'interval_2': ('dt_rew', 0, 0.2),
    },
}    

# select one of the modes
mode = modes['cue']

# use data selection method described above, but merge src and trg data
probe_names = rec.df_unt.loc[:, 'probe'].unique()
X, Y = rec.select_data_probes(probe_names, probe_names, params_mode)
Z = pd.concat([X, Y], axis=1)

# calculate and plot ramping mode
df_ramp = rec.get_ramp_mode(Z, mode['interval_1'], mode['interval_2'])
vis.plot_mode(df_ramp, group=True)

# %%
