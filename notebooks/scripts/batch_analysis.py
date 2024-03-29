# %%
# %load_ext autoreload
# %autoreload 2

from pathlib import Path
from src import (
    batch_helpers as bh,
    visualization as vis
)
from src.probe import ZProbe, YProbe


# %% [markdown]
# # Analysis of multiple recordings
#
# This workflow explains how to efficiently process multiple recordings.
#
#
# ## Interaction between probes
# We assume that each recording is stored in a separate folder.
# Each folder contains two matlab files that contain data simultaneously recorded
# from two probes
#
# For each recording, we want to calculate the following interactions:
# - probe A -> probe B
# - probe B -> probe A
# - within probe A
# - within probe B
#
# Here, we define a list of folders that we want to analyze
# using the `glob` function.
# You can also define a list of folder names manually.

# %%
# select all subfolder
p_root = Path(r'C:\temp\dual_ephys\ALM_STR')
p_dirs = [ p for p in p_root.glob('*/') if p.is_dir() ]
p_dirs

# %% [markdown]
# ### Define probe names
# To make the output more readable and to avoid confusion,
# we define short name for each probe/matlab file using the `probe_names` dictionary.
# If this dictionary is not defined, the names will be set to `proA` and `proB` by default.

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
# ### Run analysis
#
# We finally run the analysis using the `analyze_interactions` function.
# We define an output folder with `out_dir` where the results will be stored.
# This way, we try different parameter sets for the same data.
#
# Because the matlab files are in the `Z` unit structure, we pass the `ZProbe` class as the `probe_class` argument.
# We also pass the `lick_group` argument to this class via `probe_kwargs`.
#

# %%
# define parameters (see example notebook for more details)
params = {
    'bin_size': 0.2,
    'rate_src': (1, None), 
    'rate_trg': (1, None),
    'spike_width_src': (None, None), 
    'spike_width_trg': (  .5, None),
    'trial_overlap': 0.9,             
    'first_lick' : (None, None),
    'type_incl': [ 'l_n', ],
    'scoring': 'r2',
    'subtract_baseline': True,
    'min_units_src': 5,
    'lick_groups': [ 0, 0.6, 1.2, 2.4, 4.8 ],
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
bh.analyze_interactions_probes(
    p_dirs=p_dirs,
    params=params,
    probe_class=ZProbe,
    probe_kwargs={'lick_groups': params['lick_groups']},
    epochs=epochs,
    probe_names=probe_names,
    out_dir='analysis/some_parameters'
)

# %% [markdown]
# ### Y unit structure
# Alternatively, we can analyze the interactions between probes for data stored in the `Y` unit structure.
# The procedure is the same as above, with the following differences:
# - we pass the `YProbe` class as the `probe_class` argument, instead of `ZProbe`
# - we do not pass the `lick_group` argument to the `probe_kwargs` dictionary, because `YProbe` does not use this
# - we removed all lick-related entries from `epochs` and `params`
# - we have not defined the `probe_names` dictionary, so the default names `proA` and `proB` will be used

# %%
p_dirs = [ Path(r'C:\temp\trip_ephys') ]

params = {
    'bin_size': 0.2,
    'rate_src': (1, None), 
    'rate_trg': (1, None),
    'spike_width_src': (None, None), 
    'spike_width_trg': (  .5, None),
    'trial_overlap': 0.9,             
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
}

# run analysis
bh.analyze_interactions_probes(p_dirs=p_dirs,
                        params=params,
                        probe=YProbe,
                        epochs=epochs,
                        out_dir='analysis/some_parameters')

# %% [markdown]
# ### Analyzing the output
# The output folder contains four folders containing the different predictions of activity:
# - `proA_proB`: neurons in probe A -> probe B
# - `proB_proA`: neurons in probe B -> probe A
# - `proA`: subset of neurons in probe A ->  probe A
# - `proB`: subset of neurons in probe B ->  probe B
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
# ## Interaction between areas
# Here we are loading multiple probes and merge them into a single recording object (see `example.py` for details).
# We define a list of folders as `p_dirs` each of which contain multiple matlab files from the same session.
# We also need to define two sets of area codes in the `params` dictionary.
#
# Note that area codes are only available in the `Y` unit structure, not in `Z`.

# %%
p_dirs = [ Path(r'C:\temp\trip_ephys') ]
p_dirs

params = {
    'bin_size': 0.2,
    'rate_src': (1, None), 
    'rate_trg': (1, None),
    'spike_width_src': (None, None), 
    'spike_width_trg': (  .5, None),
    'trial_overlap': 0.9,             
    'type_incl': [ 'l_n', ],
    'scoring': 'r2',
    'subtract_baseline': True,
    'min_units_src': 5,
    'area_code_A': [7, 8],
    'area_code_B': [3, 17],
}
epochs = {
    'all'       : (None, None, 'cue'), 
    'pre_cue'   : (-.6,  .0,   'cue'),
    'post_cue1' : ( .0,  .6,   'cue'),
    'post_cue2' : ( .6, 1.2,   'cue'),
}

bh.analyze_interactions_areas(
    p_dirs=p_dirs,
    params=params,
    probe_class=YProbe,
    epochs=epochs,
    out_dir='analysis/default_params_A_B'
)


# %% [markdown]
#
# ## Different parameter sets
# We can easily run the analyses described above with different parameters and 
# save the results in different folders.
#
# Here, we load two parameter sets that we defined in the file `batch_parameter_sets.yml`.
# Note that the syntax in the YML file is slightly different from the python syntax.
# The two sets only differ in the `subtract_baseline` parameter, therefore,
# we can investigate the effect of baseline subtraction by comparing the results.

# %%
# load parameter sets
param_sets = bh.load_yml('./batch_parameter_sets.yml')

# use the same epochs for all parameter sets
epochs = {
    'all'       : (None, None, 'cue'), 
    'pre_cue'   : (-.6,  .0,   'cue'),
    'post_cue1' : ( .0,  .6,   'cue'),
    'post_cue2' : ( .6, 1.2,   'cue'),
    'pre_lick'  : (-.6,  .0,  'lick'),
    'post_lick' : ( .0,  .6,  'lick'),
}

# loop over parameter sets
for name, params in param_sets.items():
    print(f'>>>> now running parameter set {name}')
    bh.analyze_interactions(
        p_dirs=p_dirs,
        params=params,
        probe=ZProbe,
        epochs=epochs,
        probe_names=probe_names,
        out_dir=f'analysis/{name}'
    )


# %% [markdown]
# # Compare results
#
# Here we load:
# - scores from the ridge regression predictions (`pred_ridge_scores.csv`)
# - optimal rank for the reduced rank regression (`reg_rrr.parquet`)
#
# The results are compiled into a single dataframe.
# Note that this selection inclues all recordings under the `p_root`.

# %%
# select scores for each paramter set
p_root = Path(r'C:\temp\dual_ephys')
p_scores = [ *p_root.glob('**/pred_ridge_scores.csv') ]
scores = bh.load_scores(p_scores)
scores

# %% [markdown]
# ## Compare scores across parameter sets
#
# Here we compare the parameter sets `params_1` and `params_2` that we created in the previous notebook.
# For this we select a subste of the scores:
# - only parameter sets `params_1` and `params_2`
# - only epoch `all`
# - only probes `ALM_STR`
#
#
# The two parameter sets differ only in the `subtract_baseline` parameter.
# We add an additional column to the `scores` dataframe to use as hue in plotting.
#
# Note, that to compare parameter sets for just one animal/recording,
# choose `p_root` accordingly, or pass e.g. `col='animal'` to the plotting function.

# %%
# subset scores dataframe
idx1 = scores.loc[:, 'settings'].isin(['params_1', 'params_2'])
idx2 = scores.loc[:, 'epoch'].eq('all')
idx3 = scores.loc[:, 'probes'].eq('ALM_STR')
df = scores.loc[idx1 & idx2 & idx3, :].copy()

# create new column `subtract_baseline`
d = {'params_1': 'yes', 'params_2': 'no'}
df.loc[:, 'subtract_baseline'] = df.loc[:, 'settings'].map(d)

# plot
vis.plot_box_and_points(df, x='interaction', y='score', hue='subtract_baseline')

# %% [markdown]
# ## Compare scores across epochs
#
# To compare epochs, we select a subset of the scores:
# - only parameter set `params_1`
# - only epochs defined in `l_epochs`
# - only probes `ALM_STR`
#
# When we pass `hue_order=l_epochs` to the plotting function, we can define their oder
# in the plot.

# %%
# subset scores dataframe
l_epochs = ['all', 'pre_lick', 'post_lick']
idx1 = scores.loc[:, 'settings'].eq('params_1')
idx2 = scores.loc[:, 'epoch'].isin(l_epochs)
idx3 = scores.loc[:, 'probes'].eq('ALM_STR')
df = scores.loc[idx1 & idx2 & idx3, :].copy()

# plot
vis.plot_box_and_points(df, x='interaction', y='score', hue='epoch', hue_order=l_epochs)

# %% [markdown]
# ## Compare scores across sessions
#
# To compare sessions, we select a subset of the scores:
# - only parameter set `params_1`
# - only epoch `all`

# %%
# subset scores dataframe
idx1 = scores.loc[:, 'settings'].eq('params_1')
idx2 = scores.loc[:, 'epoch'].eq('all')
df = scores.loc[idx1 & idx2, :].copy()

# plot
vis.plot_box_and_points(df, x='probes', y='score', hue='animal')

# %% [markdown]
# ## Compare optimal rank across brain regions

# %%
# subselection of dataframe
idx1 = scores.loc[:, 'settings'].eq('params_1')
idx2 = scores.loc[:, 'epoch'].eq('all')
df = scores.loc[idx1 & idx2, :].copy()

# select only one of the units
df = scores.groupby(['recording', 'interaction']).first().reset_index()

vis.plot_opt_ranks(df)
