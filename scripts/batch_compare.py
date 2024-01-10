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
from src import visualization as vis
import seaborn as sns


# %% [markdown]
# # Load data
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
# # Compare scores across parameter sets
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
# # Compare scores across epochs
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
# # Compare scores across sessions
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
# # Compare optimal rank across brain regions

# %%
# subselection of dataframe
idx1 = scores.loc[:, 'settings'].eq('params_1')
idx2 = scores.loc[:, 'epoch'].eq('all')
df = scores.loc[idx1 & idx2, :].copy()

# select only one of the units
df = scores.groupby(['recording', 'interaction']).first().reset_index()
df



# %%
vis.plot_opt_ranks(df,)

# %%
