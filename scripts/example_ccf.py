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
import numpy as np
import pandas as pd 
from pathlib import Path

from src.recording import Recording
from src.probe import ZProbe, YProbe
from src import (
    visualization as vis,
    regression_models as reg,
)

# %%
# the only parameter we have to specify is the bin size
params = {
    'bin_size': 0.001,
}

# load probes
Probe = ZProbe
data_root = Path(r'C:\temp\dual_ephys')
probe1 = Probe(data_root / 'ALM_STR/ZY78_20211015/ZY78_20211015NP_g0_JRC_units.mat',
               bin_size=params['bin_size'], force_overwrite=False)
probe2 = Probe(data_root / 'ALM_STR/ZY78_20211015/ZY78_20211015NP_g0_imec0_JRC_units.mat',
               bin_size=params['bin_size'], force_overwrite=False)

# combine probes into a recording with descriptive names
probes = {
    'ALM': probe1,
    'STR': probe2,
}
rec = Recording(probes)

# select subset of data
probe_names = rec.df_unt.loc[:, 'probe'].unique()
X, Y = rec.select_data_probes(probe_names, probe_names, params)
Z = pd.concat([X, Y], axis=1).sort_index(axis=1)

# %%
df = Z.reset_index().groupby('trial').max()
df

# %%
bin_max = df.min().loc['bin'].item()

# %%
type(bin_max.item())

# %%
mask = Z.index.get_level_values(1) < bin_max
Z.loc[mask].reset_index().groupby('trial').max()

# %%
import seaborn as sns
sns.histplot(df,x='bin')


# %%

def calculate_positive_correlation(a, b, lag):

    n = np.sqrt( np.sum(a**2) * np.sum(b**2))
    b = np.pad(b, (0, lag), mode='constant')
    c = np.correlate(a, b, mode='valid')
    c = c / n

    return c

lag = 25
units_a = Z.loc[:, ('ALM', )].columns
units_b = Z.loc[:, ('STR', )].columns
corr = np.zeros((len(units_a), len(units_b), lag+1))

for ia, ua in enumerate(units_a):
    print(ua)
    a = Z.loc[:, ('ALM', ua)]
    a = a.values.astype(np.uint)
    for ib, ub in enumerate(units_b):
        b = Z.loc[:, ('STR', ub)]
        b = b.values.astype(np.uint)
        c = calculate_positive_correlation(a, b, lag)
        corr[ia, ib, :] = c


# %%
corr.shape


# %%

def ccf_gpu(spk_a, spk_b, time_lag, n_shuffle):

    nr, _ = spk_a.shape
    bins = np.arange(-time_lag, time_lag+1)
    a_mat = np.hstack([np.zeros((nr, time_lag)), spk_a]) # pad 0 before trials
    b_mat = np.hstack([np.zeros((nr, time_lag)), spk_b])

    a, b = a_mat.flatten(), b_mat.flatten() # 1D array
    a = a[time_lag:] # strip 0 before first trial for a
    b = np.hstack([b, np.zeros(time_lag)]) # add 0 after last trial for b

    a, b = cp.array(a).astype(cp.uint), cp.array(b).astype(cp.uint)
    norm = cp.sqrt( cp.sum(a**2) * cp.sum(b**2))

    rng = np.random.default_rng()
    a_sh = [ cp.array(rng.permutation(a_mat, axis=0).flatten()[time_lag:]).astype(cp.uint) for _ in range(n_shuffle) ]

    ys = []
    for a in [ a ] + a_sh:

        if norm:

            y = cp.correlate(a, b, mode='valid') 
            y = y / norm
            y = y.get()
        
        else:
            y = np.zeros_like(bins)
        
        ys.extend([*y])
    
    ts = [*bins] * (n_shuffle + 1)
    n_shs = [ i for i in range(n_shuffle + 1) for _ in bins ]
    data = np.array([ts, ys, n_shs]).T
    df = pd.DataFrame(data=data, columns=['t', 'ccf', 'n_sh'])
    
    return df

