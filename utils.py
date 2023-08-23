import pandas as pd
import numpy as np
from scipy.ndimage import  gaussian_filter1d

from recording import Recording

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from custom_models import RRRegressor

import matplotlib.pylab as plt
import seaborn as sns


def select_data(rec1, bin_size, rec2=None):
    # TODO implement mean matching
    
    # load from disk
    df1 = pd.read_parquet(rec1._path_name(f'bin{bin_size}.parquet'))
    
    if rec2 is not None:
        # load from disk
        df2 = pd.read_parquet(rec2._path_name(f'bin{bin_size}.parquet'))

        # select trials common to both
        trials =  rec1.trials & rec2.trials
        df1 = df1.loc[ df1.loc[:, 'trial'].isin(trials) ]
        df2 = df2.loc[ df2.loc[:, 'trial'].isin(trials) ]  

        # select units
        df1 = df1.loc[ df1.loc[:, 'unit'].isin(rec1.units) ]
        df2 = df2.loc[ df2.loc[:, 'unit'].isin(rec2.units) ]

    else: 
        # select trials
        df1 = df1.loc[ df1.loc[:, 'trial'].isin(rec1.trials) ]

        # select units

        # if only rec1, select two subsets randomly
        rng = np.random.default_rng(seed=42)
        unts = [ *rec1.units ]
        n = len(unts)

        rng.shuffle(unts)
        s = int(n / 2)
        u1, u2 = unts[:-s], unts[-s:]

        df2 = df1.loc[ df1.loc[:, 'unit'].isin(u2) ]
        df1 = df1.loc[ df1.loc[:, 'unit'].isin(u1) ]

    # convert hist to firing rate, subtract pre=cue
    df1 = rate_and_time(df1, bin_size)
    df2 = rate_and_time(df2, bin_size)

    return df1, df2

def get_matrix(df):

    # convert to matrix
    df_mat = pd.pivot_table(df, values='dfr', index='T', columns='unit').fillna(0)

    return df_mat

def matrix2df(X, dfx):

    t = dfx.loc[:, 'T'].values
    bins = dfx.loc[:, 'bins'].values
    trl = dfx.loc[:, 'trial'].values
    t2bins = pd.Series(index=t, data=bins).to_dict()   
    t2trl = pd.Series(index=t, data=trl).to_dict()   
    
    df_piv = pd.pivot_table(dfx, values='dfr', index='T', columns='unit').fillna(0)
    df_piv.loc[:, :] = X
    df_stack = df_piv.stack()
    dfr = df_stack.values
    t, unt  = [ *df_stack.index.to_frame().values.T ]
    df = pd.DataFrame(data={
        'unit': unt.astype(int),
        'trial': [ t2trl[i] for i in t ],
        'dfr': dfr,
        'bins': [ t2bins[i] for i in t ],
        'T': t,
    })

    return df


# def linear_regression(X, Y, cv=10):

#     pipe = Pipeline(steps=[
#         # ('scaler', StandardScaler()),
#         ('mod', LinearRegression())
#     ])

#     scores = cross_val_score(pipe, X, Y, cv=cv)

#     return scores


def ridge_regression(X, Y, alphas, scoring=None):

    pipe = Pipeline(steps=[
        # ('scaler', StandardScaler()),
        ('mod', Ridge())
    ])

    grd = GridSearchCV(
        pipe, 
        { 'mod__alpha': alphas, },
        scoring=scoring,
        cv=10,
        n_jobs=-1,
    )

    mods = grd.fit(X, Y)
    return mods


def reduced_rank_regression(X, Y, max_rank, scoring=None):

    r = np.arange(max_rank) + 1

    pipe = Pipeline(steps=[
        # ('scaler', StandardScaler()), 
        ('mod', RRRegressor())
    ])

    grd = GridSearchCV(
        pipe, 
        { 'mod__r': r, },
        scoring=scoring,
        cv=10,
        n_jobs=-1,
    )

    mods = grd.fit(X, Y)

    return mods

def plot_rate_dist(X, Y, path=''):
    # plot rate distributions

    fig, ax = plt.subplots()

    r1 = np.mean(X, axis=0)
    r2 = np.mean(Y, axis=0)

    sns.histplot(
        ax=ax,
        data={f'X (n = {len(r1)})': r1, f'Y (n = {len(r2)})': r2}, 
        binwidth=1, stat='density', common_norm=False)
    ax.set_xlabel('avg firing rate [Hz]')
    
    fig.tight_layout()
    if path:
        fig.savefig(path)
        plt.close(fig)

def plot_gridsearch(mods, param, other_mods=dict(), logscale=True):

    # plot ridge
    df = pd.DataFrame(mods.cv_results_)
    ncv = len([c for c in df.columns if c.startswith('split')])
    x = [ np.array(*i.values()).item() for i in df.loc[:, 'params'] ]
    y, yerr = df.loc[:, 'mean_test_score'], df.loc[:, 'std_test_score'] / np.sqrt(ncv)

    fig, ax = plt.subplots()
    ax.errorbar(x, y, yerr, label=param)
    ax.axvline(x[y.argmax()], c='gray', ls=':')

    for i, (name, mods) in enumerate(other_mods.items()):
        df = pd.DataFrame(mods.cv_results_)
        ncv = len([c for c in df.columns if c.startswith('split')])
        i_max = df.loc[:, 'mean_test_score'].argmax()
        ds = df.loc[i_max, :]
        x = np.array(*ds.loc['params'].values()).item()
        y, yerr = ds.loc['mean_test_score'], ds.loc['std_test_score'] / np.sqrt(ncv)
        ax.axhline(y, lw=1, label=name, c=f'C{i+1}')
        ax.axhline(y+yerr, ls=':', lw=1, c=f'C{i+1}')
        ax.axhline(y-yerr, ls=':', lw=1, c=f'C{i+1}')

    ax.legend()
    ax.set_title(f'best score: {y.max():1.2f}')
    ax.set_xlabel('parameter')
    ax.set_ylabel('score')

    if logscale:
        ax.set_xscale('log')

# def smooth_psth(df, sigma, bin_size):
#     'filter size in [s]'

#     for _, d in df.groupby('unit'):

#         y = d.loc[:, 'hist'].values
#         y = y.astype(float)
        
#         y = gaussian_filter1d(y, sigma=sigma / bin_size) / bin_size

#         df.loc[d.index, 'fr'] = y

#     return df

def rate_and_time(df, bin_size):
    
    # convert hist to firing rate
    y = df.loc[:, 'hist'].values.astype(float)
    y = y / bin_size
    df.loc[:, 'fr'] = y

    # subtract mean firing rate before cue (bins < 0)
    for _, d in df.groupby(['unit', 'trial']):
        
        m = d.loc[:, 'bins'] < 0 # mask for bins < 0
        fr = d.loc[:, 'fr'] # firing rate
        fr_m = fr.loc[ m ].mean() # mean firing rate for bins < 0
        df.loc[d.index, 'dfr'] = fr - fr_m # pre-cue subtracted firing rate

    # add absolute time 
    T = 0
    for _, d in df.groupby('trial'):
        n = len(d.loc[:, 'bins'].unique())
        df.loc[d.index, 'T'] = d.loc[:, 'bins'] + T
        T += n
    
    df.loc[:, 'T'] -= df.loc[:, 'T'].min()
    
    return df

def filter_trials(df_unt, thresh=0.9, plot=False):

    all_unts = [ *df_unt.loc[:, 'unit'].unique() ]

    # create valid trial matrix: trial x units
    trl_max = df_unt.loc[:, 'last_trial'].max()
    n_unt = len(all_unts)

    x = np.zeros((trl_max, n_unt))
    for i in range(n_unt):
        first, last = df_unt.iloc[i, :].loc[['first_trial', 'last_trial']].astype(int)
        x[first - 1 : last - 1, i] = 1

    x = x.astype(bool)

    # sort units by trial overlap
    idx, tot = [], [] # `idx` is index in `all_unts` and `tot` is trial overlap before removing this unit
    for _ in range(n_unt):
        l = np.all(x, axis=1).sum()
        tot.append(l)
        i = np.argmin(x.sum(axis=0))
        idx.append(i)
        x[:, i] = True

    all_unts = np.array([ all_unts[i] for i in idx ])
    tot = np.array(tot)
    tot = tot / trl_max # trial overlap in %

    if plot:
        fig, ax = plt.subplots()
        ax.plot(tot)
        ax.axhline(thresh, ls=':', c='gray')

        ax.set_xlabel('number of units dropped')
        ax.set_ylabel('fraction of trials kept')

        fig.tight_layout()

    # select based on overlap threshold
    m = tot > thresh
    unts = { *all_unts[m] }

    df = df_unt.loc[ df_unt.loc[:, 'unit'].isin(unts) ]
    trl_min = df.loc[:, 'first_trial'].max()
    trl_max = df.loc[:, 'last_trial'].max()
    trls = { *range(trl_min, trl_max + 1) }

    return unts, trls


def filter_rates(df_psth, thresh, plot=False):

    d = pd.pivot_table(data=df_psth, values='fr', index='unit', columns='T')
    d = d.fillna(0).mean(axis=1)

    if plot:
        fig, ax = plt.subplots()

        sns.histplot(data=d.values, ax=ax, cumulative=True, binwidth=0.5, element='step', fill=False)
        ax.axvline(thresh, ls=':', c='gray')
        ax.set_xlabel('rate [Hz]')

        fig.tight_layout()

    d = d.loc[d > thresh]
    unts = { *d.index }

    return unts

def filter_spike_width(df_unt, thresh):

    m = df_unt.loc[:, 'spike_width'] > thresh
    unts = set(df_unt.loc[ m, 'unit'])

    return unts

def plot_missing(df_psth, bin_size, vmax=None, path=''):


    df = pd.pivot_table(data=df_psth, values='fr', index='unit', columns='T')
    # df = df.apply(lambda x: x / np.mean(x), axis=1)

    n_unt, n_bins = df.shape
    fig, axarr = plt.subplots(ncols=2, width_ratios=(3, 1), figsize=(n_bins / 0.5e4, n_unt / 10))

    ax = axarr[0]
    im = ax.pcolormesh(df.columns * bin_size, df.index, df.values, vmax=vmax)
    
    label = 'firing rate [Hz]'
    if vmax:
        label = label + f' (capped at {vmax})'
        
    fig.colorbar(im, ax=ax, location='right', orientation='vertical', label=label)
    ax.set_xlabel('time [s]')
    ax.set_ylabel('unit')
    # for b in B:
    #     ax.axvline(b * bin_size, lw=.1, c='gray')

    ax = axarr[1]
    r_nan = (df != df).sum(axis=1) / df.shape[1]
    y, x = r_nan.index, r_nan.values * 100
    ax.barh(y, x)
    ax.margins(y=0)
    ax.set_xlim((0, 100))
    ax.set_ylabel('unit')
    ax.set_xlabel('missing [%]')

    fig.tight_layout()
    if path:
        fig.savefig(path)
        plt.close(fig)

def plot_unit(df_psth, rec, bin_size, unit, xlims=(None, None), path=''):


    fig, ax = plt.subplots(figsize=(20, 5))

    d = df_psth.groupby('unit').get_group(unit)
    x = d.loc[:, 'T'].values * bin_size
    y = d.loc[:, 'fr'].values

    rec.df_trl.loc[:, 'Tf'] = np.cumsum(rec.df_trl.loc[:, 'dtf'] - rec.df_trl.loc[:, 'dt0'])
    for trl in d.loc[:, 'trial'].unique():
        tf = rec.df_trl.groupby('trial').get_group(trl).loc[:, 'Tf'].item()
        ax.axvline(tf, c='gray', lw=.5)

    ax.plot(x, y)
    ax.margins(x=0)

    ax.set_xlabel('time [s]')
    ax.set_ylabel('firing rate [Hz]')
    ax.set_xlim(xlims)

    fig.tight_layout()
    if path:
        fig.savefig(path)
        plt.close(fig)


def plot_psth(df, bin_size, df2=None, scores={}, path=''):

    df.loc[:, 'type'] = 'true'
    df.loc[:, 't'] = df.loc[:, 'bins'] * bin_size

    if df2 is not None:
        df2.loc[:, 'type'] = 'predicted'
        df2.loc[:, 't'] = df2.loc[:, 'bins'] * bin_size
        df = pd.concat([df, df2], ignore_index=True)

    # TODO apply boxcar filter

    nu = len(df.loc[:, 'unit'].unique())
    nc = 5
    nr = int(np.ceil(nu / nc))

    fig, axmat = plt.subplots(ncols=nc, nrows=nr, figsize=(nu, 2*nr))

    for ax, (u, d) in zip(axmat.flatten(), df.groupby('unit')):
        sns.lineplot(ax=ax, data=d, x='t', y='dfr', hue='type', errorbar='sd', legend=False)

        title = f'unit {u}'
        if scores:
            title += f' ({scores[u]:1.2f})'
        ax.set_title(title)
        ax.margins(x=0)
        ax.set_xlabel('')
        ax.set_ylabel('')

    for ax in axmat[-1]:
        ax.set_xlabel('time from cue [s]')
    for ax in axmat.T[0]:
        ax.set_ylabel('firing rate [Hz]')

    fig.tight_layout()
    if path:
        fig.savefig(path)
        plt.close(fig)