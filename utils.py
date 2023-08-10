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


def get_xy(rec1, bin_size, filter_sigma, rec2=None):

    # load from disk
    df1 = pd.read_parquet(rec1._path_name(f'bin{bin_size}.parquet'))
    
    if rec2 is not None:
        # load from disk
        df2 = pd.read_parquet(rec2._path_name(f'bin{bin_size}.parquet'))

        # select trials
        trials =  rec1.trials & rec2.trials
        df1 = df1.loc[ df1.loc[:, 'trial'].isin(trials) ]
        df2 = df2.loc[ df2.loc[:, 'trial'].isin(trials) ]

        # # make sure bins are equal
        # gr1 = df1.groupby('trial')
        # gr2 = df2.groupby('trial')

        # for t in (gr1.groups.keys() & gr2.groups.keys()):

        #     # select same trial from each session
        #     d1 = gr1.get_group(t)
        #     d2 = gr2.get_group(t)

        #     # check if same number of bins
        #     b1 = len(d1.loc[:, 'bins'])
        #     b2 = len(d2.loc[:, 'bins'])

        #     if b1 != b2:
        #         print(f'INFO bin sizes in trial {t}: {b1} (rec1) and {b2} (rec2). Dropping {abs(b1-b2)} bins')
           

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

    

    # smooth with gaussian
    df1 = smooth_psth(df1, filter_sigma, bin_size)
    df2 = smooth_psth(df2, filter_sigma, bin_size)

    # convert to matrix
    X = pd.pivot_table(df1, values='fr', index='bins', columns='unit').fillna(0).values
    Y = pd.pivot_table(df2, values='fr', index='bins', columns='unit').fillna(0).values

    return X, Y


def linear_regression(X, Y, cv=10):

    pipe = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('mod', LinearRegression())
    ])

    scores = cross_val_score(pipe, X, Y, cv=cv)

    return scores


def ridge_regression(X, Y, alphas):

    pipe = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('mod', Ridge())
    ])

    grd = GridSearchCV(
        pipe, 
        { 'mod__alpha': alphas, },
        cv=10,
        n_jobs=-1
    )

    mods = grd.fit(X, Y)
    return mods


def reduced_rank_regression(X, Y, max_rank):

    r = np.arange(max_rank) + 1

    pipe = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('mod', RRRegressor())
    ])

    grd = GridSearchCV(
        pipe, 
        { 'mod__r': r, },
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

def plot_gridsearch(mods, param, logscale=True):

    # plot ridge
    df = pd.DataFrame(mods.cv_results_)
    ncv = len([c for c in pd.DataFrame(mods.cv_results_).columns if c.startswith('split')])
    x, y, yerr = df.loc[:, param], df.loc[:, 'mean_test_score'], df.loc[:, 'std_test_score'] / np.sqrt(ncv)

    fig, ax = plt.subplots()
    ax.errorbar(x, y, yerr, label=param)
    ax.axvline(x[y.argmax()], c='gray', ls=':')
    ax.set_title(f'best score: {y.max():1.2f}')
    ax.legend()
    ax.set_xlabel('parameter')
    ax.set_ylabel('score')

    if logscale:
        ax.set_xscale('log')

def smooth_psth(df, sigma, bin_size):
    'filter size in [s]'

    for _, d in df.groupby('unit'):

        y = d.loc[:, 'hist'].values
        y = y.astype(float)
        y = gaussian_filter1d(y, sigma=sigma / bin_size) / bin_size

        df.loc[d.index, 'fr'] = y

    
    B = 0
    for _, d in df.groupby('trial'):
        n = len(d.loc[:, 'bins'].unique())
        df.loc[d.index, 'bins_'] = d.loc[:, 'bins'] + B
        B += n
    
    df.loc[:, 'bins_'] -= df.loc[:, 'bins_'].min()
    
    return df

def filter_trial_ranges(rec, thresh=0.9, plot=False):

    trl_max = rec.df_trl.loc[:, 'trial'].max()
    n_unt = rec.df_unt.loc[:, 'unit'].max()

    x = np.zeros((trl_max, n_unt))
    for i in range(n_unt):
        first, last = rec.df_unt.iloc[i, :].loc[['first_trial', 'last_trial']].astype(int)
        
        x[first - 1 : last - 1, i] = 1

    x = x.astype(bool)

    unts = []
    tot = []
    for _ in range(n_unt):
        l = np.all(x, axis=1).sum()
        tot.append(l)
        i = np.argmin(x.sum(axis=0))
        unts.append(i + 1)
        x[:, i] = True

    unts = np.array(unts)
    tot = np.array(tot)
    tot = tot / tot.max()

    if plot:
        fig, ax = plt.subplots()
        ax.plot(tot)
        ax.axhline(thresh, ls=':', c='gray')

        ax.set_xlabel('number of units dropped')
        ax.set_ylabel('fraction of trials kept')

        fig.tight_layout()

    m = tot > thresh
    unts = { *unts[m] }

    df = rec.df_unt.loc[ rec.df_unt.loc[:, 'unit'].isin(unts) ]
    trl_min = df.loc[:, 'first_trial'].max()
    trl_max = df.loc[:, 'last_trial'].max()
    trls = { *range(trl_min, trl_max + 1) }

    return unts, trls


def filter_rates(df_psth, thresh, plot=False):

    d = pd.pivot_table(data=df_psth, values='fr', index='unit', columns='bins_')
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

def plot_missing(df_psth, bin_size, vmax=None, path=''):



    df = pd.pivot_table(data=df_psth, values='fr', index='unit', columns='bins_')
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
    x = d.loc[:, 'bins_'].values * bin_size
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