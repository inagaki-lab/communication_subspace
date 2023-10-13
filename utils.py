import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from custom_models import RRRegressor

import matplotlib.pylab as plt
import seaborn as sns


def filter_data(rec, params):

    # filter units/trials
    unts_rate = filter_rate(rec.df_spk, rec.df_unt, rec.df_trl, params['thresh_rate'])
    unts_sw = filter_sw(rec.df_unt, params['thresh_sw'])

    m = rec.df_unt.loc[:, 'unit'].isin(unts_rate & unts_sw)
    unts_range, trls_range = filter_trials(rec.df_unt.loc[m], thresh=params['thresh_trials'], plot=False)

    rec.units = unts_rate & unts_sw & unts_range
    rec.trials = trls_range


def bin_spikes(df_spk, df_trl, bin_size):

    # get dict with bin sized per trial
    trl2bin = dict()
    for trl, t0, tf in df_trl.loc[:, ['trial', 'dt0', 'dtf']].itertuples(index=False):
        tf = tf - tf % bin_size # clip last bin
        if tf != tf: # skip trials with nan as end
            continue

        # construct bins
        b = np.arange(t0, tf + bin_size, bin_size)    
        trl2bin[trl] = b

    # define basis for array
    unts = df_spk.loc[:, 'unit'].unique()
    i_trl = np.concatenate([ (len(v)-1) * [ k ] for k, v in trl2bin.items() ])
    i_bin = np.concatenate([ v[:-1] for k, v in trl2bin.items() ])

    # initialize array
    X = np.empty((len(i_trl), len(unts)))
    X[:] = np.nan

    # fill array
    for trl, df in df_spk.groupby('trial'):

        # get bins
        bins = trl2bin[trl]
        
        # apply bin to each unit
        gr = df.groupby('unit')
        df = gr.apply(lambda x: np.histogram(x.loc[:, 't'], bins)[0]).apply(pd.Series)
        df /= bin_size # spikes per bin -> spikes per s

        # select correct indices for array
        s0 = i_trl == trl
        s1 = np.isin(unts, df.index)
        idx = np.ix_(s0, s1)

        # assign values to array
        X[ idx ] = df.values.T

    # convert to dataframe with multiindex
    df = pd.DataFrame(
        data=X, 
        index=pd.MultiIndex.from_arrays([i_trl, i_bin], names=('trial', 'bin')), 
        columns=unts)
    
    return df    



def select_data(rec1, params, rec2=None):
    # TODO implement mean matching

    filter_data(rec1, params)
    
    df1 = rec1.df_spk.copy()
    
    if rec2 is not None:
        
        filter_data(rec2, params)

        df2 = rec2.df_spk.copy()

        # select trials common to both
        trials =  rec1.trials & rec2.trials
        df1 = df1.loc[ df1.loc[:, 'trial'].isin(trials) ]
        df2 = df2.loc[ df2.loc[:, 'trial'].isin(trials) ]  

        # select units
        df1 = df1.loc[ df1.loc[:, 'unit'].isin(rec1.units) ]
        df2 = df2.loc[ df2.loc[:, 'unit'].isin(rec2.units) ]

        # bin spikes
        df1_bin = bin_spikes(df1, rec1.df_trl, params['bin_size'])
        df2_bin = bin_spikes(df2, rec2.df_trl, params['bin_size'])

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

        # bin spikes
        df1_bin = bin_spikes(df1, rec1.df_trl, params['bin_size'])
        df2_bin = bin_spikes(df2, rec1.df_trl, params['bin_size'])

    # discard rows with only nan in both dfx_bin and dfy_bin
    m = df1_bin.isnull().all(axis=1) & df2_bin.isnull().all(axis=1)
    df1_bin = df1_bin.loc[ ~m ]
    df2_bin = df2_bin.loc[ ~m ]

    # fill all remaining nan with 0
    df1_bin = df1_bin.fillna(0)
    df2_bin = df2_bin.fillna(0)

    return df1_bin, df2_bin

def subtract_baseline(df_bin, df_spk, interval=(-2, 0)):

    # select only spikes within `interval`
    t0, tf = interval
    t = df_spk.loc[:, 't']
    m = (t < tf) & ( t > t0 )
    df = df_spk.loc[m]

    # number of spikes per unit per trial
    df_n = df.groupby(['unit', 'trial']).size()
    
    # convert to rate
    dt = tf - t0
    df_r = df_n / dt

    # convert to pivot table
    df_r = df_r.reset_index()
    df_r = pd.pivot_table(data=df_r, index='trial', columns='unit', values=0)
    # nan implies 0 Hz 
    df_r = df_r.fillna(0)

    # match structure with `df_bin`
    idx = np.unique(df_bin.index.get_level_values(0))
    cols = df_bin.columns
    df_r = df_r.loc[ idx, cols]

    # subtract
    df_bin0 = df_bin.subtract(df_r)

    return df_bin0


def ridge_regression(dfx_bin, dfy_bin, alphas, scoring=None):
    
    X, Y = dfx_bin.values, dfy_bin.values

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


def reduced_rank_regression(dfx_bin, dfy_bin, max_rank=None, scoring=None):

    X, Y = dfx_bin.values, dfy_bin.values

    if max_rank is None:
        max_rank = Y.shape[1]
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

def plot_gridsearch(mods, param, other_mods=dict(), logscale=True, path=''):

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

    fig.tight_layout()
    if path:
        fig.savefig(path)
        plt.close(fig)

def save_cv_results(mods, path):
    
    df = pd.DataFrame(mods.cv_results_)
    
    df.to_parquet(path)


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


def filter_sw(df_unt, thresh):

    # filter based on spike width
    m = df_unt.loc[:, 'spike_width'] > thresh
    unts = set(df_unt.loc[ m, 'unit'])

    return unts

def filter_rate(df_spk, df_unt, df_trl, thresh):

    # get mapping from trial to duration
    df = df_trl.loc[:, ['trial', 'dt0', 'dtf']].copy()
    df.loc[:, 'dur'] = df.loc[:, 'dtf'] - df.loc[:, 'dt0']
    ds = df.set_index('trial').loc[:, 'dur']

    # get duration for each unit
    d = dict()
    for u, f, l in df_unt.loc[:, ['unit', 'first_trial', 'last_trial']].itertuples(index=False):
        dur = np.nansum([ ds.loc[i] for i in range(f, l+1) ])
        d[u] = dur

    # number of spikes per unit
    df = df_spk.groupby('unit', as_index=False).size()

    # average firing rate
    df.loc[:, 'rate'] = df.loc[:, 'size'] / df.loc[:, 'unit'].map(d)

    # select units based on rate threshold 
    m = df.loc[:, 'rate'] > thresh
    unts = { *df.loc[m, 'unit'] }  

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

def plot_unit(rec, bin_size, unit, xlims=(None, None), path=''):


    fig, ax = plt.subplots(figsize=(20, 5))

    d = rec.df_prec.groupby('unit').get_group(unit)
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


def get_ypred(dfx_bin, dfy_bin, mod, scoring=None):
    
    X = dfx_bin.values

    # get prediction
    Y_pred = mod.predict(X)

    # get scores per unit
    cvs = lambda u: cross_val_score(mod, X, dfy_bin.loc[:, u].values, cv=10, scoring=scoring)
    unt2score = { u: cvs(u).mean() for u in dfy_bin }

    return Y_pred, unt2score

def plot_mean_response(df_bin, arr_pred=None, scores={}, path=''):

    if arr_pred is not None:
        df_bin_pred = df_bin.copy()
        df_bin_pred.loc[:, :] = arr_pred

    nu = len(df_bin.columns)
    nc = 5
    nr = int(np.ceil(nu / nc))

    fig, axmat = plt.subplots(ncols=nc, nrows=nr, figsize=(3*nc, 2*nr), squeeze=False)

    for ax, u in zip(axmat.flatten(), df_bin.columns):

        df = df_bin.loc[:, u].reset_index()
        df.loc[:, 'type'] = 'true'
        if arr_pred is not None:
            df_pred = df_bin_pred.loc[:, u].reset_index()
            df_pred.loc[:, 'type'] = 'pred'
            df = pd.concat([df, df_pred])

        sns.lineplot(ax=ax, data=df, x='bin', y=u, hue='type', errorbar='sd', legend=False)

        title = f'unit {u}'
        if scores:
            title += f' ({scores[u]:1.3f})'
        ax.set_title(title)
        ax.margins(x=0)
        ax.set_xlabel('')
        ax.set_ylabel('')

    for ax in axmat[-1]:
        ax.set_xlabel('time from cue [s]')
    for ax in axmat.T[0]:
        ax.set_ylabel('firing rate [Hz]')

    for ax in axmat.flatten():
        if not ax.has_data():
            ax.set_axis_off()

    fig.tight_layout()
    if path:
        fig.savefig(path)
        plt.close(fig)