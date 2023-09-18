import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from custom_models import RRRegressor

import matplotlib.pylab as plt
import seaborn as sns


def preproc_rec(rec, params):
    # summarize preprocessing in function

    # bin spikes
    bs = params['bin_size']
    p_bin = rec._path_name(f'bin{bs}.parquet')
    df = rec._assign_df(p_bin, rec._calculate_psth, {'bin_size': bs})

    # more preprocessing
    p_prec = rec._path_name(f'bin{bs}_prec.parquet')
    rec.df_prec = rec._assign_df(p_prec, rate_and_time, {'df': df, 'bin_size': bs})

    # filter units/trials
    unts_rate = filter_rates(rec.df_prec, params['thresh_rate'])
    unts_sw = filter_spike_width(rec.df_unt, params['thresh_sw'])

    m = rec.df_unt.loc[:, 'unit'].isin(unts_rate & unts_sw)
    unts_range, trls_range = filter_trials(rec.df_unt.loc[m], thresh=params['thresh_trials'], plot=False)

    rec.units = unts_rate & unts_sw & unts_range
    rec.trials = trls_range

def select_data(rec1, rec2=None):
    # TODO implement mean matching
    
    df1 = rec1.df_prec
    
    if rec2 is not None:
        
        df2 = rec2.df_prec

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

    return df1, df2

def get_matrices(df1, df2, signal):

    # convert to pivo
    df_piv1 = pd.pivot_table(df1, values=signal, index='T', columns='unit')
    df_piv2 = pd.pivot_table(df2, values=signal, index='T', columns='unit')

    # get time points missing in other df
    i1 = [ i for i in df_piv1.index if i not in df_piv2.index ]
    i2 = [ i for i in df_piv2.index if i not in df_piv1.index ]

    # append dummy data
    d1 = pd.DataFrame(index=i1, columns=df_piv2.columns)
    d2 = pd.DataFrame(index=i2, columns=df_piv1.columns)

    # long to wide
    df_piv1 = pd.concat([df_piv1, d2]).sort_index()
    df_piv2 = pd.concat([df_piv2, d1]).sort_index()

    # check if time points match, define time basis
    if not np.all(df_piv1.index == df_piv2.index):
        raise ValueError('Mismatching time points in df_piv1 and df_piv2')
    basis_time = df_piv1.index

    # convert to array
    mat1 = df_piv1.fillna(0).values
    mat2 = df_piv2.fillna(0).values

    return mat1, mat2, basis_time


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


def rate_and_time(df, bin_size):
    
    # convert hist to firing rate
    y = df.loc[:, 'hist'].values.astype(float)
    y = y / bin_size
    df.loc[:, 'fr'] = y

    # subtract mean firing rate before cue (bins < 0)
    dss = []
    for _, d in df.groupby(['unit', 'trial'], sort=False):
        
        m = d.loc[:, 'bins'] < 0 # mask for bins < 0
        fr = d.loc[:, 'fr'] # firing rate
        fr_m = fr.loc[ m ].mean() # mean firing rate for bins < 0
        dfr = fr - fr_m # pre-cue subtracted firing rate
        
        ds = pd.Series(index=d.index, data=dfr.values)
        dss.append(ds)

    ds = pd.concat(dss)
    df.loc[ds.index, 'dfr'] = ds.values

    # add absolute time 
    T = 0
    for _, d in df.groupby('trial', sort=False):
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


def get_ypred(X, Y, dfx, dfy, basis_time, mod, scoring=None):

    # get prediction
    Y_pred = mod.predict(X)

    # get scores per unit
    scores = [ cross_val_score(mod, X, Y[:, i], cv=10, scoring=scoring).mean() for i in range(Y.shape[1]) ]

    # get basis in bins
    t2binx = pd.Series(dfx.loc[:, 'bins'].values, index=dfx.loc[:, 'T']).to_dict()
    t2biny = pd.Series(dfy.loc[:, 'bins'].values, index=dfy.loc[:, 'T']).to_dict()
    t2bin = t2binx | t2biny

    # add t2trl mappint here

    # score mapping
    units = dfy.loc[:, 'unit'].unique()
    unt2score = { k: v for k, v in zip(units, scores)}

    # convert back to long format
    df_piv = pd.DataFrame(data=Y_pred, index=basis_time, columns=units)
    df_stack = df_piv.stack()
    v = df_stack.values
    t, u = [ *df_stack.index.to_frame().values.T ]

    dfy_pred = pd.DataFrame(data={
        'unit': u.astype(int),
        # 'trial': [ t2trl[i] for i in t ],
        'pred': v,
        'bins': [ t2bin[i] for i in t ],
        'T': t,
    })


    return dfy_pred, unt2score

def plot_mean_response(df, bin_size, signal, df_pred=None, scores={}, path=''):

    df.loc[:, 'type'] = 'true'
    df = df.rename(columns={signal: 'y'})
    df.loc[:, 'x'] = df.loc[:, 'bins'] * bin_size

    if df_pred is not None:
        df_pred.loc[:, 'type'] = 'predicted'
        df_pred = df_pred.rename(columns={'pred' : 'y'})
        df_pred.loc[:, 'x'] = df_pred.loc[:, 'bins'] * bin_size
        df = pd.concat([df, df_pred], ignore_index=True)

    nu = len(df.loc[:, 'unit'].unique())
    nc = 5
    nr = int(np.ceil(nu / nc))

    fig, axmat = plt.subplots(ncols=nc, nrows=nr, figsize=(3*nc, 2*nr), squeeze=False)

    for ax, (u, d) in zip(axmat.flatten(), df.groupby('unit')):
        sns.lineplot(ax=ax, data=d, x='x', y='y', hue='type', errorbar='sd', legend=False)

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