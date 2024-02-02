import numpy as np
import pandas as pd

def filter_trials_thresh(rec, type_incl=[], first_lick=(None, None)):
    '''Filter trials based on trial type and first lick time.

    Parameters
    ----------
    rec : Recording
        Recording to be processed.
    type_incl : list of str, optional
        Keep trials in which `trial_type` start with `str`, by default []
    first_lick : tuple, optional
        Keep trials in which lick time is between `first_lick[0]` and `first_lick[1]`.
        If None, no upper/lower limit is applied, by default (None, None)

    Returns
    -------
    trials : set of int
        Trials to be included.
    '''
    
    # filter based on trial type
    tt = rec.df_trl.loc[:, 'trial_type']
    if type_incl:
        # only trials starting with strings defined in type_incl
        l_ds = [ tt.str.startswith(s) for s in type_incl ]
        df = pd.concat(l_ds, axis=1)
        m_tt = df.any(axis=1)

    else:
        # all trials
        m_tt = tt == tt
    
    # filter based on lick time
    lck_min, lck_max = first_lick
    
    lck = rec.df_trl.loc[:, 'dt_lck']
    m_lck = lck == lck # DataSeries with all True for not nan

    if lck_min is not None:
        m = lck > lck_min
        m_lck = m_lck & m
    
    if lck_max is not None:
        m = lck < lck_max
        m_lck = m_lck & m
    
    m = m_tt & m_lck
    trials = { *rec.df_trl.loc[m, 'trial'] }
    
    return trials

def set_trial_unit_filters(rec, rate_range, sw_range, perc_trial):
    '''Apply filters to units and trials.

    Assigns `rec.units` and `rec.trials` after filtering 
    based on firing rate, spike width, and time coverage.

    Parameters
    ----------
    rec : Recording
        Recording object for which to assign filtered units and trials.
    rate_range : tuple
        Interval for firing rate in Hz.
    sw_range : tuple
        Interval for spike width in ms.
    perc_trial : float
        Fraction of trials to cover with units.
    '''

    # filter units/trials
    unts_rate = filter_rate(rec.df_spk, rec.df_unt, rec.df_trl, *rate_range)
    unts_sw = filter_sw(rec.df_unt, *sw_range)

    m = rec.df_unt.loc[:, 'unit'].isin(unts_rate & unts_sw)
    if m.any():
        unts_range, trls_range = filter_trials(rec.df_unt.loc[m], thresh=perc_trial)
        rec.units = unts_rate & unts_sw & unts_range
        rec.trials = trls_range
    else:
        print(f'INFO: 0 units after filtering left for {rec.session}')
        rec.units = set()
        rec.trials = set()


def bin_spikes(df_spk, df_trl, bin_size):
    '''Bin spike trains into bins of size `bin_size`

    Skips trial with nan as end time.



    Parameters
    ----------
    df_spk : pandas.DataFrame
        Spike times for each unit and trial in long format
    df_trl : pandas.DataFrame
        Trial information
    bin_size : float
        Bin size in seconds

    Returns
    -------
    df : pandas.DataFrame
        Binned spikes in wide format
    '''

    # get dict with bin size per trial
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
    i_bin = np.concatenate([ v[:-1] for _, v in trl2bin.items() ])

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
    '''Wrapper for selecting data from one or two recordings.

    Filteres units and trials based on parameters in `params`.

    Returns source and target activity in wide format with 
    matching time basis.

    Parameters
    ----------
    rec1 : Recording
        Recording object for first dataset.
    params : dict
        Dictionary with parameters for selecting data.
    rec2 : Recording or None, optional
        Recording object for second dataset. If None, select random sets from 
        first recording, by default None

    Returns
    -------
    df1 : pandas.DataFrame
        Binned and filtered spikes for source population
    df2 : pandas.DataFrame
        Binned and filtered spikes for target population
    '''

    # get trial to include based on trial type
    trials_incl = filter_trials_thresh(rec1, type_incl=params['type_incl'], first_lick=params['first_lick'])

    # store filtered units and trials
    set_trial_unit_filters(rec1, 
                           rate_range=params['rate_src'], 
                           sw_range=params['spike_width_src'], 
                           perc_trial=params['perc_trials'])
    
    # load or calculate binned spikes
    rec1.path_bin = rec1._path_tmp / 'bin{}.hdf'.format(params['bin_size'])
    df1 = rec1._assign_df(rec1.path_bin, bin_spikes, {'df_spk': rec1.df_spk, 'df_trl': rec1.df_trl, 'bin_size': params['bin_size']})
    
    if rec2 is not None:

        # store filtered units and trials
        set_trial_unit_filters(rec2, 
                            rate_range=params['rate_trg'], 
                            sw_range=params['spike_width_trg'], 
                            perc_trial=params['perc_trials'])
        
        # load or calculate binned spikes
        rec2.path_bin = rec2._path_tmp / 'bin{}.hdf'.format(params['bin_size'])
        df2 = rec2._assign_df(rec2.path_bin, bin_spikes, {'df_spk': rec2.df_spk, 'df_trl': rec2.df_trl, 'bin_size': params['bin_size']})

        # select trials common to both
        trials =  rec1.trials & rec2.trials & trials_incl
        idx1 = df1.index.get_level_values(0).isin(trials) 
        idx2 = df2.index.get_level_values(0).isin(trials)
        assert np.array_equal(idx1, idx2)

        # select units
        col1 = df1.columns.isin(rec1.units)
        col2 = df2.columns.isin(rec2.units)

        # apply filters
        df1 = df1.loc[ idx1, col1 ]
        df2 = df2.loc[ idx2, col2 ]

    else: 
        # select trials
        idx = df1.index.get_level_values(0).isin(rec1.trials & trials_incl)

        # select units
        # if only rec1, select two subsets randomly
        rng = np.random.default_rng(seed=42)
        unts = [ *rec1.units ]
        n = len(unts)

        rng.shuffle(unts)
        s = int(n / 2)
        u1, u2 = unts[:-s], unts[-s:]

        col1 = df1.columns.isin(u1)
        col2 = df1.columns.isin(u2)

        # apply filters
        df2 = df1.loc[ idx, col2 ]
        df1 = df1.loc[ idx, col1 ]


    # discard rows with only nan in both dfx_bin and dfy_bin
    m = df1.isnull().all(axis=1) & df2.isnull().all(axis=1)
    df1 = df1.loc[ ~m ]
    df2 = df2.loc[ ~m ]

    # fill all remaining nan with 0
    df1 = df1.fillna(0)
    df2 = df2.fillna(0)

    return df1, df2


def filter_and_bin(rec, params):
    '''Apply filters from params to single recording

    Filters units and trials and returnes binned spikes.

    Parameters
    ----------
    rec : Recording
        Recording object
    params : dict
        Dictionary with parameters for selecting data.

    Returns
    -------
    df : pandas.DataFrame
        Binned and filtered spikes
    '''

    # get trial to include based on trial type
    trials_incl = filter_trials_thresh(rec, type_incl=params['type_incl'], first_lick=params['first_lick'])

    # store filtered units and trials
    set_trial_unit_filters(rec, 
                            rate_range=params['rate_src'], 
                            sw_range=params['spike_width_src'], 
                            perc_trial=params['perc_trials'])

    # load or calculate binned spikes
    rec.path_bin = rec._path_tmp / 'bin{}.hdf'.format(params['bin_size'])
    df = rec._assign_df(rec.path_bin, bin_spikes, {'df_spk': rec.df_spk, 'df_trl': rec.df_trl, 'bin_size': params['bin_size']})
    
    # select trials
    idx = df.index.get_level_values(0).isin(rec.trials & trials_incl)

    # select units
    col = df.columns.isin(rec.units)

    # apply filters
    df = df.loc[ idx, col ]

    # discard rows with only nan 
    m = df.isnull().all(axis=1)
    df = df.loc[ ~m ]

    # fill all remaining nan with 0
    df = df.fillna(0)

    return df

def subtract_baseline(df_bin, df_spk, interval=(-2, 0)):
    '''Subtract baseline firing rate from binned spikes.

    Returns new dataframe with same basis as `df_bin`, 
    but with baseline subtracted.

    Subtracts the mean firing rate during `interval` from
    each unit and trial.

    Parameters
    ----------
    df_bin : pandas.DataFrame
        Binned spikes
    df_spk : pandas.DataFrame
        Spike times
    interval : tuple of float, optional
        Interval for baseline calculation in seconds relative to cue, by default (-2, 0)

    Returns
    -------
    df_bin0 : pandas.DataFrame
        Binned spikes with baseline subtracted
    '''

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

def select_epoch(df_bin, epoch, df_trl=None):
    '''Select only bins within `epoch` from binned spikes.

    Returns new dataframe with only bins within `epoch`.

    Parameters
    ----------
    df_bin : pandas.DataFrame
        Binned spikes
    epoch : tuple
        Epoch defined as `(t0, tf, align)`, where `t0` and `tf` are beginning and end
        relative to `align` (cue or lick) in seconds.
        `t0` and `tf` can be None (no limit) if `align == 'cue'`.
    df_trl : pandas.DataFrame, optional
        Trial info, necessary when `align == 'lick'`, by default None

    Returns
    -------
    df_epo : pandas.DataFrame
        Binned spikes within `epoch`

    Raises
    ------
    NotImplementedError
        If `align` is not 'cue' or 'lick'.
    '''

    t0, tf, align = epoch

    if align == 'cue':
        # bins are already aligned to cue
        df_epo = df_bin.loc[ (slice(None), slice(t0, tf)), :].copy() # copy necessary?

    elif align == 'lick':

        # check if df_trl has been passed
        assert df_trl is not None, f'Need df_trl when aligning to {align}'
        
        # map trial to lick time ('dt_lck' is aligned to cue)
        trl2lck = { k: v for k, v in df_trl.loc[:, ['trial', 'dt_lck']].itertuples(index=False)}

        # cycle trhough trials in df_bin
        dfs = [] # collect relevant dataframes here
        for trl in np.unique(df_bin.index.get_level_values(0)):
            t_lck = trl2lck[trl]
            df = df_bin.loc[ ([trl], slice(t0 + t_lck, tf + t_lck)), : ]
            dfs.append(df)

        # combine snippets again
        df_epo = pd.concat(dfs)

    else:
        raise NotImplementedError(f'Do not know how to align to {align}')

    return df_epo

def filter_trials(df_unt, thresh=0.9):
    '''Filter units based on trial overlap.

    Returns trials and units that survive filtering.

    Because not all units cover the whole recording,
    dropping units can increase the total time overlap.
    Here, units with the lowest overlap are dropped 
    iteratively until the total overlap is just above `thresh`.


    Parameters
    ----------
    df_unt : pandas.DataFrame
        Unit info
    thresh : float, optional
        Fraction of trials to keep, by default 0.9

    Returns
    -------
    unts : set of int
        Units to keep
    trls : set of int
        Trials to keep
    '''

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

    # select based on overlap threshold
    m = tot > thresh
    unts = { *all_unts[m] }

    df = df_unt.loc[ df_unt.loc[:, 'unit'].isin(unts) ]
    trl_min = df.loc[:, 'first_trial'].min()
    trl_max = df.loc[:, 'last_trial'].max()
    if (trl_min != trl_min) or (trl_max != trl_max):
        trls = set()
    else:
        trls = { *range(trl_min, trl_max + 1) }

    return unts, trls


def filter_sw(df_unt, sw_min=None, sw_max=None):
    '''Filter units based on spike width.

    Returns set of units with spike width between `sw_min` and `sw_max`.

    Parameters
    ----------
    df_unt : pandas.DataFrame
        Unit info
    sw_min : float or None, optional
        Min spike width in ms. If None, no lower limit, by default None
    sw_max : float or None, optional
        Max spike width in ms. If None, no upper limit, by default None

    Returns
    -------
    unts : set of int
        Units to keep
    '''
    
    # filter based on spike width
    sw = df_unt.loc[:, 'spike_width']
    m = sw == sw # DataSeries with all `True` if not nan

    if sw_min is not None:
        m_min = sw > sw_min
        m = m & m_min

    if sw_max is not None:
        m_max = sw < sw_max
        m = m & m_max

    unts = { *df_unt.loc[ m, 'unit'] }

    return unts

def filter_rate(df_spk, df_unt, df_trl, r_min=None, r_max=None):
    '''Filter units based on firing rate.

    Returns set of units with firing rate between `r_min` and `r_max`.

    Parameters
    ----------
    df_spk : pandas.DataFrame
        Spike times
    df_unt : pandas.DataFrame
        Unit info
    df_trl : pandas.DataFrame
        Trial info
    r_min : float or None, optional
        Min firing rate in Hz. If None, no lower limit, by default None
    r_max : float or None, optional
        Max firing rate in Hz. If None, no upper limit, by default None

    Returns
    -------
    unts : set of int
        Units to keep
    '''

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
    rate = df.loc[:, 'rate']
    m = rate == rate # DataSeries with all `True` if not nan

    if r_min is not None:
        m_min = rate > r_min
        m = m & m_min

    if r_max is not None:
        m_max = rate < r_max
        m = m & m_max

    unts = { *df.loc[m, 'unit'] }  

    return unts

def add_lick_group(df_trl, lick_groups=[0]):
    '''Add a column to the dataframe that specifies the lick group for each trial.

    Parameters
    ----------
    df_trl : pandas.DataFrame
        Trial information
    lick_groups : dict, optional
        Lick times to construct intervals in seconds relative to cue, by default [0]
    '''
    
    dt_lck = df_trl.loc[:, 'dt_lck']
    t0, tf = dt_lck.min(), dt_lck.max()

    if t0 < lick_groups[0]:
        lick_groups = [ t0 ] + lick_groups
    if tf > lick_groups[-1]:
        lick_groups = lick_groups + [ tf ]

    labels = [ f'{t0:.1f} - {tf:.1f}' for t0, tf in zip(lick_groups[:-1], lick_groups[1:]) ]

    lck_grp = pd.cut(dt_lck, bins=lick_groups, labels=labels, right=True, include_lowest=True)
    lck_grp = lck_grp.cat.add_categories(['no_lick'])
    lck_grp = lck_grp.fillna('no_lick')

    df_trl.loc[:, 'lick_group'] = lck_grp

def group_ramp_mode(ds_ramp, df_trl, col):
    '''Group ramp mode by column in `df_trl`

    Returns dataframe with bin, trials and `col` for plotting.

    Parameters
    ----------
    ds_ramp : pd.Series
        Ramp mode with (trial, bin) as MultiIndex
    df_trl : pd.DataFrame
        Trial information
    col : str
        Column in `df_trl` to group by

    Returns
    -------
    df : pd.DataFrame
        Mean ramp mode for each group
    '''

    ramp_trl = ds_ramp.index.get_level_values('trial').unique()

    l = []
    for n, df in df_trl.groupby(col):
        # all trials belonging to this lick group
        trls = df.loc[:, 'trial'].unique()
        # and only those that are also in the ramp mode
        trls = trls[np.isin(trls, ramp_trl)]

        df_n = ds_ramp.loc[trls].reset_index()
        if not df_n.empty:
            df_n.loc[:, col] = n
            l.append(df_n)
        else:
            print(f'INFO Not trials found in `ds_ramp` for grop {n}')

    df = pd.concat(l, ignore_index=True) 
    df.rename(columns={0: 'ramp_activity'}, inplace=True)

    return df

def get_ramp_mode(X, df_trl, dt=0.2, atol=1e-10, group=None):
    '''Calculate ramp mode

    For each trial, find the linear combination of units that
    maximizes the difference between the mean activity `dt` before
    the cue and the first lick.

    Parameters
    ----------
    X : pandas.DataFrame
        binned spikes
    df_trl : pandas.DataFrame
        trial information
    dt : float, optional
        Time interval in seconds, by default 0.2
    atol : float, optional
        Floating point tolerance for interval detection, by default 1e-10
    group : str, optional
        Column in `df_trl` to add to output, by default None

    Returns
    -------
    df_ramp : pandas.DataFrame
        Ramp mode with columns trial, bin, ramp_mode (and optional: group)
    '''

    df = X.reset_index()
    i1 = pd.Series(index=df.index, data=False)
    i2 = i1.copy()

    for _, row in df_trl.iterrows():
        trl, cue, lck = row.loc[ ['trial', 'dt_cue', 'dt_lck'] ]

        m_trl = df.loc[:, 'trial'].eq(trl)
        m_cue = df.loc[:, 'bin'].between(cue - dt - atol, cue + atol, inclusive='both')
        m_lck = df.loc[:, 'bin'].between(lck - dt - atol, lck + atol, inclusive='both')

        i1.loc[ m_trl & m_cue ] = True
        i2.loc[ m_trl & m_lck ] = True
    
    df1 = df.loc[ i1, : ].drop(columns='bin')
    df2 = df.loc[ i2, : ].drop(columns='bin')

    mean1 = df1.groupby('trial').mean()
    mean2 = df2.groupby('trial').mean()
    diff = mean2 - mean1

    ds_ramp = X.multiply(diff).sum(axis=1)
    ds_ramp.name = 'mode_activity'
    df_ramp = ds_ramp.reset_index()

    if group is not None:
        trl2grp = pd.Series(index=df_trl.loc[:, 'trial'].values, data=df_trl.loc[:, 'lick_group'].values)
        df_ramp.loc[:, group] = df_ramp.loc[:, 'trial'].map(trl2grp).cat.remove_unused_categories()

    return df_ramp