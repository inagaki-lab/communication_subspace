import numpy as np
import pandas as pd


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