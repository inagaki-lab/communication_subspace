
import pandas as pd
import numpy as np

from os.path import commonprefix

class Recording:
    '''Class to merge and analyze data from multiple probes'''

    def __init__(self, probes):
        '''Merge data from multiple BaseProbe instances

        Parameters
        ----------
        probes : dict of str:BaseProbe
            Dict with probe names as keys and probe instances as values

        Raises
        ------
        ValueError
            If probes are not of the same type or if trial info dataframes do not match
        
        Attributes
        ----------
        df_trl : pandas.DataFrame
            Trial info
        df_unt : pandas.DataFrame
            Unit info with 'probe' column added
        df_spk : pandas.DataFrame
            Spike times with 'probe' column added
        df_bin : pandas.DataFrame
            Binned spike times with MultiIndex columns (probe, unit)
        probe_tmp_dir : dict of Path
            Temporary directories for each probe
        session : str
            Session name derived from probe basenames
        '''

        probe_names = list(probes.keys())

        l_unt, l_spk, l_bin = [], [], []
        probe_0 = probes[probe_names[0]]

        for name, probe in probes.items():

            if not isinstance(probe, type(probe_0)):
                raise ValueError(f'Can only merge recordings of type {type(self)}')
            if not probe.df_trl.equals(probe_0.df_trl):
                raise ValueError('Trial info dataframes do not match (rec.df_trl)')
            
            df_unt = probe.df_unt.copy()
            df_unt.loc[:, 'probe'] = name
            l_unt.append(df_unt)

            df_spk = probe.df_spk.copy()
            df_spk.loc[:, 'probe'] = name
            l_spk.append(df_spk)

            df_bin = probe.df_bin.copy()
            df_bin.columns = pd.MultiIndex.from_tuples([(name, unit) for unit in df_bin.columns])
            l_bin.append(df_bin)

        df_unt = pd.concat(l_unt, ignore_index=True)
        df_spk = pd.concat(l_spk, ignore_index=True)
        df_bin = pd.concat(l_bin, axis=1)

        # update attributes if merge is successful
        self.df_unt, self.df_spk, self.df_bin = df_unt, df_spk, df_bin
        self.df_trl = probe_0.df_trl

        self.probe_tmp_dir = { name: probe._path_tmp for name, probe in probes.items() }

        # new session name: either common prefix or concatenation
        pref = commonprefix([ probe.basename for probe in probes.values() ])
        self.session = pref if pref else '+'.join(probes.values())

        # unset attributes that are not valid for merged recordings
        self._matlab = None
        del self._matlab

    def _filter_trial_types(self, type_incl):
        '''Filter trials based on trial type

        Adds a column `incl_trial_type` to `self.df_trl` with boolean values.

        Parameters
        ----------
        type_incl : list of str
            Keep trials in which `trial_type` start with `str`
        '''
        
        tt = self.df_trl.loc[:, 'trial_type']
        if not type_incl:
            type_incl = [''] # include all trials if type_incl is empty

        self.df_trl['incl_trial_type'] = tt.str.startswith(tuple(type_incl))


    def _filter_trials(self, min_max, col):
        '''Filter trials based on if `col` is in between `min_max`.

        Lower limit is inclusive, upper limit is exclusive.

        Adds a column `incl_{col}` to `self.df_trl` with boolean values.

        Parameters
        ----------
        min_max : tuple
            Keep trials in which `col` is between `min_max[0]` and `min_max[1]`.
            If None, no upper/lower limit is applied.
        '''
        
        x_min, x_max = min_max
        
        x = self.df_trl.loc[:, col]
        m_x = x == x # DataSeries with all True for not nan

        if x_min is not None:
            m = x >= x_min
            m_x = m_x & m
        
        if x_max is not None:
            m = x < x_max
            m_x = m_x & m
        
        self.df_trl.loc[~m_x, f'incl_{col}'] = False
        self.df_trl.loc[m_x, f'incl_{col}'] = True
        self.df_trl[f'incl_{col}'] = self.df_trl[f'incl_{col}'].astype(bool)



    def _filter_firing_rate(self, r_min, r_max, df_bin=None):
        '''Filter units based on firing rate.

        Adds a column `incl_firing_rate` to `self.df_unt` with boolean values.

        Parameters
        ----------
        r_min : float or None
            Min firing rate in Hz. If None, no lower limit, by default None
        r_max : float or None
            Max firing rate in Hz. If None, no upper limit, by default None
        df_bin : pandas.DataFrame, optional
            If not None, only keep units in `df_bin`, by default None
        '''

        # get mapping from trial to duration
        df = self.df_trl.loc[:, ['trial', 'dt0', 'dtf']].copy()
        df.loc[:, 'dur'] = df.loc[:, 'dtf'] - df.loc[:, 'dt0']
        ds = df.set_index('trial').loc[:, 'dur']

        # get duration for each unit
        d = dict()
        for p, u, f, l in self.df_unt.loc[:, ['probe', 'unit', 'first_trial', 'last_trial']].itertuples(index=False):
            dur = np.nansum([ ds.loc[i] for i in range(f, l+1) ])
            d[(p, u)] = dur

        # number of spikes per unit
        df = self.df_spk.groupby(['probe', 'unit'], as_index=False).size()

        # average firing rate
        df.loc[:, 'rate'] = df.loc[:, 'size'] / df.loc[:, ['probe', 'unit']].apply(lambda x: d[(x[0], x[1])], axis=1)

        # select units based on rate threshold
        rate = df.loc[:, 'rate']
        m = rate == rate # DataSeries with all `True` if not nan

        if r_min is not None:
            m_min = rate > r_min
            m = m & m_min

        if r_max is not None:
            m_max = rate < r_max
            m = m & m_max

        if df_bin is not None:
            # if df_bin, only set to true for units in df_bin
            m_unt = self.df_unt.set_index(['probe', 'unit']).index.isin(df_bin.columns)
            m = m & m_unt

        self.df_unt.loc[m, 'incl_firing_rate'] = True
        self.df_unt.loc[:, 'incl_firing_rate'].fillna(False, inplace=True)


    def _filter_spike_widths(self, sw_min, sw_max, df_bin=None):
        '''Filter units based on spike width.

        Adds a column `incl_spike_width` to `self.df_unt` with boolean values.

        Parameters
        ----------
        sw_min : float or None, optional
            Min spike width in ms. If None, no lower limit, by default None
        sw_max : float or None, optional
            Max spike width in ms. If None, no upper limit, by default None
        df_bin : pandas.DataFrame, optional
            If not None, only keep units in `df_bin`, by default None
        Returns
        '''
        
        sw = self.df_unt.loc[:, 'spike_width']
        m = sw == sw # DataSeries with all `True` if not nan

        if sw_min is not None:
            m_min = sw > sw_min
            m = m & m_min

        if sw_max is not None:
            m_max = sw < sw_max
            m = m & m_max

        if df_bin is not None:
            # if df_bin, only set to true for units in df_bin
            m_unt = self.df_unt.set_index(['probe', 'unit']).index.isin(df_bin.columns)
            m = m & m_unt

        self.df_unt.loc[m, 'incl_spike_width'] = True
        self.df_unt.loc[:, 'incl_spike_width'].fillna(False, inplace=True)

    def _filter_trial_overlap(self, thresh):
        '''Filter units and trials based on trial overlap.

        Adds columns `incl_trial_overlap` to `self.df_unt` and `self.df_trl`
        with boolean values.

        Adds column `avg_unit_act` to `self.df_trl` with average activity.

        This method determines the fraction of units that are active in each trial,
        then chooses the first and the last trial above `thresh` as the valid trial range.

        Parameters
        ----------
        thresh : float, optional
            Fraction of trials to keep
        '''

        n_trl = self.df_trl.max().loc['trial']
        n_unt = len(self.df_unt)
        activity = np.zeros((n_unt, n_trl), dtype=int)
        for i, row in self.df_unt.iterrows():
            activity[i, row['first_trial'] - 1 : row['last_trial']] = 1
        average_activity = np.mean(activity, axis=0)
        self.df_trl.loc[:, 'avg_unit_act'] = average_activity

        active = np.flatnonzero(average_activity > thresh)
        first_active = active[0] + 1
        last_active = active[-1] + 1

        self.df_trl.loc[:, 'incl_trial_overlap'] = self.df_trl.loc[:, 'trial'].between(first_active, last_active)

        m_first = self.df_unt.loc[:, 'first_trial'] >= first_active
        m_last = self.df_unt.loc[:, 'last_trial'] <= last_active 
        self.df_unt.loc[:, 'incl_trial_overlap'] = m_first & m_last

    def _filter_good_units(self, only_good):
        '''Filter units based on good_unit column.

        Adds columns `incl_good_unit` to `self.df_unt` with boolean values.

        Parameters
        ----------
        only_good : bool
            If True, only keep units with `good_unit` set to 1
        '''

        if only_good:
            self.df_unt.loc[:, 'incl_good_unit'] = self.df_unt.loc[:, 'good_unit'] == 1
        else:
            self.df_unt.loc[:, 'incl_good_unit'] = True

    def _subtract_baseline(self, df_bin, interval):
        '''Subtract baseline firing rate from binned spikes.

        Returns new dataframe with same basis as `df_bin`, 
        but with baseline subtracted.

        Subtracts the mean firing rate during `interval` from
        each unit and trial.

        Parameters
        ----------
        df_bin : pandas.DataFrame
            Binned spikes
        interval : tuple of float, optional
            Interval for baseline calculation in seconds relative to cue, by default (-2, 0)

        Returns
        -------
        df_bin0 : pandas.DataFrame
            Binned spikes with baseline subtracted
        '''

        # select only spikes within `interval`
        t0, tf = interval
        t = self.df_spk.loc[:, 't']
        m = (t < tf) & ( t > t0 )
        df = self.df_spk.loc[m]

        # number of spikes per unit per trial
        df_n = df.groupby(['probe', 'unit', 'trial']).size()
        
        # convert to rate
        dt = tf - t0
        df_r = df_n / dt

        # convert to pivot table
        df_r = df_r.reset_index()
        df_r = pd.pivot_table(data=df_r, index='trial', columns=['probe', 'unit'], values=0)
        # nan implies 0 Hz 
        df_r = df_r.fillna(0)

        # match structure with `df_bin`
        idx = np.unique(df_bin.index.get_level_values(0))
        cols = df_bin.columns
        df_r = df_r.loc[ idx, cols]

        # subtract
        df_bin0 = df_bin.subtract(df_r)

        return df_bin0
        

    def _filter_data(self, params, df_src, df_trg):

        # delete previous columns starting with 'incl_'
        cols_incl = self.df_unt.columns.str.startswith('incl_')
        self.df_unt = self.df_unt.loc[:, ~cols_incl]
        cols_incl = self.df_trl.columns.str.startswith('incl_')
        self.df_trl = self.df_trl.loc[:, ~cols_incl]

        # create at least one column per df_unt and df_trl
        self.df_unt.loc[:, 'incl_all'] = True
        self.df_trl.loc[:, 'incl_all'] = True

        # filter trials
        if 'type_incl' in params:
            self._filter_trial_types(params['type_incl'])
        if 'first_lick' in params:
            self._filter_trials(params['first_lick'], 'dt_lck')
        if 'water_ratio' in params:
            self._filter_trials(params['water_ratio'], 'water_ratio')
        if 'reward_delay' in params:
            self._filter_trials(params['reward_delay'], 'reward_delay')

        # filter units        
        if 'spike_width_src' in params:
            self._filter_spike_widths(*params['spike_width_src'], df_src)
        if 'spike_width_trg' in params:
            self._filter_spike_widths(*params['spike_width_trg'], df_trg)

        if 'rate_src' in params:
            self._filter_firing_rate(*params['rate_src'], df_src)
        if 'rate_trg' in params:
            self._filter_firing_rate(*params['rate_trg'], df_trg)

        if 'only_good' in params:
            self._filter_good_units(params['only_good'])

        # filter trials and units
        if 'trial_overlap' in params:
            self._filter_trial_overlap(params['trial_overlap'])

        # select trials left after filtering
        cols_incl = self.df_trl.columns.str.startswith('incl_')
        df_incl = self.df_trl.loc[:, cols_incl]
        idx_incl = df_incl.all(axis=1)
        self.df_trl.loc[:, 'incl_all'] = idx_incl
        trl_incl = self.df_trl.loc[idx_incl, 'trial']
        trl_incl = trl_incl.loc[ trl_incl.isin(df_src.index.get_level_values(0))]

        # select units left after filtering
        cols_incl = self.df_unt.columns.str.startswith('incl_')
        df_incl = self.df_unt.loc[:, cols_incl]
        idx_incl = df_incl.all(axis=1)
        self.df_unt.loc[:, 'incl_all'] = idx_incl
        unts_incl = self.df_unt.loc[idx_incl, ['probe', 'unit']]
        unts_incl_multi = unts_incl.set_index(['probe', 'unit']).index

        unts_incl_src = df_src.columns.isin(unts_incl_multi)
        unts_incl_trg = df_trg.columns.isin(unts_incl_multi)

        df_src = df_src.loc[ trl_incl, unts_incl_src ]
        df_trg = df_trg.loc[ trl_incl, unts_incl_trg ]

        # discard rows with only nan in both source and targets
        m = df_src.isnull().all(axis=1) & df_trg.isnull().all(axis=1)
        df_src = df_src.loc[ ~m ]
        df_trg = df_trg.loc[ ~m ]

        # fill all remaining nan with 0
        df_src = df_src.fillna(0)
        df_trg = df_trg.fillna(0)

        # subtract baseline
        if params.get('subtract_baseline', False):
            df_src = self._subtract_baseline(df_src, interval=params['baseline_period'])
            df_trg = self._subtract_baseline(df_trg, interval=params['baseline_period'])

        # sort columns
        df_src = df_src.sort_index(axis=1)
        df_trg = df_trg.sort_index(axis=1)

        return df_src, df_trg
    
    def select_data_probes(self, probes_src, probes_trg, params):

        if isinstance(probes_src, str):
            probes_src = [ probes_src ]
        else:
            probes_src = list(probes_src)
        if isinstance(probes_trg, str):
            probes_trg = [ probes_trg ]
        else:
            probes_trg = list(probes_trg)

        # select source and target units
        if set(probes_src) == set(probes_trg):
            # if source and target probes are the same 
            # select random subset of units from that probe(s)
            df_tot = self.df_bin.loc[:, probes_src]

            unts = [ *df_tot.columns ]
            rng = np.random.default_rng(seed=42)
            rng.shuffle(unts)
            s = len(unts) // 2
            unt_src, unt_trg = unts[:-s], unts[-s:]

            df_src = df_tot.loc[:, unt_src]
            df_trg = df_tot.loc[:, unt_trg]
        elif set(probes_src) & set(probes_trg):
            raise ValueError('Some but not all probe(s) are both source and target.')
        else:
            df_src = self.df_bin.loc[:, probes_src]
            df_trg = self.df_bin.loc[:, probes_trg]

        # apply filter
        df_src, df_trg = self._filter_data(params, df_src, df_trg)

        return df_src, df_trg
    
    def _select_units_area_code(self, area_code, df_bin):


        # select only units with given area code 
        unt = self.df_unt.loc[:, 'area_code'].isin(area_code)
        unt_idx = self.df_unt.loc[unt, :].set_index(['probe', 'unit']).index
        m = df_bin.columns.isin(unt_idx)
        df = df_bin.loc[:, m]

        return df

    def select_data_area_code(self, area_code_src, area_code_trg, params):

        if isinstance(area_code_src, int):
            area_code_src = [ area_code_src ]
        else:
            area_code_src = list(area_code_src)
        if isinstance(area_code_trg, int):
            area_code_trg = [ area_code_trg ]
        else:
            area_code_trg = list(area_code_trg)

        # select source and target units
        if set(area_code_src) == set(area_code_trg):
            # if source and target area codes are the same 
            # select random subset of units from that area code(s)
            df_tot = self._select_units_area_code(area_code_src, self.df_bin)

            unts = [ *df_tot.columns ]
            rng = np.random.default_rng(seed=42)
            rng.shuffle(unts)
            s = len(unts) // 2
            unt_src, unt_trg = unts[:-s], unts[-s:]

            df_src = df_tot.loc[:, unt_src]
            df_trg = df_tot.loc[:, unt_trg]

        elif set(area_code_src) & set(area_code_trg):
            raise ValueError('Some but not all area code(s) are both source and target.')
        else:
            # select only units with given area code 
            df_src = self._select_units_area_code(area_code_src, self.df_bin)
            df_trg = self._select_units_area_code(area_code_trg, self.df_bin)

        # apply filter
        df_src, df_trg = self._filter_data(params, df_src, df_trg)

        return df_src, df_trg

    def select_epoch(self, df_bin, epoch):
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
            assert self.df_trl is not None, f'Need df_trl when aligning to {align}'
            
            # map trial to lick time ('dt_lck' is aligned to cue)
            trl2lck = { k: v for k, v in self.df_trl.loc[:, ['trial', 'dt_lck']].itertuples(index=False)}

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
    
    def get_ramp_mode(self, df_bin, t1, t2, atol=1e-10):
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

        Returns
        -------
        df_ramp : pandas.DataFrame
            Ramp mode with columns trial, bin, ramp_mode (and optional: group)
        '''

        df = df_bin.reset_index()
        i1 = pd.Series(index=df.index, data=False)
        i2 = i1.copy()

        al1, dti1, dtf1 = t1
        al2, dti2, dtf2 = t2

        for _, row in self.df_trl.iterrows():
            trl, t_al1, t_al2 = row.loc[ ['trial', al1, al2] ]

            m_trl = df.loc[:, 'trial'].eq(trl)
            m1 = df.loc[:, 'bin'].between(t_al1 + dti1 - atol, t_al1 + dtf1 + atol, inclusive='both')
            m2 = df.loc[:, 'bin'].between(t_al2 + dti2 - atol, t_al2 + dtf2 + atol, inclusive='both')

            i1.loc[ m_trl & m1 ] = True
            i2.loc[ m_trl & m2 ] = True
        
        df1 = df.loc[ i1, : ].drop(columns='bin', level=0)
        df2 = df.loc[ i2, : ].drop(columns='bin', level=0)

        mean1 = df1.groupby('trial').mean()
        mean2 = df2.groupby('trial').mean()
        diff = mean2 - mean1

        ds_ramp = df_bin.multiply(diff).sum(axis=1)
        ds_ramp.name = 'mode_activity'
        df_ramp = ds_ramp.reset_index()

        group_col = self.df_trl.columns.str.endswith('_group')
        if group_col.sum() == 1:
            col = self.df_trl.columns[ group_col ][0]
            trl2grp = pd.Series(index=self.df_trl.loc[:, 'trial'].values, data=self.df_trl.loc[:, col].values)
            df_ramp.loc[:, col] = df_ramp.loc[:, 'trial'].map(trl2grp).cat.remove_unused_categories()

        return df_ramp