
import pandas as pd
import numpy as np

from scipy.io import loadmat
from pathlib import Path

class BaseProbe:
    '''Class to load data for a single matlab file

    Attributes
    ----------
    df_trl : pandas.DataFrame 
        Trial info, see self._load_trial_info
    df_unt : pandas.DataFrame 
        unit info, see self._load_unit_info
    df_spk : pandas.DataFrame 
        spike times, see self._load_spike_times

    Notes
    -----
    The pandas.DataFrames are stored in the temporary folder `tmp_dir` and
    are loaded from disk if they exist and `force_overwrite` is False.
    '''

    def __init__(self, matlab_file, bin_size, trial_groups=None, tmp_dir='tmp_data', force_overwrite=False):
        '''Load data from matlab file and store in pandas.DataFrames

        Parameters
        ----------
        matlab_file : path-like
            Path to matlab data file
        bin_size : float
            Bin size in seconds
        trial_groups : (str, list of float), optional
            Name of the column to group by and list of times to construct intervals in seconds relative to cue
            If None, intervals > 0 and < 0 are constructed, by default None
        tmp_dir : path-like, optional
            Name of the folder to store processed data, by default 'tmp'
            Folder is created in the same folder as `matlab_file` and a subfolder
            with the name of the basename is created inside `tmp_dir`.
        force_overwrite : bool, optional
            If True, do not load data from `tmp_dir`, by default False
        '''

        self.bin_size = bin_size
        self._path_mat = Path(matlab_file)
        self.force_overwrite = force_overwrite

        # define basename name
        self.basename = self._path_mat.with_suffix('').name
        
        # temporary folder, used to recycle precalculated data
        self._path_tmp = self._path_mat.parent / '{}/{}'.format(tmp_dir, self.basename)
        self._path_tmp.mkdir(exist_ok=True, parents=True)

        # load trial info
        self._path_trl = self._path_tmp / 'trl.parquet'
        self.df_trl = self._assign_df(self._path_trl, self._load_trial_info)

        # add time groups column
        col, times = ('dt_cue', [0]) if trial_groups is None else trial_groups
        self._add_trial_groups(col, times)

        # load unit info
        self._path_unt = self._path_tmp / 'unt.parquet'
        self.df_unt = self._assign_df(self._path_unt, self._load_unit_info)

        # load spike times
        self._path_spk = self._path_tmp / 'spk.parquet'
        self.df_spk = self._assign_df(self._path_spk, self._load_spike_times)

        # bin spike times
        self.path_bin = self._path_tmp / f'bin{bin_size}.hdf'
        self.df_bin = self._assign_df(self.path_bin, self._bin_spikes)

        
    def _assign_df(self, path, function, function_args=dict()):
        '''Load data from `path` if it exists, otherwise run `function` and save to `path`.

        Can handle storing and loading pandas.DataFrame in `.parquet` and `.hdf` format.

        Parameters
        ----------
        path : path-like
            Path to file to load or save. Must have a known extension.
        function : callable
            Function to run if `path` does not exist. Must return a pandas.DataFrame
        function_args : dict, optional
            Additional arguments to pass to `function`, by default dict()

        Returns
        -------
        df : pandas.DataFrame
            Data loaded from `path` or returned by `function`

        Raises
        ------
        NotImplementedError
            If `path` does not have a known extension
        '''

        if path.suffix == '.parquet':
            read = lambda path: pd.read_parquet(path)
            write = lambda df, path: df.to_parquet(path, compression='brotli')

        elif path.suffix == '.hdf':
            read = lambda path: pd.read_hdf(path)
            write = lambda df, path: df.to_hdf(path, key='df', mode='w', complevel=9, complib='bzip2')

        else:
            raise NotImplementedError(f'Does not know how to handle `{path.suffix}` (only implemented `.parquet` and `.hdf`)')

        if path.is_file() and not self.force_overwrite:
            df = read(path)
        else:
            df = function(**function_args)
            write(df, path)

        return df


    def _load_matlab(self):
        '''Load and return matlab file with `scipy.io.loadmat`, if not already loaded'''
        
        try: # check if matlab file has been loaded
            self._matlab
        except AttributeError: # if not, load
            self._matlab = loadmat(self._path_mat, squeeze_me=True, struct_as_record=False)

        return self._matlab


    def _get_trial_range(self):
        '''Define available trial range
         
        Based on Trial_info.Trial_range_to_analyze in matlab file

        Currently, the trial range is limited to 1020 trials.

        Returns
        -------
        trl_min : int
            Index for first available trial
        trl_max : int
            Index for last available trial
        '''

        m = self._load_matlab()

        trl_min, trl_max = np.inf, -np.inf

        for u in m['unit']:
            trl_i, trl_f = vars(vars(u)['Trial_info'])['Trial_range_to_analyze']
            trl_min = np.min([trl_min, trl_i])
            trl_max = np.max([trl_max, trl_f])
            
        trl_max = np.min([trl_max, 1020]) # TODO
        trl_min, trl_max = int(trl_min), int(trl_max)
        
        return trl_min, trl_max
       

    def _load_spike_times(self):
        '''Load spike times from matlab file

        Each row is a spike event with the following information:
        - `t` : spike time in seconds relative to cue onset
        - `unit` : unit index (1-based)
        - `trial` : trial index (1-based)

        Spike times are filtered based on `dt0` and `dtf` for each trial.
        If either `dt0` or `dtf` is nan, all spikes are disregarded.

        Only trials between `first_trial` and `last_trial` 
        as defined in `self.df_unt` are considered.

        Returns
        -------
        df : pandas.DataFrame
            Spike times in long format
        '''
        
        m = self._load_matlab()
        
        # combine SpikeTimes and Trial_idx_of_spike in one data frame
        l = []
        for i, u in enumerate(m['unit']):
            t = vars(u)['SpikeTimes']
            trl = vars(u)['Trial_idx_of_spike']
            
            d = pd.DataFrame(data={
                't'     : t,
                'unit'  : i + 1,
                'trial' : trl,
            })

            l.append(d)

        df = pd.concat(l, ignore_index=True)

        # align spike times to precue and filter based on dt0 and dtf (see _load_trial_info)
        gr = df.groupby('trial')
        l = []
        for i in self.df_trl.index:

            # Note: if either dt_pre, dt0 or dtf is nan
            # all spikes will be disregarded
            # this ignores e.g. early lick or no cue trials
            trl, dt_pre, dt0, dtf = self.df_trl.loc[ i, ['trial', 'dt_pre', 'dt0', 'dtf'] ]
            
            # unaligned spike times
            d = gr.get_group(trl).copy()
            t = d.loc[:, 't']

            # spike times: aligned to PreSample
            t += dt_pre # now everything is aligned to cue

            # filter spike times based on dt0 and dtf
            d = d.loc[ (t > dt0) & (t < dtf )]

            l.append(d)
        
        df = pd.concat(l, ignore_index=True)
    
        # filter trials per unit according to defined trial ranges defined in df_trl
        gr_unt = self.df_unt.groupby('unit')
        l = []
        for unt, d in df.groupby('unit'):

            # get trial ranges per unit
            row = gr_unt.get_group(unt)
            first = row.loc[:, 'first_trial'].item()
            last = row.loc[:, 'last_trial'].item()
            trl_range = np.arange(first, last+1)

            # only keep trials in trial range
            d = d.loc[ d.loc[:, 'trial'].isin(trl_range) ]
            l.append(d)

        df = pd.concat(l, ignore_index=True)     

        return df
    
    def _bin_spikes(self):
        '''Bin spike trains according to `self.bin_size`

        Skips trial with nan as end time.


        Returns
        -------
        df : pandas.DataFrame
            Binned spikes in wide format
        '''

        # get dict with bin size per trial
        trl2bin = dict()
        for trl, t0, tf in self.df_trl.loc[:, ['trial', 'dt0', 'dtf']].itertuples(index=False):
            tf = tf - tf % self.bin_size # clip last bin
            if tf != tf: # skip trials with nan as end
                continue

            # construct bins
            b = np.arange(t0, tf + self.bin_size, self.bin_size)    
            trl2bin[trl] = b

        # define basis for array
        unts = self.df_spk.loc[:, 'unit'].unique()
        i_trl = np.concatenate([ (len(v)-1) * [ k ] for k, v in trl2bin.items() ])
        i_bin = np.concatenate([ v[:-1] for _, v in trl2bin.items() ])

        # initialize array
        X = np.empty((len(i_trl), len(unts)))
        X[:] = np.nan

        # fill array
        for trl, df in self.df_spk.groupby('trial'):

            # get bins
            bins = trl2bin[trl]
            
            # apply bin to each unit
            gr = df.groupby('unit')
            df = gr.apply(lambda x: np.histogram(x.loc[:, 't'], bins)[0]).apply(pd.Series)
            df /= self.bin_size # spikes per bin -> spikes per s

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

    def _add_trial_groups(self, col, times):
        '''Add a column to self.df_trl that specifies the time group for each trial

        Parameters
        ----------
        col : str
            Name of the column in self.df_trl to group by
        times : list of float
            Times to construct intervals in seconds relative to cue
        '''
        
        dt = self.df_trl.loc[:, col]
        t0, tf = dt.min(), dt.max()

        if t0 < times[0]:
            times = [ t0 ] + times
        if tf > times[-1]:
            times = times + [ tf ]

        labels = [ f'{t0:.1f} - {tf:.1f}' for t0, tf in zip(times[:-1], times[1:]) ]

        grp = pd.cut(dt, bins=times, labels=labels, right=True, include_lowest=True)
        grp = grp.cat.add_categories(['no'])
        grp = grp.fillna('no')

        self.df_trl.loc[:, f'{col}_group'] = grp
    
class ZProbe(BaseProbe):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _load_trial_info(self, dt=2.0):
        '''Load trial info from matlab file

        For each trial, the following information is stored:
        - `trial` : trial index
        - `t_pre`, `t_cue`, `t_lck` : time of pre-sample, cue and first lick
        in seconds relative to trigger onset
        -  `dt_pre`, `dt_cue`, `dt_lck` : time of pre-sample, cue and first lick
        in seconds relative to cue onset
        - `dt0`, `dtf` : trial start and end time in seconds relative to cue onset
        - `trial_type` : trial type as defined in `Behavior.stim_type_name` in matlab file
        - `response` : response type as defined in `Behavior.Trial_types_of_response` in matlab file
        - `response_id` : response type as defined in `Behavior.Trial_types_of_response_vector` in matlab file

        Parameters
        ----------
        dt : float, optional
            Time in seconds before cue and after first lick, by default 2.0

        Returns
        -------
        df : pandas.DataFrame
            Trial info
        '''

        m = self._load_matlab()
        beh = vars(vars(m['unit'][0])['Behavior'])
        
        # time events aligned to trigger
        t_cue = beh['Sample_start'] # aligned to trigger
        t_pre = beh['PreSample'] # aligned to trigger
        t_lck = beh['First_lick'] + t_cue # First_lick aligned to cue 

        # aligned to cue
        dt_pre = t_pre - t_cue
        dt_cue = 0
        dt_lck = t_lck - t_cue

        # trial type
        typ = beh['stim_type_name']

        # response types
        i_res = beh['Trial_types_of_response_vector']
        res = beh['Trial_types_of_response']
        
        # trial index
        trl = np.arange(1, len(t_pre) + 1)

        df = pd.DataFrame(data={
            'trial'         : trl,            
            't_pre'         : t_pre,
            't_cue'         : t_cue,
            't_lck'         : t_lck,
            'dt_pre'        : dt_pre,
            'dt_cue'        : dt_cue,
            'dt_lck'        : dt_lck,
            'dt0'           : -dt,
            'dtf'           : dt_lck + dt,
            'trial_type'    : typ,
            'response'      : res,
            'response_id'   : i_res,
        })

        return df
    
    def _load_unit_info(self):
        '''Load unit info from matlab file

        For each unit, the following information is stored:
        - `unit` : unit index (1-based, as in matlab file)
        - `spike_width` : spike width in ms
        - `first_trial`, `last_trial` : first and last available trial
        as defined in `unit.Trial_info.Trial_range_to_analyze` in matlab file

        Returns
        -------
        df : pandas.DataFrame
            Unit info
        '''
        
        m = self._load_matlab()

        df = pd.DataFrame()

        for i, u in enumerate(m['unit']):

            ti, tf = vars(vars(u)['Trial_info'])['Trial_range_to_analyze']
            tf = np.min([1020, tf]) # limit to 1020

            sw = vars(u)['SpikeWidth'] 

            d = pd.DataFrame(data={
                'unit': i+1,
                'spike_width': sw,
                'first_trial': ti,
                'last_trial': tf,
            }, index=[i])
            df = pd.concat([df, d])
        
        return df
    
    
class YProbe(BaseProbe):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_trial_info(self, dt_pre_cue=-2.0, dt_post_rew=3.0):
        '''Load trial info from matlab file

        For each trial, the following information is stored:
        - `trial` : trial index
        - `t_pre`, `t_cue`, `t_rew` : time of pre-sample, cue and reward
        in seconds relative to trigger onset
        -  `dt_pre`, `dt_cue`, `dt_rew` : time of pre-sample, cue and reward
        in seconds relative to cue onset
        - `dt0`, `dtf` : trial start and end time in seconds relative to cue onset
        - `trial_type` : trial type as defined in `Behavior.stim_type_name` in matlab file
        - `response` : response type as defined in `Behavior.Trial_types_of_response` in matlab file
        - `response_id` : response type as defined in `Behavior.Trial_types_of_response_vector` in matlab file
        - `water_small`, `water_large`, `water_ratio` : water delivered in small and large reward and ratio

        Parameters
        ----------
        dt_pre_cue : float, optional
            Trial start relative to cue in seconds , by default -2.0
        dt_post_rew : float, optional
            Trial end relative to reward in seconds, by default 3.0

        Returns
        -------
        df : pandas.DataFrame
            Trial info
        '''

        m = self._load_matlab()
        beh = vars(vars(m['unit'][0])['Behavior'])

        # time events aligned to trigger (= Presample)
        pre_aligned = vars(vars(beh['Time'])['PresampleAligned'])
        t_cue = pre_aligned['Sample']
        t_rew = pre_aligned['Reward']
        
        # aligned to cue (= Sample)
        dt_rew = t_rew - t_cue
        dt_cue = t_cue - t_cue

        # trial type
        typ = beh['stim_type_name']

        # response types
        i_res = beh['Trial_types_of_response_vector']

        # water
        water_small = beh['WaterSmall']
        water_large = beh['WaterLarge']
        water_ratio = beh['WaterRatio']
        
        # trial index
        trl = np.arange(1, len(dt_rew) + 1)

        df = pd.DataFrame(data={
            'trial'         : trl,            
            't_pre'         : 0,
            't_cue'         : t_cue,
            't_rew'         : t_rew,
            'dt_pre'        : -t_cue,
            'dt_cue'        : dt_cue,
            'dt_rew'        : dt_rew,
            'dt0'           : dt_pre_cue,
            'dtf'           : dt_rew + dt_post_rew,
            'trial_type'    : typ,
            'response_id'   : i_res,
            'water_small'   : water_small,
            'water_large'   : water_large,
            'water_ratio'   : water_ratio,
        })

        return df
    
    def _load_unit_info(self):
        '''Load unit info from matlab file

        For each unit, the following information is stored:
        - `unit` : unit index (1-based, as in matlab file)
        - `spike_width` : spike width in ms
        - `first_trial`, `last_trial` : first and last available trial
        as defined in `unit.Trial_info.Trial_range_to_analyze` in matlab file

        Returns
        -------
        df : pandas.DataFrame
            Unit info
        '''
        
        m = self._load_matlab()

        df = pd.DataFrame()

        for i, u in enumerate(m['unit']):

            ti, tf = vars(vars(u)['Trial_info'])['Trial_range_to_analyze']
            tf = np.min([1020, tf]) # limit to 1020

            sw = vars(u)['SpikeWidth'] 
            ontology = vars(u)['Ontology'] 
            ontology_id = vars(u)['OntID'] 
            hemisphere = vars(u)['Hemisphere']
            area_code = vars(u)['area_code']
            good_unit = vars(u)['GoodUnits']

            d = pd.DataFrame(data={
                'unit': i+1,
                'spike_width': sw,
                'first_trial': ti,
                'last_trial': tf,
                'hemisphere': hemisphere,
                'ontology': ontology,
                'ontology_id': ontology_id,
                'area_code': area_code,
                'good_unit': good_unit,
            }, index=[i])
            df = pd.concat([df, d])
        
        return df