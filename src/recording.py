
import pandas as pd
import numpy as np

from scipy.io import loadmat
from pathlib import Path

import matplotlib.pylab as plt
plt.rcParams['savefig.facecolor'] = 'w'

class Recording:
    '''Class to load data for a single recording stored in a matlab file

    Attributes
    ----------
    df_trl : pandas.DataFrame 
        Trial info, see self._load_trial_info
    df_unt : pandas.DataFrame 
        unit info, see self._load_unit_info
    df_spk : pandas.DataFrame 
        spike times, see self._load_spike_times
    trials : set of int
        indices for available trials (1-based)
    units : set of int
        indices for ailable unit/neuron (1-based)

    Notes
    -----
    The pandas.DataFrames are stored in the temporary folder `tmp_dir` and
    are loaded from disk if they exist and `force_overwrite` is False.
    '''

    def __init__(self, matlab_file, tmp_dir='tmp', force_overwrite=False):
        '''Load data from matlab file and store in pandas.DataFrames

        Parameters
        ----------
        matlab_file : path-like
            Path to matlab data file
        tmp_dir : path-like, optional
            Name of the folder to store processed data, by default 'tmp'
            Folder is created in the same folder as `matlab_file` and a subfolder
            with the name of the session is created inside `tmp_dir`.
        force_overwrite : bool, optional
            If True, do not load data from `tmp_dir`, by default False
        '''

        self.path_mat = Path(matlab_file)
        self.force_overwrite = force_overwrite

        # define session name
        self.session = self.path_mat.with_suffix('').name
        
        # temporary folder, used to recycle precalculated data
        self._path_tmp = self.path_mat.parent / '{}/{}'.format(tmp_dir, self.session)
        self._path_tmp.mkdir(exist_ok=True, parents=True)

        # load trial info
        self.path_trl = self._path_tmp / 'trl.parquet'
        self.df_trl = self._assign_df(self.path_trl, self._load_trial_info)
        self.trials = { *self.df_trl.loc[:, 'trial'] }

        # load unit info
        self.path_unt = self._path_tmp / 'unt.parquet'
        self.df_unt = self._assign_df(self.path_unt, self._load_unit_info)
        self.units = { *self.df_unt.loc[:, 'unit'] }

        # load spike times
        self.path_spk = self._path_tmp / 'spk.parquet'
        self.df_spk = self._assign_df(self.path_spk, self._load_spike_times)


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
            self._matlab = loadmat(self.path_mat, squeeze_me=True, struct_as_record=False)

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


    def _load_trial_info(self, dt=2):
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
            Time in seconds before cue and after first lick, by default 2

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

            ws = vars(u)['SpikeWidth'] 

            d = pd.DataFrame(data={
                'unit': i+1,
                'spike_width': ws,
                'first_trial': ti,
                'last_trial': tf,
            }, index=[i])
            df = pd.concat([df, d])
        
        return df
        

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