
import pandas as pd
import numpy as np
from scipy.ndimage import uniform_filter1d

from scipy.io import loadmat
from pathlib import Path

from joblib import Parallel, delayed, parallel_backend

import matplotlib.pylab as plt
plt.rcParams['savefig.facecolor'] = 'w'

class Recording:

    def __init__(self, matlab_file, bin_size=1e-3, force_overwrite=False):

        self.path_mat = Path(matlab_file)
        self.session = self.path_mat.with_suffix('').name

        self._matlab = None
        self._path_name = lambda x : self.path_mat.parent / '{}_{}'.format(self.path_mat.with_suffix('').name, x)

        self.force_overwrite = force_overwrite
        self.bin_size = bin_size

        self.path_trl = self._path_name('trl.parquet')
        self.df_trl = self._assign_df(self.path_trl, self._load_trial_info)
        self.trials = { *self.df_trl.loc[:, 'trial'] }

        self.path_unt = self._path_name('unt.parquet')
        self.df_unt = self._assign_df(self.path_unt, self._load_unit_info)
        self.units = { *self.df_unt.loc[:, 'unit'] }

        self.path_spk = self._path_name('spk.parquet')
        self.df_spk = self._assign_df(self.path_spk, self._load_spike_times)

        self.path_psth = self._path_name('psth.parquet')
        self.df_psth = self._assign_df(self.path_psth, self._calculate_psth)


    def _assign_df(self, path, function, kw=dict()):

        if path.is_file() and not self.force_overwrite:
            df = pd.read_parquet(path)
        else:
            df = function(**kw)
            df.to_parquet(path, compression='brotli')

        return df


    def _load_matlab(self):

        if not self._matlab:
            self._matlab = loadmat(self.path_mat, squeeze_me=True, struct_as_record=False)

        return self._matlab


    def _get_trial_range(self):

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
            'response'      : res,
            'response_id'   : i_res,
        })

        return df

    # def _load_raw_trial_info(self):

    #     m = self._load_matlab()

    #     # sample rate
    #     sample_rate = vars(vars(vars(m['unit'][0])['Meta_data'])['parameters'])['Sample_Rate']

    #     # sample start in absolute time
    #     T_onset = np.array([vars(i)['onset'] for i in m['trial_info']]) / sample_rate

    #     behavior = vars(vars(m['unit'][0])['Behavior'])
    #     t_cue = behavior['Sample_start']
    #     t_pre = behavior['PreSample']
    #     T_cue = T_onset + t_cue - t_pre
    #     T_on = T_onset - t_pre

    #     # valid trial range
    #     trl_min, trl_max = self._get_trial_range()
    #     trl = np.arange(trl_min, trl_max + 1)
    #     i_trl = trl - 1 

    #     df = pd.DataFrame(data={
    #         'trial' : trl,
    #         'T_0'   : T_on[i_trl],
    #         'T_f'   : T_on[i_trl + 1],
    #         'T_cue' : T_cue[i_trl],
    #         't_0'   : 0,
    #         't_f'   : T_on[i_trl+1] - T_on[i_trl],
    #         't_cue' : t_cue[i_trl],
    #     })

    #     df.loc[:, 'dt_0'] = -df.loc[:, 't_cue']
    #     df.loc[:, 'dt_f'] = df.loc[:, 't_f'] - df.loc[:, 't_cue']
    #     df = df.astype({'trial': int})

    #     return df

    def _load_unit_info(self):
        
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
        
        m = self._load_matlab()
        
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

        gr = df.groupby('trial')


        l = []
        for i in self.df_trl.index:

            trl, dt_pre, dt0, dtf = self.df_trl.loc[ i, ['trial', 'dt_pre', 'dt0', 'dtf'] ]
            
            d = gr.get_group(trl).copy()
            t = d.loc[:, 't']

            # spike times: aligned to PreSample
            t += dt_pre # now everything is aligned to cue

            # filter spike times
            d = d.loc[ (t > dt0) & (t < dtf )]

            l.append(d)
        
        df = pd.concat(l, ignore_index=True)
    
        # filter trials per unit according to defined trial ranges
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
    
    # def _load_raw_spike_times(self):
        
    #     m = self._load_matlab()
        
    #     df = pd.DataFrame()

    #     for i, u in enumerate(m['unit']):
    #         t = vars(u)['RawSpikeTimes']

    #         d = pd.DataFrame(data={
    #             'T': t,
    #             'unit': i + 1,
    #         })

    #         df = pd.concat([df, d], ignore_index=True)

        
    #     for i in self.df_trl.index:

    #         trl, T_0, T_f, T_cue = self.df_trl.loc[ i, ['trial', 'T_0', 'T_f', 'T_cue'] ]

    #         idx_trl = df.loc[ (df.loc[:, 'T'] > T_0) & (df.loc[:, 'T'] < T_f) ].index
            
    #         df.loc[idx_trl, 'trial'] = trl
    #         df.loc[idx_trl, 't'] = df.loc[idx_trl, 'T'] - T_0
    #         df.loc[idx_trl, 'dt'] = df.loc[idx_trl, 'T'] - T_cue
            
    #     df = df.dropna(axis=0)

    #     df = df.astype({'trial': int})

    #     return df


    # def fun(self, unt, df, bin_size):

    #     unts, trls, binss, hists = [], [], [], []

    #     for trl, d in df.groupby('trial'):
    #         t0, tf = self.df_trl.groupby('trial').get_group(trl).loc[:, ['dt0', 'dtf']].values[0]
    #         t_spk = d.loc[:, 't'].values

    #         bins = np.arange(int(t0 / bin_size), int(tf / bin_size))
    #         t2hist = lambda t: np.histogram((t / bin_size).astype(int), bins=bins)[0]
    #         hist = t2hist(t_spk)

    #         hists.extend([*hist])
    #         binss.extend([*bins[:-1]])
    #         unts.extend([unt] * len(hist))
    #         trls.extend([trl] * len(hist))

    #     return hists, binss, unts, trls
    
    # def _calculate_psth(self, bin_size=None):

    #     bin_size = bin_size if bin_size else self.bin_size

    #     unts, trls, binss, hists = [], [], [], []

    #     from itertools import chain


        
    #     with parallel_backend('loky', n_jobs=1):
    #         res = Parallel()(
    #             delayed(
    #                 self.fun)(unt, df, bin_size) for unt, df in self.df_spk.groupby('unit'))
    #     print('done')
    #     h, b, u, t = res
    #     hists = chain(h)
    #     binss = chain(b)
    #     unts = chain(u)
    #     trls = chain(t)

    #     data = np.array([unts, trls, binss, hists]).T
    #     df = pd.DataFrame(data, columns=['unit', 'trial', 'bins', 'hist'])
    #     df = df.astype({'unit': int, 'trial': int, 'bins': int})

    #     return df
    
    
    def _calculate_psth(self, bin_size=None):

        bin_size = bin_size if bin_size else self.bin_size

        unts, trls, binss, hists = [], [], [], []

        for (unt, trl), df in self.df_spk.groupby(['unit', 'trial']):
                
            t0, tf = self.df_trl.groupby('trial').get_group(trl).loc[:, ['dt0', 'dtf']].values[0]
            t_spk = df.loc[:, 't'].values

            bins = np.arange(int(t0 / bin_size), int(tf / bin_size))
            t2hist = lambda t: np.histogram((t / bin_size).astype(int), bins=bins)[0]
            hist = t2hist(t_spk)

            hists.extend([*hist])
            binss.extend([*bins[:-1]])
            unts.extend([unt] * len(hist))
            trls.extend([trl] * len(hist))

        data = np.array([unts, trls, binss, hists]).T
        df = pd.DataFrame(data, columns=['unit', 'trial', 'bins', 'hist'])
        df = df.astype({'unit': int, 'trial': int, 'bins': int})

        return df
    
    # def _calculate_raw_psth(self, align='start', bin_size=None):

    #     bin_size = bin_size if bin_size else self.bin_size

    #     if align == 'cue':
    #         c_t, c_t0, c_tf = 'dt', 'dt_0', 'dt_f'
    #     elif align == 'start':
    #         c_t, c_t0, c_tf = 't', 't_0', 't_f'
    #     else:
    #         c_t, c_t0, c_tf = 'T', 'T_0', 'T_f'


    #     unts, trls, binss, hists = [], [], [], []

    #     for (unt, trl), df in self.df_spk.groupby(['unit', 'trial']):
                
    #         t0, tf = self.df_trl.groupby('trial').get_group(trl).loc[:, [c_t0, c_tf]].values[0]
    #         t_spk = df.loc[:, c_t].values

    #         bins = np.arange(int(t0 / bin_size), int(tf / bin_size))
    #         t2hist = lambda t: np.histogram((t / bin_size).astype(int), bins=bins)[0]
    #         hist = t2hist(t_spk)

    #         hists.extend([*hist])
    #         binss.extend([*bins[:-1]])
    #         unts.extend([unt] * len(hist))
    #         trls.extend([trl] * len(hist))

    #     data = np.array([unts, trls, binss, hists]).T
    #     df = pd.DataFrame(data, columns=['unit', 'trial', 'bins', 'hist'])
    #     df = df.astype({'unit': int, 'trial': int, 'bins': int})


    #     return df
    
    
    def plot_psth(self, filter_size=50, xlims=(None, None), unts=None, path=''):

        if unts is None:
            unts = self.df_unt.loc[:, 'unit'].unique()
        n = len(unts)

        fig, axmat = plt.subplots(nrows=n, figsize=(10, 3*n), squeeze=False)
        fig.suptitle('PSTH | moving average {} ms | {}'.format(filter_size, self.session), y=1.0)
        gr_psth = self.df_psth.groupby('unit')
        gr_unt = self.df_unt.groupby('unit')

        axarr = axmat[0]
        for unt, ax in zip(unts, axarr):

            trl_i = gr_unt.get_group(unt).loc[:, 'first_trial'].item()
            trl_f = gr_unt.get_group(unt).loc[:, 'last_trial'].item()
            trls = [ *range(trl_i, trl_f + 1) ]

            df = gr_psth.get_group(unt)
            df = df.loc[ df.loc[:, 'trial'].isin(trls) ]


            ds = df.groupby('bins').mean().loc[:, 'hist'].sort_index()

            x, y = ds.index, ds.values
            x = x * self.bin_size
        
            y = uniform_filter1d(y, size=int(filter_size * 1e-3 / self.bin_size)) / self.bin_size
            if xlims != (None, None):
                mask = ( (x > xlims[0]) & (x < xlims[1]) )
                x, y = x[mask], y[mask]
            ax.plot(x, y)
            ax.set_xlim(xlims)

            ax.axvline(0, color='gray', lw=1, ls='--')
            ax.axhline(0, color='gray', lw=1)

            ax.set_title('unit {} | trial range = [{}, {}]'.format(unt, trl_i, trl_f))
            ax.set_xlabel('time from [s]')
            ax.set_ylabel('rate [Hz]')

        fig.tight_layout()

        if path:
            fig.savefig(path)
            plt.close(fig)