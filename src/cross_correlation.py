import numpy as np
import pandas as pd
from tqdm.auto import tqdm

def _positive_correlation(a, b, lag):

    n = np.sqrt( np.sum(a**2) * np.sum(b**2))
    b = np.pad(b, (0, lag), mode='constant')
    c = np.correlate(a, b, mode='valid')
    c = c / n

    return c

def _cross_correlation(X, Y, lag=0):
    
    corr = np.empty((lag+1, len(X.columns), len(Y.columns)), dtype=float)

    for ix, ux in tqdm(enumerate(X), desc=f'X units -> {len(Y.columns)} Y units', total=len(X.columns), leave=False):
        x = X.loc[:,ux].values
        for iy, uy in enumerate(Y):
            y = Y.loc[:,uy].values
            c = _positive_correlation(x, y, lag)
            corr[:, ix, iy] = c

    l = []
    for lag, corr_lag in enumerate(corr):
        
        df = pd.DataFrame(corr_lag, index=X.columns, columns=Y.columns)
        df = df.stack(level=[0, 1])
        df = df.reset_index().rename(columns={
            'level_0': 'probe_X',
            'level_1': 'unit_X',
            'level_2': 'probe_Y',
            'level_3': 'unit_Y',
            0: 'cc'})
        df.loc[:, 'lag'] = lag
        l.append(df)

    df = pd.concat(l, ignore_index=True)
    
    return df

def calculate_cross_correlation(X, Y, lag=0, n_shuffle=0, random_seed=42):

    rng = np.random.default_rng(seed=random_seed)
    trials = Y.index.get_level_values('trial').unique().values

    l = []
    for i in tqdm(range(n_shuffle + 1), desc=f'Shuffle', total=n_shuffle + 1):
        if i == 0:
            df = _cross_correlation(X, Y, lag)
        else:
            rng.shuffle(trials)
            df = _cross_correlation(X, Y.loc[trials, :], lag)
        df.loc[:, 'shuffle'] = i
        l.append(df)
    df = pd.concat(l, ignore_index=True)
    return df
        
