
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from src.recording import Recording
from src import (
    recording_operations as rec_ops,
    regression_models as reg,
    visualization as vis
)

def load_yml(path):
    """Load config yml as dict.

    Parameters
    ----------
    path: str
        Path to config yml file

    Returns
    -------
    config: dict
        dictionary with multiple parameter sets
    """

    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)

    return cfg

def regression_wrapper(p_out, rec, dfx_bin, dfy_bin, params, epochs):
    
    # check if sufficient units in source population
    n_src = len(dfx_bin.columns)
    if n_src < params['min_units_src']:
        print(f'WARNING: {n_src} units in source region after filtering, skipping analysis')
        return

    if dfx_bin.empty:
        print(f'INFO no data left, skipping rec: {rec.session}')
        return

    # do fit for each epoch separately
    for name, epo in epochs.items():

        # output folder for epoch
        p_out_epo = p_out / name
        p_out_epo.mkdir(exist_ok=True)

        # select subset of data
        dfx_bin_epo = rec.select_epoch(dfx_bin, epo)
        dfy_bin_epo = rec.select_epoch(dfy_bin, epo)

        # save data
        dfx_bin_epo.to_parquet(p_out_epo / 'dfx_bin.parquet')
        dfy_bin_epo.to_parquet(p_out_epo / 'dfy_bin.parquet')

        if dfx_bin_epo.empty:
            print(f'INFO no data left in epoch {name}, skipping rec: {rec.session}')
            continue

        # linear regression (= ridge with alpha=0)
        lin_mods = reg.ridge_regression(dfx_bin_epo, dfy_bin_epo, scoring=params['scoring'], alphas=[0])
        reg.save_cv_results(lin_mods, path=p_out_epo / 'reg_linear.parquet')

        # ridge regression
        ridge_mods = reg.ridge_regression(dfx_bin_epo, dfy_bin_epo, scoring=params['scoring'], alphas=np.logspace(-13, 13, 27))
        ridge_mod = ridge_mods.best_estimator_
        reg.save_cv_results(ridge_mods, path=p_out_epo / 'reg_ridge.parquet')

        # RRR
        rr_mods = reg.reduced_rank_regression(dfx_bin_epo, dfy_bin_epo, scoring=params['scoring'])
        reg.save_cv_results(rr_mods, path=p_out_epo / 'reg_rrr.parquet')

        # plot regressions
        vis.plot_gridsearch(ridge_mods, 'ridge', other_mods={'linear': lin_mods}, logscale=True, path=p_out_epo / 'reg_ridge.png')
        vis.plot_gridsearch(rr_mods, 'reduced rank', other_mods={'linear': lin_mods, 'ridge': ridge_mods}, logscale=False, path=p_out_epo / 'reg_rrr.png')

        # prediction
        Y_pred, scores = reg.get_ypred(dfx_bin_epo, dfy_bin_epo, ridge_mod, scoring=params['scoring'])
        vis.plot_mean_response(dfy_bin_epo, Y_pred, scores, path=p_out_epo / 'pred_ridge.png')
        ds = pd.Series(scores, name=params['scoring'])
        ds.index.name = 'unit'
        ds.to_csv(p_out_epo / 'pred_ridge_scores.csv', index=True)

def processing_wrapper_probes(p_out, probe_name_A, probe_name_B, rec, params, epochs, overwrite):

    # create folder
    p_out.mkdir(exist_ok=True, parents=True)

    # path for params.json
    p_params = p_out / 'params.json'
    if p_params.exists() and not overwrite:
        print(f'params.json found. Skipping {p_out}')
        return 
    
    # load data, select trials and units based on `params`
    dfx_bin, dfy_bin = rec.select_data_probes(probe_name_A, probe_name_B, params=params)

    # do regressions
    regression_wrapper(p_out, rec, dfx_bin, dfy_bin, params, epochs)

    # save params and dataframes
    pd.Series(epochs).to_json(p_out / 'epochs.json')
    pd.Series(params).to_json(p_params)
    rec.df_trl.to_parquet(p_out / 'df_trl.parquet')
    rec.df_unt.to_parquet(p_out / 'df_unt.parquet')

def analyze_interactions_probes(p_dirs, params, probe_class, probe_kwargs=dict(), epochs=dict(), probe_names=dict(), overwrite=False, out_dir='analysis'):
    '''Wrapper for analyzing probe interactions for multiple sessions.

    This creates the subfolders for all possible interactions between two recordings
    and then calls a processing wrapper for each of them.

    If `epochs` is empty, the only epoch will be 'all'.

    If some .mat file is not defined in `probe_names`,
    the probe names will be set to 'proA' and 'proB'.

    Parameters
    ----------
    p_dirs : list of path-like
        Folders with recordings, must contain two .mat files each
    params : dict
        Parameters for data selection and regression
    probe_class : BaseProbe
        Probe object with specific methods for each matlab file structure
    probe_kwargs : dict, optional
        Additional arguments for Probe object, by default dict()
    epochs : dict, optional
        Run separate analyses for each epoch, by default dict()
    probe_names : dict, optional
        Mapping of matlab file names to short probe names, by default dict()
    overwrite : bool, optional
        Overwrite existing analysis, by default False
    out_dir : path-like, optional
        Output folder name, by default 'analysis'
    '''

    print(f'>>>> starting analysis ')
    print(f'>>>> writing output to: {out_dir}')
    print(f'>>>> params: {params}')

    if not epochs: # ensure that at least one epoch is specified
        epochs = {'all': (None, None, 'cue')}
    print(f'>>>> epochs: {epochs}')

    print(f'>>>> now processing recordings ....')
    for p_dir in p_dirs:
        print(f'     {p_dir}')
        p_out = Path(p_dir) / out_dir
        p_out.mkdir(exist_ok=True, parents=True)

        # load recordings
        p_matA, p_matB = [ *p_dir.glob('*.mat')][:2] # get first two .mat files
        probe_A = probe_class(p_matA, bin_size=params['bin_size'], **probe_kwargs)
        probe_B = probe_class(p_matB, bin_size=params['bin_size'], **probe_kwargs)
        try:  # define short names for regions
            probe_name_A, probe_name_B = probe_names[p_matA.name], probe_names[p_matB.name]

        except KeyError:
            probe_name_A, probe_name_B = 'proA', 'proB'
            print(f'WARNING: no probe name found for {p_matA.name} or {p_matB.name}')
            print(f'         using {probe_name_A} and {probe_name_B} instead')
        rec = Recording({
            probe_name_A: probe_A,
            probe_name_B: probe_B
        })

        # plot general info
        vis.plot_trial_infos(rec.df_trl, path=p_out / f'trial_infos.png')

        other_args = {
            'params': params,
            'epochs': epochs,
            'rec': rec,
            'overwrite': overwrite
        }
        # A -> B
        print(f'     now doing {probe_name_A} -> {probe_name_B}')
        processing_wrapper_probes(
            p_out / f'{probe_name_A}_{probe_name_B}', probe_name_A, probe_name_B, **other_args)

        # A -> B
        p_out = p_dir / f'{out_dir}/{probe_name_B}_{probe_name_A}'
        processing_wrapper_probes(
            p_out / f'{probe_name_B}_{probe_name_A}', probe_name_B, probe_name_A, **other_args)

        # A -> A'
        print(f'     now doing within {probe_name_A}')
        processing_wrapper_probes(
            p_out / f'{probe_name_A}', probe_name_A, probe_name_A, **other_args)

        # B -> B'
        print(f'     now doing within {probe_name_B}')
        processing_wrapper_probes(
            p_out / f'{probe_name_B}', probe_name_B, probe_name_B, **other_args)

def processing_wrapper_areas(p_out, areas_A, areas_B, rec, params, epochs, overwrite):

    # create folder
    p_out.mkdir(exist_ok=True, parents=True)

    # path for params.json
    p_params = p_out / 'params.json'
    if p_params.exists() and not overwrite:
        print(f'params.json found. Skipping {p_out}')
        return 
    
    # load data, select trials and units based on `params`
    dfx_bin, dfy_bin = rec.select_data_area_code(areas_A, areas_B, params)

    # do regressions
    regression_wrapper(p_out, rec, dfx_bin, dfy_bin, params, epochs)

    # save params and dataframes
    pd.Series(epochs).to_json(p_out / 'epochs.json')
    pd.Series(params).to_json(p_params)
    rec.df_trl.to_parquet(p_out / 'df_trl.parquet')
    rec.df_unt.to_parquet(p_out / 'df_unt.parquet')


def analyze_interactions_areas(p_dirs, params, probe_class, probe_kwargs=dict(), epochs=dict(), overwrite=False, out_dir='analysis'):
    '''Wrapper for analyzing area interactions for multiple sessions.

    This creates the subfolders for all possible interactions between two recordings
    and then calls a processing wrapper for each of them.

    If `epochs` is empty, the only epoch will be 'all'.

    Parameters
    ----------
    p_dirs : list of path-like
        Folders with recordings, must contain two .mat files each
    params : dict
        Parameters for data selection and regression
    probe_class : BaseProbe
        Probe object with specific methods for each matlab file structure
    probe_kwargs : dict, optional
        Additional arguments for Probe object, by default dict()
    epochs : dict, optional
        Run separate analyses for each epoch, by default dict()
    overwrite : bool, optional
        Overwrite existing analysis, by default False
    out_dir : path-like, optional
        Output folder name, by default 'analysis'
    '''

    print(f'>>>> starting analysis ')
    print(f'>>>> writing output to: {out_dir}')
    print(f'>>>> params: {params}')

    if not epochs: # ensure that at least one epoch is specified
        epochs = {'all': (None, None, 'cue')}
    print(f'>>>> epochs: {epochs}')

    print(f'>>>> now processing recordings ....')
    for p_dir in p_dirs:
        print(f'     {p_dir}')
        p_out = Path(p_dir) / out_dir
        p_out.mkdir(exist_ok=True, parents=True)

        # load recordings
        p_mats = [ *p_dir.glob('*.mat')]
        
        # find differences in file names
        name_parts = [ p.stem.split('_') for p in p_mats ]
        for idx, s in enumerate(name_parts[0]):
            if s not in name_parts[1]:
                break

        probes = [probe_class(p, bin_size=params['bin_size'], **probe_kwargs) for p in p_mats]
        rec = Recording({name_part[idx]: probe for name_part, probe in zip(name_parts, probes)})
        areas_A, areas_B = params['area_code_A'], params['area_code_B']
        # plot general info
        vis.plot_trial_infos(rec.df_trl, path=p_out / f'trial_infos.png')

        other_args = {
            'params': params,
            'epochs': epochs,
            'rec': rec,
            'overwrite': overwrite
        }
        s_A, s_B = ', '.join(map(str, areas_A)), ', '.join(map(str, areas_B))

        # A -> B
        print(f'     now doing areas A -> B ({s_A} -> {s_B})')
        processing_wrapper_areas(p_out / 'A_B', areas_A, areas_B, **other_args)

        # A -> B
        print(f'     now doing areas B -> A ({s_B} -> {s_A})')
        processing_wrapper_areas(p_out / 'B_A', areas_B, areas_A, **other_args)

        # A -> A'
        print(f'     now doing areas A -> A\' ({s_A} -> {s_A})')
        processing_wrapper_areas(p_out / 'A', areas_A, areas_A, **other_args)

        # B -> B'
        print(f'     now doing areas B -> B\' ({s_B} -> {s_B})')
        processing_wrapper_areas(p_out / 'B', areas_B, areas_B, **other_args)

def load_scores(ps_csv):
    '''Load scores from multiple csv files.

    Additinoal info in the output dataframe assumes following folder structure:
    `{probe}/{animal}_{date}/analysis/{parameter_set}/{interaction}/{epoch}/pred_ridge_scores.csv`

    Parameters
    ----------
    ps_csv : list of path-like
        CSV files containing scores created during `processing_wrapper`

    Returns
    -------
    df : pandas.DataFrame
        Scores for each unit in each recording
    '''

    dfs = []
    for p_csv in ps_csv:

        parts = p_csv.parts
        epoch = parts[-2]
        inter = parts[-3]
        setti = parts[-4]
        recor = parts[-6]
        anima, date = recor.split('_')
        probe = parts[-7]

        df_scores = pd.read_csv(p_csv)

        data = {
            'unit':         df_scores.loc[:, 'unit'], # TODO change this for newer data
            'score':        df_scores.iloc[:, 1],
            'epoch':        epoch,
            'interaction':  inter,
            'settings':     setti,
            'recording':    recor,
            'animal':       anima,
            'date' :        date,
            'probes':       probe
        }

        p_prq = p_csv.parent / 'reg_rrr.parquet'
        if p_prq.exists():
            df_ranks = pd.read_parquet(p_prq)
            rrr = reg.analyze_rrr(df_ranks)
            data.update(rrr)
        else:
            print(f'INFO {p_prq} not found, not loading RRR results')

        dfs.append(pd.DataFrame(data=data))
    df = pd.concat(dfs, ignore_index=True)
    df.loc[:, 'interaction_'] = df.loc[:, 'interaction'].map(lambda x: x.replace('ALM1', 'ALM').replace('ALM2', 'ALM'))
    df.loc[:, 'n_regions'] = df.loc[:, 'interaction'].apply(lambda x: len(x.split('_')))

    return df
