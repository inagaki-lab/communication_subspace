
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

def processing_wrapper(p_out, params, epochs, recX, recY):

    # create folder
    p_out.mkdir(exist_ok=True, parents=True)

    # path for params.json
    p_params = p_out / 'params.json'
    p_epochs = p_out / 'epochs.json'
    if p_params.exists():
        print(f'params.json found. Skipping {p_out}')
        return 
    
    # load data, select trials and units based on `params`
    dfx_bin, dfy_bin = rec_ops.select_data(recX, rec2=recY, params=params)

    # subtract baseline, if required
    if params['subtract_baseline']:
        dfx_bin = rec_ops.subtract_baseline(dfx_bin, recX.df_spk)
        dfy_bin = rec_ops.subtract_baseline(dfy_bin, recX.df_spk if recY is None else recY.df_spk)
    if dfx_bin.empty:
        print(f'INFO no data left, skipping recX: {recX.session}, recY: {recY.session}')
        return

    # do fit for each epoch separately
    for name, epo in epochs.items():

        # output folder for epoch
        p_out_epo = p_out / name
        p_out_epo.mkdir(exist_ok=True)

        # select subset of data
        dfx_bin_epo = rec_ops.select_epoch(dfx_bin, epo, recX.df_trl)
        dfy_bin_epo = rec_ops.select_epoch(dfy_bin, epo, recX.df_trl if recY is None else recY.df_trl)

        if dfx_bin_epo.empty:
            print(f'INFO no data left in epoch {name}, skipping recX: {recX.session}, recY: {recY.session}')
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

    # save params
    pd.Series(epochs).to_json(p_epochs)
    pd.Series(params).to_json(p_params)

def analyze_interactions(p_dirs, params, epochs=dict(), probe_names=dict(), out_dir='analysis'):
    '''Wrapper for processing multiple recordings.

    This creates the subfolders for all possible interactions between two recordings
    and then calls `processing_wrapper` for each of them.

    If `epochs` is empty, the only epoch will be 'all'.

    If some .mat file is not defined in `probe_names`,
    the probe names will be set to 'regA' and 'regB'.

    Parameters
    ----------
    p_dirs : list of path-like
        Folders with recordings, must contain two .mat files each
    params : dict
        Parameters for data selection and regression
    epochs : dict, optional
        Run separate analyses for each epoch, by default dict()
    probe_names : dict, optional
        Mapping of matlab file names to short brain regions names, by default dict()
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
        p_dir = Path(p_dir)

        # load recordings
        p_matA, p_matB = [ *p_dir.glob('*.mat')]
        recA, recB = Recording(p_matA), Recording(p_matB)
        try: # define short names for regions
            regionA, regionB = probe_names[p_matA.name], probe_names[p_matB.name]
        except KeyError:
            regionA, regionB = 'regA', 'regB'
            print(f'WARNING: no probe name found for {p_matA.name} or {p_matB.name}')
            print(f'         using {regionA} and {regionB} instead')

        # regionA -> regionB
        p_out = p_dir / f'{out_dir}/{regionA}_{regionB}'
        processing_wrapper(p_out, params, epochs, recA, recB)

        # regionB -> regionA
        p_out = p_dir / f'{out_dir}/{regionB}_{regionA}'
        processing_wrapper(p_out, params, epochs, recB, recA)

        # regionA
        p_out = p_dir / f'{out_dir}/{regionA}'
        processing_wrapper(p_out, params, epochs, recA, None)

        # regionB
        p_out = p_dir / f'{out_dir}/{regionB}'
        processing_wrapper(p_out, params, epochs, recB, None)
        