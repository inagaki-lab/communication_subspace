import numpy as np
import pandas as pd

# custom model
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

# hyperparameter search
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline


class RRRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, fit_intercept=True, r=2):

        self.fit_intercept = fit_intercept
        self.r = r
        # self.is_precal_ = False
        # if (X_pre is not None) and (Y_pre is not None):
        #     self._precalc(X_pre, Y_pre)
            

    def _lstsq_prediction(self, X, Y):

        # least-squares solution
        res = np.linalg.lstsq(X, Y, rcond=None) # lstsq(a, b) solves a * x = b
        B = res[0]
        Y_p = X @ B # predict targets 

        return Y_p, B
    
    def _get_pca_components(self, X):
        'calulate PCA components (columns of matrix V) for X (rows: observations, cols: variables)'

        # PCA (alternatives: sklearn.decomposition.PCA or np.linalg.svd)

        C = np.cov(X, rowvar=False)           # covariance matrix
        val, vec = np.linalg.eigh(C.T)         # diagonalize
        V = vec[:, np.argsort(val)[::-1]]     # ensure descending components

        # from sklearn.decomposition import PCA
        # pca = PCA()
        # pca.fit(Yf)
        # V = pca.components_

        # U, S, V = np.linalg.svd(Yf)

        # V = loadmat('./communication-subspace-master/V.mat', squeeze_me=True)['V']

        return V
    
    def _add_bias(self, X):
        'add bias (ones) to an N x p feature array'

        bias = np.ones_like(X[:, :1])
        Z = np.concatenate([bias, X], axis=1)

        return Z

    def _precalc(self, X, Y):

        # least-squares solution            
        Yls, self.Bls = self._lstsq_prediction(X, Y)
        
        # PCA
        self.V = self._get_pca_components(Yls)

        self.is_precal_ = True


    def fit(self, X, Y):

        # checks
        X, Y = check_X_y(X, Y, multi_output=True)
        self.n_features_in_ = X.shape[1]

        self.max_rank = Y.shape[1]
        if self.r > self.max_rank:
            raise ValueError(f'requested rank {self.r} is larger than maximum possible rank {self.max_rank}')

        # add bias feature
        if self.fit_intercept:
            X = self._add_bias(X)

        # get V and Bls
        self._precalc(X, Y)

        # do reducted rank regression
        Vr = self.V[:, :self.r]          # first r principle components
        Br = self.Bls @ Vr @ Vr.T        # enforcing low rank on B_ls

        # store results
        self.intercept_ = Br[0, :]
        self.coef_ = Br[1:, :]
        self.is_fitted_ = True

        return self
    
    
    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        if self.fit_intercept:
            X = self._add_bias(X)
        
        B_r = np.concatenate([np.expand_dims(self.intercept_, axis=0), self.coef_], axis=0)
        Y = X @ B_r

        return Y
    

def ridge_regression(dfx_bin, dfy_bin, alphas, scoring=None, n_cv=10, n_jobs=-1):
    '''Ridge regression with cross-validation.

    Return a GridSearchCV object with the fitted models, where
    the best model can be accessed with mods.best_estimator_.

    Parameters
    ----------
    dfx_bin : pandas.DataFrame
        Data for the predictor variables
    dfy_bin : pandas.DataFrame
        Data for the target variables
    alphas : array-like of float
        Regularization parameters to test
    scoring : str or None, optional
        Overwrite score for CV evaluation, by default None
    n_cv : int, optional
        Number of cross-validation folds, by default 10
    n_jobs : int, optional
        Number of jobs to run in parallel.
        If -1, use all available processors, by default -1

    Returns
    -------
    mods : GridSearchCV
        GridSearchCV object with the fitted models
    '''
    
    X, Y = dfx_bin.values, dfy_bin.values

    pipe = Pipeline(steps=[
        ('mod', Ridge())
    ])

    grd = GridSearchCV(
        pipe, 
        { 'mod__alpha': alphas, },
        scoring=scoring,
        cv=n_cv,
        n_jobs=n_jobs,
    )

    mods = grd.fit(X, Y)

    return mods


def reduced_rank_regression(dfx_bin, dfy_bin, max_rank=None, scoring=None, n_cv=10, n_jobs=-1):
    '''Reduced rank regression with cross-validation.

    Return a GridSearchCV object with the fitted models, where
    the best model can be accessed with mods.best_estimator_.

    Parameters
    ----------
    dfx_bin : pandas.DataFrame
        Data for the predictor variables
    dfy_bin : pandas.DataFrame
        Data for the target variables
    max_rank : int or None, optional
        Fit models with ranks up to this value.
        If None, deduce max rank from the number of target variables, by default None.
    scoring : str or None, optional
        Overwrite score for CV evaluation, by default None
    n_cv : int, optional
        Number of cross-validation folds, by default 10
    n_jobs : int, optional
        Number of jobs to run in parallel.
        If -1, use all available processors, by default -1

    Returns
    -------
    mods : GridSearchCV
        GridSearchCV object with the fitted models
    '''

    X, Y = dfx_bin.values, dfy_bin.values

    if max_rank is None:
        max_rank = Y.shape[1]
    r = np.arange(max_rank) + 1

    pipe = Pipeline(steps=[
        ('mod', RRRegressor())
    ])

    grd = GridSearchCV(
        pipe, 
        { 'mod__r': r, },
        scoring=scoring,
        cv=n_cv,
        n_jobs=n_jobs,
    )

    mods = grd.fit(X, Y)

    return mods


def save_cv_results(mods, path):
    '''Save grid search results to disk.

    Parameters
    ----------
    mods : GridSearchCV
        GridSearchCV object with the fitted models
    path : path-like
        Path to save the results
    '''
    
    df = pd.DataFrame(mods.cv_results_)
    
    df.to_parquet(path)


def get_ypred(dfx_bin, dfy_bin, mod, scoring=None, n_cv=10):
    '''Predict activity using source activity `dfx_bin` and fitted model `mod`.

    Returns predicted activity and cross-validation scores per unit.

    `dfy_bin` is necessary for the cross-validation scores.

    Parameters
    ----------
    dfx_bin : pandas.DataFrame
        Data for the predictor variables
    dfy_bin : pandas.DataFrame
        Data for the target variables
    mod : GridSearchCV
        GridSearchCV object with the fitted models
    scoring : str or None, optional
        Overwrite score for CV evaluation, by default None
    n_cv : int, optional
        Number of cross-validation folds, by default 10

    Returns
    -------
    Y_pred : numpy.ndarray
        Predicted activity, same shape as `dfy_bin.values`
    unt2score : dict
        Dictionary with cross-validation scores per unit in `dfy_bin`
    '''
    
    X = dfx_bin.values

    # get prediction
    Y_pred = mod.predict(X)

    # get scores per unit
    cvs = lambda u: cross_val_score(mod, X, dfy_bin.loc[:, u].values, cv=n_cv, scoring=scoring)
    unt2score = { u: cvs(u).mean() for u in dfy_bin }

    return Y_pred, unt2score