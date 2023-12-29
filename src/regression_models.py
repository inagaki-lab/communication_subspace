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
    '''Reduced-rank regression model.

    Reduced-rank regression is a linear regression model with a 
    low-rank constraint on the weight matrix.
    
    The general steps are:
    1. Fit least-squares model to feature matrix X and target matrix Y:
    X @ Bls = Y
    2. Predict target matrix Yls from X using least-squares weights:
    Yls = X @ Bls
    3. Calculate PCA components of Yls:
    V = PCA(Yls)
    4. Choose first r PCs and constrain Bls to have rank r:
    Vr = V[:, :r]
    Br = Bls @ Vr @ Vr.T

    This is a scikit-learn reimplementation of matlab code from
    https://github.com/joao-semedo/communication-subspace/tree/master


    Attributes
    ----------
    coef_ : ndarray of shape (n_features,) or (n_targets, n_features)
        Weight vector(s).
    intercept_ : float or ndarray of shape (n_targets,)
        Independent term in decision function. Set to 0.0 if
        ``fit_intercept = False``.
    n_features_in_ : int
        Number of features seen during `fit`
    V : ndarray

    Bls : ndarray

    Methods
    -------
    fit(X, Y):
        Fit the model to data matrix X and target(s) Y.
    predict(X):
        Predict based on X using the fitted model.
    '''

    def __init__(self, fit_intercept=True, rank=2):
        '''
        Parameters
        ----------
        fit_intercept : bool, optional
            If true, include additional bias feature, by default True
        rank : int, optional
            Rank to use for the reduced rank regression, by default 2
        '''
        self._fit_intercept = fit_intercept
        self._rank = rank
            

    def _lstsq_prediction(self, X, Y):
        '''Least-squares prediction of Y from X.

        Parameters
        ----------
        X : numpy.ndarray
            N x p feature array
        Y : numpy.ndarray
            N x q target array

        Returns
        -------
        Yls : numpy.ndarray
            N x q predicted target array
        B : numpy.ndarray
            p x q weight matrix
        '''

        # least-squares solution
        res = np.linalg.lstsq(X, Y, rcond=None) # lstsq(a, b) solves a * x = b
        Bls = res[0]
        Yls = X @ Bls # predict targets 

        return Yls, Bls
    
    def _get_pca_components(self, X):
        '''PCA components for X.

        Parameters
        ----------
        X : numpy.ndarray
            N x p feature array

        Returns
        -------
        V : numpy.ndarray
            p x p matrix with PCA components as columns
        '''
        'calulate PCA components (columns of matrix V) for X (rows: observations, cols: variables)'

        C = np.cov(X, rowvar=False)           # covariance matrix
        val, vec = np.linalg.eigh(C.T)        # diagonalize
        V = vec[:, np.argsort(val)[::-1]]     # ensure descending components

        # PCA alternatives: sklearn.decomposition.PCA or np.linalg.svd

        return V
    
    def _add_bias(self, X):
        '''Add bias column to X.

        Parameters
        ----------
        X : numpy.ndarray
            N x p feature array

        Returns
        -------
        X_bias : numpy.ndarray
            N x (p+1) feature array with bias column
        '''
        'add bias (ones) to an N x p feature array'

        bias = np.ones_like(X[:, :1])
        X_bias = np.concatenate([bias, X], axis=1)

        return X_bias

  
    def fit(self, X, Y):
        '''Fit reduced-rank regression model.

        Parameters
        ----------
        X : numpy.ndarray
            N x p feature array
        Y : numpy.ndarray
            N x q target array

        Returns
        -------
        self : object
            Fitted estimator.

        Raises
        ------
        ValueError
            If requested rank is larger than maximum possible rank.
        '''
        # checks
        X, Y = check_X_y(X, Y, multi_output=True)
        self.n_features_in_ = X.shape[1]

        self._max_rank = Y.shape[1]
        if self._rank > self._max_rank:
            raise ValueError(f'requested rank {self._rank} is larger than maximum possible rank {self._max_rank}')

        # add bias feature
        if self._fit_intercept:
            X = self._add_bias(X)

        # get V and Bls
        Yls, self.Bls = self._lstsq_prediction(X, Y)
        self.V = self._get_pca_components(Yls)

        # do reduced rank regression
        Vr = self.V[:, :self._rank]     # first r principle components
        Br = self.Bls @ Vr @ Vr.T       # enforcing low rank on B_ls

        # store results
        if self._fit_intercept:
            self.intercept_ = Br[0, :]
            self.coef_ = Br[1:, :]
        else:
            self.intercept_ = 0.0
            self.coef_ = Br

        return self
    
    
    def predict(self, X):
        '''Predict based on X using the fitted model.

        Parameters
        ----------
        X : numpy.ndarray
            N x p feature array

        Returns
        -------
        Y_pred : numpy.ndarray
            N x q predicted target array
        '''
        check_is_fitted(self)
        X = check_array(X)

        if self.fit_intercept:
            X = self._add_bias(X)
        
        B_r = np.concatenate([np.expand_dims(self.intercept_, axis=0), self.coef_], axis=0)
        Y_pred = X @ B_r

        return Y_pred
    

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
        { 'mod__rank': r, },
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