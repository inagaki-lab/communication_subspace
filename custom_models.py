import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


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

        C = np.cov(X, rowvar=False)        # covariance matrix
        val, vec = np.linalg.eig(C.T)         # diagonalize
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
    