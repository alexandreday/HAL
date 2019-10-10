from sklearn.decomposition import PCA
from .error import raise_error

def preprocess(X, whiten=False, zscore=True, verbose=True, **kwargs):
    if whiten not in [True, False]:
        raise_error(error_type='parameter','whiten', whiten)
    if zscore not in [True, False]:
        raise_error(error_type='parameter','zscore', zscore)
    

    if verbose:
        print("Preprocessing data, whiten = %s, zscore = %s"%(str(whiten), str(zscore)))
    X_tmp = X
    from sklearn.decomposition import PCA
    if whiten is True:
        X_tmp = PCA(whiten=True).fit_transform(X)
    if zscore is True:
        if hasattr(self, 'robust_scaler'):
            X_tmp = self.robust_scaler.transform(X_tmp)
        else:
            self.robust_scaler = RobustScaler()
            X_tmp = self.robust_scaler.fit_transform(X_tmp)
    return X_tmp