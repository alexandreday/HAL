from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from .error import ErrorCheck

def preprocess(X, whiten=False, zscore=True, verbose=True, **kwargs):
    """Function for preprocessing data (X) by optionally whitening the data and performing
    zscore normalization of the features. This is used prior to training classifiers. Returns
    the trained whitening and zscore transformers along with the transformed data.
    """

    param_dict = {
        'whiten':whiten,
        'zscore':zscore,
        'verbose':verbose
    }

    error_obj = ErrorCheck()
    error_obj.check_all_parameters(param_dict)

    if verbose:
        print("Preprocessing data, whiten = %s, zscore = %s"%(str(whiten), str(zscore)))

    pca_transformer = None
    robust_scaler = None

    if whiten: # PCA whitening of the data
        pca_transformer = PCA(whiten=True)
        X = pca_transformer.fit_transform(X)
    if zscore: # this will zscore using the median in the inter-quartile range
        robust_scaler = RobustScaler()
        X = robust_scaler.fit_transform(X)

    result = {
        'X':X,
        'pca_transformer':pca_transformer,
        'robust_scaler':robust_scaler
    }

    return result
