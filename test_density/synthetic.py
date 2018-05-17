from sklearn.datasets import make_blobs
from vac import DENSITY_PROFILER
from fdc import plotting, FDC
import numpy as np
from tsne_visual import dataset
from fitsne import FItSNE
from sklearn.decomposition import PCA
import pickle

def main():
    testMNIST()

    
def testMNIST():
    X, y = dataset.load_mnist()
    #y = y.flatten()[:10000]
    #pickle.dump(y, open('ymnist.pkl','wb'))
    y = pickle.load(open('ymnist.pkl','rb'))

    Xpca = PCA(n_components=40).fit_transform(X)[:10000]
    Xtsne = FItSNE(np.ascontiguousarray(Xpca), perplexity=40, late_exag_coeff=4.0,start_late_exag_iter=900)
    pickle.dump(Xtsne,open('tsne2.pkl','wb'))
    Xtsne = pickle.load(open('tsne2.pkl','rb'))
    exit()

    model_fdc = FDC(eta=0.5, n_cluster_init=40, nh_size=40, test_ratio_size=0.2)
    model_DP = DENSITY_PROFILER(model_fdc, outlier_ratio=0.2, nn_pure_ratio=0.9, min_size_cluster=40).fit(Xtsne)

    plotting.cluster_w_label(Xtsne, y)
    ypred = model_DP.y
    model_DP.check_purity(y)

    plotting.cluster_w_label(Xtsne, ypred)

    # check purity w.r.t. to original assignment


def testLOWD():
    np.random.seed(0)
    X, y =  make_blobs(5000, n_features=2, centers = 10)
    model_fdc = FDC(eta=0.,merge=False)
    model_DP = DENSITY_PROFILER(model_fdc, outlier_ratio=0.1, min_size_cluster=360).fit(X)
    y = model_DP.y
    model_DP.describe()
    plotting.cluster_w_label(X, y)



if __name__ == "__main__":
    main()