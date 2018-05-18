from sklearn.datasets import make_blobs
from vac import DENSITY_PROFILER
from fdc import plotting, FDC
import numpy as np
from tsne_visual import dataset
from fitsne import FItSNE
from sklearn.decomposition import PCA
import pickle
from sklearn.preprocessing import StandardScaler as Scaler
from MulticoreTSNE import MulticoreTSNE as TSNE

def main():
    np.random.seed(0)
    testMNIST()

    
def testMNIST():


    """ X, y = dataset.load_mnist()
    y = y.flatten()
    pickle.dump(y, open('ymnist.pkl','wb'))

    Xpca = PCA(n_components=40).fit_transform(X)
    #Xpca = Scaler().fit_transform(Xpca)
    #Xtsne=TSNE(perplexity=30,verbose=1).fit_transform(Xpca)
    Xtsne = FItSNE(np.ascontiguousarray(Xpca), perplexity=20, max_iter=1000, late_exag_coeff=2.0, start_late_exag_iter=800)
    pickle.dump(Xtsne,open('tsne4.pkl','wb')) """
    y = pickle.load(open('ymnist.pkl','rb'))
    Xtsne = pickle.load(open('tsne4.pkl','rb'))
    #exit()    np.random.seed(0)
    model_fdc = FDC(eta=1.5, n_cluster_init=30, nh_size=80, test_ratio_size=0.8)
    model_DP = DENSITY_PROFILER(model_fdc, outlier_ratio=0.1, nn_pure_ratio=0.95, min_size_cluster=80).fit(Xtsne)

    plotting.cluster_w_label(Xtsne, y)
    ypred = model_DP.y
    model_DP.check_purity(y) # this can be used to test t-SNE and initial set-up ...
    model_DP.describe()
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