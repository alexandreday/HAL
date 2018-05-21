from hal import HAL
from fdc import plotting
import numpy as np
from tsne_visual import dataset
from sklearn.decomposition import PCA
import pickle
from sklearn.preprocessing import StandardScaler as Scaler

def main():
    testMNIST()

def testMNIST():
    X, y = dataset.load_mnist()
    Xpca = PCA(n_components=40).fit_transform(X)
    pickle.dump(Xpca, open('xpca.pkl','wb'))
    Xpca = pickle.load(open('xpca.pkl','rb'))
    #exit()
    np.random.seed(0)
    model = HAL(root='info/',plot_inter=False)#, clf_type='rf', clf_args={'class_weight':'balanced', 'n_estimators': 30})
    model.fit(Xpca)

    X_tsne = pickle.load(open(model.file_name['tsne'],'rb'))
    ypred = model.predict(Xpca, cv = 0.97)
    print(np.unique(ypred))
    #plotting.cluster_w_label(X_tsne, y.flatten())
    plotting.select_data(X_tsne, ypred, X, option='mnist', loop=True, kwargs={'psize': 8, 'w_label':False})

    exit()
    
    #X_tsne = pickle.load(open('tsne_perp=30_niter=1000_alphaLate=2.0.pkl','rb'))
    #plotting.cluster_w_label(X_tsne, y.flatten())
    #exit()

    #print("here")
    #model.fit(Xpca)
    #exit()


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

if __name__ == "__main__":
    main()