from hal import HAL, metric
from fdc import plotting
import numpy as np
from tsne_visual import dataset
from sklearn.decomposition import PCA
import pickle
from sklearn.preprocessing import StandardScaler as Scaler

def main():
    testMNIST()

def testMNIST():
    X, ytrue = dataset.load_mnist()
    ytrue = ytrue.flatten()
    """ Xpca = PCA(n_components=40).fit_transform(X)
    pickle.dump(Xpca, open('xpca.pkl','wb')) """
    Xpca = pickle.load(open('xpca.pkl','rb'))

    np.random.seed(0)

    model = HAL(nh_size=100, min_size_cluster=30, 
        n_cluster_init=30,
        plot_inter=True,
        root='/Users/alexandreday/GitProject/HAL/MNIST/info/',
        late_exag=900,
        alpha_late=2.0,
        clf_type = 'rf',
        clf_args = {'class_weight':'balanced','max_features':40}
    )
    
    model.fit(Xpca)

    model.plot_tree(Xpca,0.9)
    exit()


    model.load()

    X_tsne = model.load('tsne')

    ypred = model.predict(Xpca, cv = 0.98)

    m1, m2 = metric.FLOWCAP_score(ytrue, ypred), metric.HUNG_score(ytrue, ypred)

    plotting.cluster_w_label(X_tsne, ypred, psize=5, title="%.3f, %.3f"%(m1[0],m2[0]))

    plotting.cluster_w_label(X_tsne, ytrue, psize=5, title="%.3f, %.3f"%(m1[0],m2[0]))

    exit()

if __name__ == "__main__":
    main()