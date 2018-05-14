"""
05/11/2018
Script for analyzing FLOWCAP dataset using VAC
"""
import fcsparser
import numpy as np
import pandas as pd
from vac import metric
from sklearn.preprocessing import StandardScaler as Scaler
from vac import CLUSTER
from fdc import plotting
import pickle
from tsne_visual import dataset
from sklearn.decomposition import PCA

def main():

    X, y = dataset.load_mnist()
    Xpca = PCA(n_components=40).fit_transform(X)

    model = CLUSTER(
        nh_size=40, n_cluster_init=20, n_iteration_tsne = 1000,
        angle=0.5, 
        plot_inter=True,
        run_tSNE='auto',
        root='/Users/alexandreday/GitProject/VAC/MNIST/info/'
    )

    xtsne = pickle.load(open(model.file_name['tsne'],'rb'))
    plotting.cluster_w_label(xtsne, y.flatten(), psize=5)
    exit()


    np.random.seed(0)
    model.fit(Xpca)
    ypred = model.predict(Xpca, cv=0.97)

    xtsne = pickle.load(open(model.file_name['tsne'],'rb'))
    plotting.cluster_w_label(xtsne, ypred, psize=5)


if __name__ == "__main__":
    main()