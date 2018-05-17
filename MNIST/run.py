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

    #X, y = dataset.load_mnist()
    #Xpca = PCA(n_components=40).fit_transform(X)

    model = CLUSTER(
        nh_size=40, n_cluster_init=30, n_iteration_tsne = 1000, angle=0.5, 
        plot_inter=False,
        run_tSNE='auto',
        try_load = False,
        root='/Users/alexandreday/GitProject/VAC/MNIST/info/'
    )
    
    #xtsne = model.run_tSNE(Scaler().fit_transform(Xpca))
    #np.savetxt('tsne.txt',xtsne)
    #xtsne=np.loadtxt('tsne.txt')
    #plotting.select_data(xtsne, y.flatten(), X, option='mnist',loop=True)
    #plotting.cluster_w_label(xtsne, y.flatten(), psize=5)
    #exit()
    #plot
    #xtsne = pickle.load(open(model.file_name['tsne'],'rb'))
    #plotting.cluster_w_label(xtsne, y.flatten(), psize=5)
    #exit()

    np.random.seed(0)
    #model.fit(Xpca, clf_type='rf', clf_args = {'class_weight':'balanced','n_estimators': 30})
    #pickle.dump(Xpca, open('tmp.pkl','wb'))
    Xpca = pickle.load(open('tmp.pkl','rb'))

    model.load_clf()
    #model.fit(Xpca, clf_type='rf')#, clf_args = {'class_weight':'balanced','kernel':'linear'})
    ypred = model.predict(Xpca, cv=0.999)
    xtsne = pickle.load(open(model.file_name['tsne'],'rb'))
    plotting.cluster_w_label(xtsne, ypred, psize=5)


if __name__ == "__main__":
    main()