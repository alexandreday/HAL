"""
05/11/2018
Script for analyzing FLOWCAP dataset using VAC
"""
import fcsparser
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler as Scaler
from hal import HAL, metric
from fdc import plotting
import pickle

def load_raw_data(file_no=1):
    fname = "{:0>3}.fcs".format(file_no)
    root = '/Users/alexandreday/Dropbox/Work/Project_PHD/Immunology/Visit/FLOWCAP/Data/FCM/fcs/NDD/FCS/'
    path = root+fname
    meta, data = fcsparser.parse(path, reformat_meta=True)
    return data

def load_manual_gate(file_no=1):
    fname = "{:0>3}.csv".format(file_no)
    root = '/Users/alexandreday/Dropbox/Work/Project_PHD/Immunology/Visit/FLOWCAP/Data/Labels/NDD/'
    path = root + fname
    return np.loadtxt(path, dtype=int, skiprows=1)

def main():
    
    # Load and standardize data
    df = load_raw_data(file_no=1)
    X = df.values#[:,2:]
    #X = Scaler().fit_transform(X)
    # Run vac clustering

    model = HAL(
        nh_size=100, min_size_cluster=30, 
        n_cluster_init=30,
        plot_inter=True,
        root='/Users/alexandreday/GitProject/HAL/flowCAP/info/',
        late_exag=900,
        alpha_late=2.0,
        clf_type = 'svm',
        #clf_args = {'class_weight':'balanced','max_features':40}
    )

    np.random.seed(0)

    model.fit(X)

    model.load()

    ypred = model.predict(X, cv=0.96)

    Xtsne = model.load('tsne')

    ytrue = load_manual_gate()

    idx = np.where(ytrue > 0)[0]

    m1, m2 = metric.FLOWCAP_score(ytrue[idx], ypred[idx]), metric.HUNG_score(ytrue[idx], ypred[idx])

    plotting.cluster_w_label(Xtsne, ypred, psize=5, title="%.3f, %.3f"%(m1[0],m2[0]))#, w_legend=True)

    plotting.cluster_w_label(Xtsne, ytrue, psize=5, title="%.3f, %.3f"%(m1[0],m2[0]))#, w_legend=True)
    
    #run(model, X)
    #tmp(model)


def run(model, X):
    import time
    model.fit(X)
    exit()
    model.load_clf()
    s = time.time()
    ypred = model.predict(X, cv=0.985)
    print('Elapsed: \t', time.time() - s)
    np.savetxt('ypred.txt', ypred, fmt='%i')

""" def tmp(model:CLUSTER):

    xtsne = pickle.load(open(model.file_name['tsne'],'rb'))
    ypred = np.loadtxt('ypred.txt', dtype=int)
    ytrue = load_manual_gate()
    plotting.cluster_w_label(xtsne, ytrue, psize=5, w_legend=True) """

if __name__ == "__main__":
    main()