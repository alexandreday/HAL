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
    model = CLUSTER(nh_size=20, n_cluster_init=20, angle=0.5,root='/Users/alexandreday/GitProject/VAC/flowCAP/info/')
    np.random.seed(0)
    run(model, X)
    tmp(model)


def run(model, X):
    import time
    model.fit(X)
    exit()
    model.load_clf()
    s =time.time()
    ypred = model.predict(X, cv=0.985)
    print('Elapsed: \t', time.time() - s)
    np.savetxt('ypred.txt', ypred, fmt='%i')

def tmp(model:CLUSTER):

    xtsne = pickle.load(open(model.file_name['tsne'],'rb'))
    ypred = np.loadtxt('ypred.txt', dtype=int)
    ytrue = load_manual_gate()
    plotting.cluster_w_label(xtsne, ytrue, psize=5,w_legend=True)

if __name__ == "__main__":
    main()