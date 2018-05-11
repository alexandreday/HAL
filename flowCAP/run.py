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
    X = df.values
    #X = Scaler().fit_transform(X)

    # Run vac clustering
    model = CLUSTER(nh_size=20, n_cluster_init=20)
    model.fit(X)
    ypred = model.predict(X)






if __name__ == "__main__":
    main()