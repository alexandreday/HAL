""" Benchmark for computing scores
Here we just reproduce F1-score (flowCAP and Samusik score) for different FLOWCAP submissions
"""

import fcsparser
import numpy as np
import pandas as pd


def load_raw_data(file_no=1):
    fname = "{:0>3}.fcs".format(file_no)
    root = '/Users/alexandreday/Dropbox/Work/Project_PHD/Immunology/Visit/FLOWCAP/Data/FCM/fcs/NDD/FCS/'
    path = root+fname
    meta, data = fcsparser.parse(path, reformat_meta=True)
    return data
    #data = data.loc[data['sample'].astype(int) == sample_number]
    #data = data.loc[data['label'] != np.nan] # nan are outliers or unlabelled data (... note data is already preprocessed)
    #data = data.dropna(axis=0, how='any')

def load_submission(method=0, file_no=1):
    fname = "{:0>3}.csv".format(file_no)
    method_d = {
        0: "ADICyt/",
        1: "FLOCK/",
        2: "flowMeans/",
        3: "flowMerge/",
        4: "SamSPECTRAL/",
        5: "SWIFT/"
    }
    root = '/Users/alexandreday/Dropbox/Work/Project_PHD/Immunology/Visit/FLOWCAP/Data/Submissions/CH1/'
    path = root+method_d[method]+"NDD/"+fname
    #print(path)
    #exit()
    return np.loadtxt(path,dtype=int,skiprows=1)

def load_manualgate(file_no=1):
    fname = "{:0>3}.csv".format(file_no)

    root = '/Users/alexandreday/Dropbox/Work/Project_PHD/Immunology/Visit/FLOWCAP/Data/Labels/NDD/'
    path = root + fname

    return np.loadtxt(path,dtype=int,skiprows=1)
    

#def load_data('
def main():
    file_no = 1

    ytrue = 
    ypred = load_submission(file_no=file_no)




    #print(load_raw_data(10).describe())
    #print(load_submission())
    #exit()




if __name__ == "__main__":
    main()
