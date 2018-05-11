""" Benchmark for computing scores
Here we just reproduce F1-score (flowCAP and Samusik score) for different FLOWCAP submissions
"""

import fcsparser
import numpy as np
import pandas as pd
from vac import metric


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
    print(method_d[method])
    root = '/Users/alexandreday/Dropbox/Work/Project_PHD/Immunology/Visit/FLOWCAP/Data/Submissions/CH1/'
    path = root+method_d[method]+"NDD/"+fname

    tmp = pd.read_csv(path)
    if method == 4:
        print('NA counts:\t', np.count_nonzero(np.isnan(tmp['component.of']) == True))
        tmp[np.isnan(tmp['component.of']) == True] = -1 # >> NAN DATA ...
        tmp = tmp.astype(int)

    return tmp.values

def load_manual_gate(file_no=1):
    fname = "{:0>3}.csv".format(file_no)
    root = '/Users/alexandreday/Dropbox/Work/Project_PHD/Immunology/Visit/FLOWCAP/Data/Labels/NDD/'
    path = root + fname
    return np.loadtxt(path, dtype=int, skiprows=1)
    
def main():
    # Loading manual gates, etc. 

    method = 5
    file_no = 15

    ytrue = load_manual_gate(file_no=file_no)
    ypred = load_submission(method=method, file_no=file_no)

    pos = (ytrue != 0) # remove ungated cells

    ytrue = ytrue[pos]
    ypred = ypred[pos]
    s1, m1 = metric.FLOWCAP_score(ytrue, ypred)
    s2, m2 = metric.HUNG_score(ytrue, ypred)
    print(s1,'\t', s2)

    exit()
    #print(ytrue)
    #print(np.unique(ypred))
    #exit()

    #print(ypred)
    #print(ypred.iloc[158:165])
    #exit()
    #print(ypred['component.of'].iloc[160:165])
    #exit()
    #print(ypred[ypred['component.of'] != 'NA'])
    #data.loc[data['label'] != np.nan]
    #exit()
    #print(ypred[ypred == 1.0])
    #print(ypred)
    #exit()

    # remove ungated cells

    pos = (ytrue != 0)
    ytrue = ytrue[pos]
    ypred = ypred[pos]

    fs = metric.FLOWCAP_score(ytrue, ypred)
    hs = metric.HUNG_score(ytrue, ypred)
    print(fs)
    print(hs)


    #print(load_raw_data(10).describe())
    #print(load_submission())
    #exit()




if __name__ == "__main__":
    main()
