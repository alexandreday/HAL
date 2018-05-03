from sklearn.datasets import make_blobs
import numpy as np
from matplotlib import pyplot as plt
import pickle
from vac import CLUSTER
from fdc import plotting
from collections import Counter

def main():
    root = '/Users/alexandreday/GitProject/VAC/benchmark/b1_results/'
    
    np.random.seed(0)
    X, ytrue = make_blobs(n_samples=5000, n_features=30, centers=10)
    model = CLUSTER(root='b1_results/', run_tSNE='auto', plot_inter=False)
    tree, scaler = model.fit(X)

    ypred = tree.predict(scaler.transform(X))
    pickle.dump(ypred, open(root+'ypred.pkl','wb'))
    ypred = pickle.load(open(root+'ypred.pkl','rb'))
    
    #xtsne = pickle.load(open(root+'tsne_perp=50_niter=1000_outratio=0.20_pureratio=0.99_minsize=0_nhsize=40_eta=1.50_testfdcsize=0.80.pkl','rb'))
    
    print('True labels')
    plotting.cluster_w_label(xtsne, ytrue)

    print('Predicted labels')
    plotting.cluster_w_label(xtsne, ypred)


if __name__ == "__main__":
    main()