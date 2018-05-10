from sklearn.datasets import make_blobs
import numpy as np
from matplotlib import pyplot as plt
import pickle
from vac import CLUSTER
from fdc import plotting
from collections import Counter
from vac import metric
import subprocess


########################## BENCHMARK ON easy artificial DATASET ################
########################## Important for more involved examples ###############

def main():
    root = '/Users/alexandreday/GitProject/VAC/benchmark/b1_results/'
    
    np.random.seed(0)
    X, ytrue = make_blobs(n_samples=5000, n_features=30, centers=10)
    ytrue+=1 # shifting by 1 !! :O
    Xtsne = pickle.load(open('b1_results/tsne_perp=50_niter=1000_outratio=0.20_pureratio=0.99_minsize=0_nhsize=40_eta=1.50_testfdcsize=0.80.pkl','rb'))
    test_Kmean(X, Xtsne, ytrue)
    exit()

    model = CLUSTER(root='b1_results/')#run_tSNE='auto', plot_inter=False)
    tree, scaler = model.fit(X)

    ypred = tree.predict(scaler.transform(X))
    pickle.dump(ypred, open(root+'ypred.pkl','wb'))
    
    ypred = pickle.load(open(root+'ypred.pkl','rb'))
    ytrue, _, _ = metric.reindex(ytrue)
    ypred, _, _ = metric.reindex(ypred)

    metric.summary(ytrue, ypred, fmt=".2f")

    xtsne = pickle.load(open(model.file_name['tsne'],'rb'))

    print('True labels')
    plotting.cluster_w_label(xtsne, ytrue, title='True labels')

    print('Predicted labels')

    HungS, match_Hung = metric.HUNG_score(ytrue,ypred)
    FlowS, match_Flow = metric.FLOWCAP_score(ytrue,ypred)

    print("Matching FlowScore:\t", match_Flow)
    print("Matching HungScore:\t", match_Hung)

    plotting.cluster_w_label(xtsne, ypred, title='Predicted labels, HungS=%.3f, FlowS=%.3f'%(HungS,FlowS))

def test_Kmean(X, Xtsne, ytrue):
    from sklearn.preprocessing import StandardScaler as scaler
    Xss = scaler().fit_transform(X)
    k=15
    from sklearn.cluster import KMeans
    model = KMeans(n_clusters=k)
    model.fit(Xss)
    ypred = model.labels_

    fscore, match = metric.FLOWCAP_score(ytrue, ypred)
    metric.plot_table(match)
    
    xcenter = []
    for i in range(k):
        xcenter.append(Xtsne[np.argmin(np.linalg.norm(Xss-model.cluster_centers_[i],axis=1))])
    
    print("MATCH:\n",match)
    plotting.cluster_w_label(Xtsne, ytrue, title='$F=%.3f$'%fscore, savefile='b1_results/true.pdf')
    plotting.cluster_w_label(Xtsne, ypred, xcenter,title='$F=%.3f$'%fscore, savefile='b1_results/pred.pdf')
    metric.summary(ytrue, ypred, fontsize=8)


if __name__ == "__main__":
    main()