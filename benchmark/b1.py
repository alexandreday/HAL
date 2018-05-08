from sklearn.datasets import make_blobs
import numpy as np
from matplotlib import pyplot as plt
import pickle
from vac import CLUSTER
from fdc import plotting
from collections import Counter
from vac import metric


########################## BENCHMARK ON easy artificial DATASET ################
########################## Important for more involved examples ###############

def main():
    root = '/Users/alexandreday/GitProject/VAC/benchmark/b1_results/'
    
    np.random.seed(0)
    X, ytrue = make_blobs(n_samples=5000, n_features=30, centers=10)
    model = CLUSTER(root='b1_results/', run_tSNE='auto', plot_inter=False)
    tree, scaler = model.fit(X)

    ypred = tree.predict(scaler.transform(X))

    pickle.dump(ypred, open(root+'ypred.pkl','wb'))
    
    ypred = pickle.load(open(root+'ypred.pkl','rb'))
    ytrue, _, _ = metric.reindex(ytrue)
    ypred, _, _ = metric.reindex(ypred)

    metric.summary(ytrue, ypred, fmt=".2f")

    xtsne = pickle.load(open(model.file_name['tsne'],'rb')))

    print('True labels')
    plotting.cluster_w_label(xtsne, ytrue, title='True labels')

    print('Predicted labels')

    HungS, match_Hung = metric.HUNG_score(ytrue,ypred)
    FlowS, match_Flow = metric.FLOWCAP_score(ytrue,ypred)

    print("Matching FlowScore:\t", match_Flow)
    print("Matching HungScore:\t", match_Hung)

    plotting.cluster_w_label(xtsne, ypred, title='Predicted labels, HungS=%.3f, FlowS=%.3f'%(HungS,FlowS))

if __name__ == "__main__":
    main()