from sklearn.datasets import make_blobs
import numpy as np
from matplotlib import pyplot as plt
from tsne_visual import TSNE
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
    X, ytrue = make_blobs(n_samples=10000, n_features=30, centers=15)
    #Xtsne = TSNE(perplexity=40).fit_transform(X)
    #pickle.dump(Xtsne, open('b1_results/xtsne.pkl','wb'))
    Xtsne = pickle.load(open('b1_results/xtsne.pkl','rb'))
    #plotting.cluster_w_label(Xtsne, ytrue)
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
    from sklearn.cluster import KMeans

    Xss = scaler().fit_transform(X)

    fcscore_ls = []
    hgscore_ls = []
    nmi_ls = []
    nk = 30
    kcheck = [5,15,25]

    for k in range(1, nk):
        model = KMeans(n_clusters=k)
        model.fit(Xss)
        ypred = model.labels_
        fcscore, matchfc = metric.FLOWCAP_score(ytrue, ypred)
        hgscore, matchhg = metric.HUNG_score(ytrue, ypred)
        #print(hgscore)
        fcscore_ls.append(fcscore)
        hgscore_ls.append(hgscore)
        nmi_ls.append(metric.NMI(ytrue, ypred))
        if k in kcheck:
            print('Check ',k)
            xcenter = []
            for i in range(k):
                xcenter.append(Xtsne[np.argmin(np.linalg.norm(Xss-model.cluster_centers_[i],axis=1))])

            metric.plot_table(matchhg, fname = 'b1_results/table_k=%i.pdf'%k)
            #plotting.cluster_w_label(Xtsne, ytrue, title='$F=%.3f$'%fscore, savefile='b1_results/true_k=%i.pdf'%k,show=False)
            
            plotting.cluster_w_label(Xtsne, ypred, xcenter,title="$FC=%.3f,HG=%.3f$"%(fcscore,hgscore), savefile='b1_results/pred_k=%i.pdf'%k, show=False)
            plt.close()
            metric.summary(ytrue, ypred, fontsize=6,savefile='b1_results/summary_k=%i.pdf'%k, show=False)
            plt.close()

    plt.scatter(np.arange(1,nk),fcscore_ls,label='flowCAP')
    plt.scatter(np.arange(1,nk),hgscore_ls,label='Hung')
    plt.scatter(np.arange(1,nk),nmi_ls,label='NMI')
    plt.legend(loc='best')
    plt.show()

if __name__ == "__main__":
    main()