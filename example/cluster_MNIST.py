from read_MNIST import load_mnist
from sklearn.preprocessing import StandardScaler
import numpy as np
from vac import CLUSTER
from fdc import plotting
from sklearn.decomposition import PCA
import pickle

p = "/Users/alexandreday/GitProject/tsne_visual/example/MNIST"

X, ytrue = load_mnist(dataset="training", fmt="pandas", digits=np.arange(10), path=p)

##Xtsne = pickle.load(open('tsne.pkl','rb'))
#plotting.cluster_w_label(Xtsne, ytrue[:n_down_sample].astype(int).flatten())

ytrue = ytrue.astype(int).flatten()

n_down_sample = 10000
model = CLUSTER(n_down_sample=n_down_sample)

tree = model.fit(X)
#tree = pickle.load(open('myTree.pkl','rb'))

Xtsne = pickle.load(open('tsne.pkl','rb'))
plotting.cluster_w_label(Xtsne, ytrue[:n_down_sample])

cv = 0.95

ypred = tree.predict(StandardScaler().fit_transform(X[:n_down_sample]), cv=cv)
np.savetxt('ypred_s=%.3f.txt'%cv, ypred)

ypred=np.loadtxt('ypred_s=%.3f.txt'%cv)
from sklearn.metrics import normalized_mutual_info_score as NMI
print('score: ', NMI(ypred, ytrue[:len(ypred)]))

plotting.cluster_w_label(Xtsne, ypred[:n_down_sample])







