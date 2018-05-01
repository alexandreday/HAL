from read_MNIST import load_mnist
from sklearn.preprocessing import StandardScaler
import numpy as np
from vac import CLUSTER
from fdc import plotting
from sklearn.decomposition import PCA
import pickle

# Link to your dataset
p = "/Users/alexandreday/GitProject/tsne_visual/example/MNIST"
X, ytrue = load_mnist(dataset="training", fmt="pandas", digits=np.arange(10), path=p)
ytrue = ytrue.astype(int).flatten()
# X is the data, y are the true labels

# downsampling size
n_down_sample = 10000

model = CLUSTER(n_down_sample=n_down_sample, plot_inter=False)

# don't forget to transform your data here. The clustering will only zscore it.
tree = model.fit(X) # this will save your tree in myTree.pkl

# d
tree = pickle.load(open('myTree.pkl','rb'))

cv = 0.9 # choose your score
ypred = tree.predict(StandardScaler().fit_transform(X[:n_down_sample]), cv=cv) # prediction on the samples used for training !

np.savetxt('ypred_s=%.3f.txt'%cv, ypred) # save the labels

ypred = np.loadtxt('ypred_s=%.3f.txt'%cv) # load the labels

Xtsne = pickle.load(open('tsne.pkl','rb')) 

plotting.cluster_w_label(Xtsne, ypred[:n_down_sample]) # plot the data






