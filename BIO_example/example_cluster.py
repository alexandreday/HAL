from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from vac import CLUSTER
from fdc import plotting, transform
from sklearn.decomposition import PCA
import pickle
from tsne_visual import TSNE
from matplotlib import pyplot as plt
import phenograph
from sklearn.metrics import normalized_mutual_info_score as NMI
from collections import Counter

# Link to your dataset
file = "/Users/alexandreday/Dropbox/Work/Project_PHD/Immunology/Visit/FLOWCAP/FlowCAP-I/Data/FCM/csv/NDD/CSV/"
file_label = "/Users/alexandreday/Dropbox/Work/Project_PHD/Immunology/Visit/FLOWCAP/FlowCAP-I/Data/Labels/NDD/"

i=30
datafile = ("%i"%i).zfill(3)+".csv"
data = pd.read_csv(file+datafile)
X = data.values
y = pd.read_csv(file_label+datafile).values.flatten() # "expert" labels
columns = data.columns.values
#print(y)
count = Counter(y)
count = {k:count[k]/len(y) for k in count.keys()}
fopen = open('count_%i.txt'%i,'w')
for k, v in count.items():
    tmp ='%i\t%.5f\n'%(k, v)
    fopen.write(tmp)

X = StandardScaler().fit_transform(X)
np.random.seed(0)
xtsne = TSNE().fit_transform(X[:10000])
np.savetxt('xtsne_%i.txt'%i, xtsne)
ypred, graph, Q = phenograph.cluster(X[:10000])

# what to do with this ? -> make some plots, run some more code
plotting.cluster_w_label(xtsne, ypred, savefile='phenograph_%i.png'%i, title='NMI=%.3f'%NMI(ypred, y[:10000]),
w_legend=True, w_label=False, dpi=150)
plotting.cluster_w_label(xtsne, y[:10000], savefile='yexpert_%i.png'%i
,w_legend=True, w_label=False,dpi=150)
#print(NMI(ypred, y[:10000]))
exit()





exit()


# down_sampling size
n_down_sample = 5000

# full pipeline model
model = CLUSTER(n_down_sample=n_down_sample, plot_inter=False, run_tSNE=False)

# data is zscored within the object. Fitting your data.
# also note that tSNE will be saved, so you can reuse it later.
# just use run_tSNE = False in CLUSTER(...)
tree = model.fit(X) # this will save your tree in myTree.pkl

tree = pickle.load(open('myTree.pkl','rb'))

cv = 0.9 # choose your score

# prediction on the samples used for training !
ypred = tree.predict(StandardScaler().fit_transform(X[:n_down_sample]), cv=cv)

np.savetxt('ypred_s=%.3f.txt'%cv, ypred) # save the labels

ypred = np.loadtxt('ypred_s=%.3f.txt'%cv) # load the labels

# PLOTTING HERE --> need fdc to do that
#Xtsne = pickle.load(open('tsne.pkl','rb')) 
#plotting.cluster_w_label(Xtsne, ypred[:n_down_sample]) # plot the data






