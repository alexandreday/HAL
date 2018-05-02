import numpy as np
from collections import OrderedDict
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment as LSA

def main():
    from matplotlib import pyplot as plt
    import seaborn as sns
    from fdc import FDC, plotting
    from sklearn import datasets
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # Confusion matrix example with F-measure stuff ->
    np.random.seed(0)
    X,y = datasets.load_iris(return_X_y=True)
    Xss = StandardScaler().fit_transform(X)
    xpca = PCA(n_components=2).fit_transform(Xss)
    #plotting.cluster_w_label(xpca, y)
    #exit()
    print(xpca.shape)
    model = FDC(test_ratio_size=0.5, eta=0.1).fit(xpca)
    #print(model.cluster_label)
    #print(y)
    #plotting.cluster_w_label(xpca, model.cluster_label)
    print(FLOWCAP_score(y, model.cluster_label))
    print(HUNG_score(y, model.cluster_label))
    exit()
    summary(y, model.cluster_label)

def summary(y_true, y_pred, fmt= ".4f"):
    C = confusion_matrix(y_true, y_pred) # automatically padded to include zeros  -> match number of clusters and populations
    F = F_matrix(y_true, y_pred)
    
    fig, ax_ori = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    ax = sns.heatmap(C, annot=True, fmt="d", ax = ax_ori[0])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion')
    ax = sns.heatmap(F, annot=True, fmt=fmt, ax = ax_ori[1])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('F1-measure')
    plt.tight_layout()
    plt.show()

def F_matrix(y_true, y_pred):
    C = confusion_matrix(y_true, y_pred)
    
    ntrue, npred = C.shape # ok, all good so far
    sensitivity = np.vstack([C[i]/np.sum(C[i]) for i in range(ntrue)])
    precision = np.vstack([C[:,j]/np.sum(C[:,j]) for j in range(npred)]).T
    return 2*np.reciprocal(1./(sensitivity+1e-8) + 1./(precision+1e-8))

def FLOWCAP_score(y_true, y_pred):
    """F score is maximized individually for each population (true label)
    Caveat : Some clusters (pred. label) may be matched to multiple populations
    Also, the Fscore is computed via a weighted average w.r.t to the true label population size ratios"""

    F = F_matrix(y_true, y_pred)
    y_u_true, counts = np.unique(y_true,return_counts = True)
    y_u_pred = np.unique(y_pred)
    weight = counts/len(y_true)
    match = {}

    Fscore = 0.
    for i, yu in enumerate(y_u_true):
        match[yu] = y_u_pred[np.argmax(F[i])] # problem with this is that a population may be assigned to the same cluster twice ...
        Fscore += np.max(F[i])*weight[i] # weigthed average !
    return Fscore, match 

def HUNG_score(y_true, y_pred):
    """F score is computed via the Hungarian algorithm which
    determined the optimal match of populations to clusters
    If there are more clusters than populations, then some clusters are left unassigned
    If there are more populations than clusters, than some populations are unassigned ... (problem !?)  
    """

    F = F_matrix(y_true, y_pred)
    C = 1 - F
    r, c = LSA(C)
    match = {r[i] : c[i] for i in range(len(r))}
    Fscore = np.mean(1.-C[r, c]) # equally weighted average
    return Fscore, match

def clustering(y):
    yu = np.sort(np.unique(y))
    clustering = OrderedDict()
    for ye in yu:
        clustering[ye] = np.where(y == ye)[0]
    return clustering

def entropy(c, n_sample):
    h = 0.
    for kc in c.keys():
        p=len(c[kc])/n_sample
        h+=p*np.log(p)
    h*=-1.
    return h

def NMI(y_true, y_pred):
    """ Computes normalized mutual information: where w and c are both clustering assignments
    For a neat discussion on different possible metrics, see : https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html
    Other main ref is : Image Clustering Using Local Discriminant, Models and Global Integration.
    There they also define ACC, which requires hungarian algorithm and only works for 1-to-1 assigments.
     """
    w = clustering(y_true)
    c = clustering(y_pred)
    n_sample = len(y_true)

    Iwc = 0.
    for kw in w.keys():
        for kc in c.keys():
            w_intersect_c=len(set(w[kw]).intersection(set(c[kc])))
            if w_intersect_c > 0:
                Iwc += w_intersect_c*np.log(n_sample*w_intersect_c/(len(w[kw])*len(c[kc])))
    Iwc/=n_sample
    Hc = entropy(c,n_sample)
    Hw = entropy(w,n_sample)

    return 2*Iwc/(Hc+Hw)

if __name__ == "__main__":
    main()
