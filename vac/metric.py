import numpy as np
from collections import OrderedDict
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment as LSA
import pandas as pd
import subprocess

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

def summary(y_true, y_pred, fmt= ".2f", fontsize=10):
    
    from matplotlib import pyplot as plt
    import seaborn as sns

    C = confusion_matrix(y_true, y_pred) # automatically padded to include zeros  -> match number of clusters and populations
    F = F_matrix(y_true, y_pred)
    fig, ax_ori = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
    ax = sns.heatmap(C.T, annot=True, fmt="d", ax = ax_ori[0], annot_kws={'fontsize':fontsize})
    ax.set_ylabel('Predicted')
    ax.set_xlabel('True')
    ax.set_title('Confusion')

    ax = sns.heatmap(F.T, annot=True, fmt=fmt, ax = ax_ori[1], annot_kws={'fontsize':fontsize})
    ax.set_ylabel('Predicted')
    ax.set_xlabel('True')
    ax.set_title('F1-measure')
    plt.tight_layout()
    plt.show()

def reindex(y):
    """ Array of labels (arbitrary integer labels)
    These are reindexed starting from 0

    Returns
    -------
    new vector of labels, dictionary map (from old to new labels), inverse map
    """
    yu = np.unique(y) # 
    mapidx = dict(zip(yu, np.arange(len(yu))))
    inv_mapidx = dict(zip(np.arange(len(yu)),yu))
    return np.vectorize(mapidx.get)(y), mapidx, inv_mapidx

def F_matrix(y_true, y_pred, eps=1e-10):
    """ From the confusion matrix, build the F-matrix
    -> The first index is the true clusters, the second index is the predicted cluster
    """
    C = confusion_matrix(y_true, y_pred) # works with actual indexing, will make it a sq. matrix if n_class aren't the same
    ntrue, npred = C.shape # ok, all good so far -> this should make it squared

    sensitivity = np.vstack([C[i]/(np.sum(C[i])+eps) for i in range(ntrue)])
    precision = np.vstack([C[:,j]/(np.sum(C[:,j])+eps) for j in range(npred)]).T

    return 2*np.reciprocal(1./(sensitivity+eps) + 1./(precision+eps))

def FLOWCAP_score(y_true_, y_pred_):
    """F score is maximized individually for each population (true label)
    Caveat : Some clusters (pred. label) may be matched to multiple populations
    Also, the Fscore is computed via a weighted average w.r.t to the true label population size ratios
    
    Returns
    -------
    Fscore, dataframe of matching
    
    """


    y_true, maptrue, invmaptrue = reindex(y_true_)
    y_pred, mappred, invmappred = reindex(y_pred_)

    F = F_matrix(y_true, y_pred) # will pad for missing classes
    y_u_true, counts = np.unique(y_true, return_counts = True)
    y_u_pred = np.unique(y_pred)
    weight = counts/len(y_true) # weights of true populations

    match = []
    match_F_score = []
    match_weight = []
    match_FlowScore = []
    Fscore = 0.
    
    for i, yu in enumerate(y_u_true):
        match.append(y_u_pred[np.argmax(F[i])]) # problem with this is that a population may be assigned to the same cluster twice ...
        match_F_score.append(np.max(F[i]))
        match_weight.append(weight[i])
        Fscore += match_F_score[-1]*match_weight[-1] # weigthed average !

    match_F_score=np.array(match_F_score)
    match_weight=np.array(match_weight)

    df = pd.DataFrame(OrderedDict({'true':list(range(len(y_u_true))),'predict':match, 'Fmeasure':match_F_score,'weight':match_weight,'FCscore':match_weight*match_F_score}))

    translateDF(df, invmaptrue, 'true')
    translateDF(df, invmappred, 'predict')
    return Fscore, df

def plot_table(df, dpi=500, fname='table.pdf'):
    import subprocess
    df.to_html('table.html')
    subprocess.call('wkhtmltopdf --dpi %i table.html table.pdf'%dpi, shell=True)
    subprocess.call('pdfcrop table.pdf', shell=True)
    subprocess.call('rm table.pdf', shell=True)
    subprocess.call('rm table.html', shell=True)
    subprocess.call('mv table-crop.pdf %s'%fname, shell=True)

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

def translateDF(df, mymap, col='true'):

    translation = []
    for i, element in enumerate(df[col]):
        if element in mymap.keys():
            translation.append(mymap[element])
        else:
            translation.append(-1)
    df[col] = np.array(translation, dtype=int)

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
