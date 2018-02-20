from fdc.classify import CLF
import numpy as np
from copy import deepcopy
from collections import Counter, OrderedDict
from matplotlib import pyplot as plt
import pickle

class VGRAPH:
    """ Validation graph class - builds a graph with nodes corresponding
    to clusters and draws edges between clusters that have a low validation score
    """
    def __init__(self, n_average = 10, cv_score = 0., edge_min=0.8, test_size_ratio = 0.8, clf_type='svm', clf_args=None):
        self.n_average = n_average
        self.cv_score_threshold = cv_score
        self.test_size_ratio = test_size_ratio
        self.clf_type = clf_type
        self.clf_args = clf_args
        self.cluster_label = None
        self.edge_min = edge_min
        self.edge_score = OrderedDict()
        self.quick_estimate = 50

    def fit(self, X, y_pred):
        self.compute_nn_list(X, y_pred)
        #self.fit_all_clf(model, X)
        return self
    
    def compute_nn_list(self, X, y_pred):  
        """ In this graph representation neighbors are define by clusters that share an edge
        with a score lower than some predefined threshold
            
        Parameters:
        ----------------

        X : array, shape = (n_sample, n_feature)
            Original space coordinates
        y_pred: array, shape = (n_sample,)
            Cluster labels for each data point
        """

        y_unique = np.unique(y_pred)
        n_cluster = len(y_unique)
        n_iteration = (n_cluster*(n_cluster-1))/2

        print('[vgraph.py]  Performing classification sweep over %i pairs of clusters'%n_iteration)
        self.graph_fast = OrderedDict() 
        self.edge_score = OrderedDict()
        self.cluster_label = np.copy(y_pred)

        for i, yu1 in enumerate(y_unique):
            for j, yu2 in enumerate(y_unique):
                if i<j:
                    idx_tuple = (yu1, yu2)
                    
                    clf = self.classify_edge(idx_tuple, X, quick_estimate = self.quick_estimate)
                    
                    self.edge_score[idx_tuple] = [clf.cv_score, clf.cv_score_std]
                    edge_info(idx_tuple, clf.cv_score, clf.cv_score_std, self.cv_score_threshold)

                    self.graph_fast[idx_tuple] = clf

        scores = []
        keys = []
        for k, clf in self.graph.items():
            scores.append(clf.cv_score - clf.cv_score_std)
            keys.append(k)

        asort = np.argsort(scores)[:max(int(0.05*n_iteration),100)] # just work with those edges

        self.graph = OrderedDict()
        print('Performing deeper sweep over worst edges and coarse-graining from there')
        for idx in asort:

            idx_tuple= keys[idx]
            clf = self.classify_edge(idx_tuple, X)
            
            self.edge_score[idx_tuple] = [clf.cv_score, clf.cv_score_std]
            edge_info(idx_tuple, clf.cv_score, clf.cv_score_std, self.cv_score_threshold)

            self.graph[idx_tuple] = clf
            self.graph[idx_tuple_reverse] = self.graph_fast[idx_tuple]

        
            

    def classify_edge(self, edge_tuple, X, C = 1.0, quick_estimate = None, n_average=3):
        """ Trains a classifier on the childs of "root" and returns a classifier for these types.

        Important attributes are (for CLF object):

            self.scaler_list -> [mu, std]

            self.cv_score -> mean cv score

            self.mean_train_score -> mean train score

            self.clf_list -> list of sklearn classifiers (for taking majority vote)
        
        Returns
        ---------
        CLF object (from classify.py). Object has similar syntax to sklearn's classifier syntax

        """
        ## ok need to down sample somewhere here
        test_size_ratio = self.test_size_ratio
        n_average = self.n_average

        y = np.copy(self.cluster_label)
        y[(y != edge_tuple[0]) & (y != edge_tuple[1])] = -1

        pos_subset =  (y != -1)
        Xsubset = X[pos_subset] # original space coordinates
        ysubset = y[pos_subset] # labels
        n_sample = len(ysubset)

        if quick_estimate is not None:
            n_sub = quick_estimate # should be an integer
            pos_0 = np.where(ysubset == edge_tuple[0])[0]
            pos_1 = np.where(ysubset == edge_tuple[1])[0]
            np.random.shuffle(pos_0)
            np.random.shuffle(pos_1)
            Xsubset = np.vstack((Xsubset[pos_0[:quick_estimate]], Xsubset[pos_1[:quick_estimate]]))
            ysubset = np.hstack((ysubset[pos_0[:quick_estimate]], ysubset[pos_1[:quick_estimate]]))

        return CLF(clf_type=self.clf_type, n_average=n_average, test_size=self.test_size_ratio,clf_args=self.clf_args).fit(Xsubset, ysubset)

def edge_info(edge_tuple, cv_score, std_score, min_score):
    edge_str = "{0:5<d}{1:4<s}{2:5<d}".format(edge_tuple[0]," -- ",edge_tuple[1])
    if cv_score > min_score:
        print("[graph.py] : {0:<15s}{1:<15s}{2:<15s}{3:<7.4f}{4:<16s}{5:>6.5f}".format("robust edge ",edge_str,"score =",cv_score,"\t+-",std_score))
    else:
        print("[graph.py] : {0:<15s}{1:<15s}{2:<15s}{3:<7.4f}{4:<16s}{5:>6.5f}".format("reject edge ",edge_str,"score =",cv_score,"\t+-",std_score))