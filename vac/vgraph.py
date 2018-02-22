from fdc.classify import CLF
from fdc import FDC
import numpy as np
from copy import deepcopy
from collections import Counter, OrderedDict
from matplotlib import pyplot as plt
import pickle
from .tupledict import TupleDict

class VGraph:
    """ Validation graph class - builds a graph with nodes corresponding
    to clusters and draws edges between clusters that have a low validation score
    """
    def __init__(self, n_average = 10, cv_score = 0., edge_min=0.8, test_size_ratio = 0.8, clf_type='rf', clf_args=None):
        self.n_average = n_average
        self.cv_score_threshold = cv_score
        self.test_size_ratio = test_size_ratio
        self.clf_type = clf_type
        self.clf_args = clf_args
        self.cluster_label = None
        self.edge_min = edge_min
        self.edge_score = OrderedDict()
        self.quick_estimate = 50 # should we replace this 
        self.fout = open('out.txt','a')

    def fit(self, X, y_pred):  
        """ In this graph representation neighbors are define by clusters that share an edge
        with a score lower than some predefined threshold (edge_min). Compute nn_list (dict of sets)
        for edges that have a score below the edge_min requirement. 
            
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
        self.cluster_label = np.copy(y_pred)

        for i, yu1 in enumerate(y_unique):
            for j, yu2 in enumerate(y_unique):
                if i<j:
                    idx_tuple = (yu1, yu2)
                    clf = self.classify_edge(idx_tuple, X)# quick_estimate = self.quick_estimate), can shortcut this ?
                    edge_info(idx_tuple, clf.cv_score, clf.cv_score_std, self.cv_score_threshold)
                    self.graph_fast[idx_tuple] = clf

        scores = []
        keys = []
        for k, clf in self.graph_fast.items():
            scores.append(clf.cv_score - clf.cv_score_std)
            keys.append(k)

        asort = np.argsort(scores)[:max(int(0.2*n_iteration), 200)] # just --- work with those edges

        # How to select the edges ? Should you look at distribution and take a percentile, maybe <= =(
        print("Edges that will be used ...")
        for i in asort:
            print("{0:<5d}".format(keys[i][0]),'\t',"{0:<5d}".format(keys[i][1]),'\t',"%.4f"%scores[i])

        # Now looping over those edges and making sure scores are accurately estimated (IMPORTANT)
        self.graph = TupleDict()
        self.nn_list = OrderedDict()
        self.edge_score = TupleDict()

        print('\n\n\n')
        print('Performing deeper sweep over worst edges and coarse-graining from there:')

        for idx in asort:

            idx_tuple = keys[idx]
            i1, i2 = idx_tuple

            if i1 in self.nn_list.keys():
                self.nn_list[i1].add(i2)
            else:
                self.nn_list[i1] = set([i2])
            
            if i2 in self.nn_list.keys():
                self.nn_list[i2].add(i1)
            else:
                self.nn_list[i1] = set([i1])

            clf = self.classify_edge(idx_tuple, X, n_average=10)

            edge_info(idx_tuple, scores[idx], 0., self.cv_score_threshold)
            self.edge_score[idx_tuple] = [clf.cv_score, clf.cv_score_std]
            edge_info(idx_tuple, clf.cv_score, clf.cv_score_std, self.cv_score_threshold)

            self.graph[idx_tuple] = clf
        
        del self.graph_fast
        
        return self
        
    def classify_edge(self, edge_tuple, X, quick_estimate = None, n_average=3):
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

        return CLF(clf_type=self.clf_type, n_average=n_average, test_size=self.test_size_ratio, clf_args=self.clf_args).fit(Xsubset, ysubset)

    def merge_edge(self, X, edge_tuple, n_average=10):
        """ relabels data according to merging, and recomputing new classifiers for new edges """
        
        idx_1, idx_2 = edge_tuple
        pos_1 = (self.cluster_label == idx_1)
        pos_2 = (self.cluster_label == idx_2)
        new_cluster_label = self.init_n_cluster + self.current_n_merge
        
        self.cluster_label[pos_1] = self.init_n_cluster + self.current_n_merge # updating labels !
        self.cluster_label[pos_2] = self.init_n_cluster + self.current_n_merge # updating labels !
        self.current_n_merge += 1

        # recompute classifiers for merged edge
        new_idx = []
        idx_to_del = set([]) # avoids duplicates
        for e in self.nn_list[idx_1]:
            idx_to_del.add((e, idx_1))
            idx_to_del.add((idx_1, e))
            new_idx.append(e)

        for e in self.nn_list[idx_2]:
            idx_to_del.add((e, idx_2))
            idx_to_del.add((idx_2, e))
            new_idx.append(e)
        
        new_nn_to_add = set([])

        for k, v in self.nn_list.items():
            if idx_1 in v:
                v.remove(idx_1)
                v.add(new_cluster_label)
                new_nn_to_add.add(k)
            if idx_2 in v:
                v.remove(idx_2)
                v.add(new_cluster_label)
                new_nn_to_add.add(k)
        
        self.nn_list[new_cluster_label] = new_nn_to_add
        
        if idx_1 in self.nn_list.keys():
            del self.nn_list[idx_1]
        if idx_2 in self.nn_list.keys():
            del self.nn_list[idx_2]
        
        for k, v in self.nn_list.items():
            if idx_1 in v:
                v.remove(idx_1)
            if idx_2 in v:
                v.remove(idx_2)

        ########################################
        #########################################        

        new_idx.remove(idx_1)
        new_idx.remove(idx_2)
    
        for idxd in idx_to_del:
            del self.graph[idxd]
        
        new_idx_set = set([])
        for ni in new_idx:
            new_idx_set.add((new_cluster_label, ni))

        for idx_tuple in new_idx_set:
            clf = self.classify_edge(idx_tuple, X, n_average=n_average)
            self.edge_score[idx_tuple] = [clf.cv_score, clf.cv_score_std]
            edge_info(idx_tuple, clf.cv_score, clf.cv_score_std, self.cv_score_threshold)
            self.graph[idx_tuple] = clf
            idx_tuple_reverse = (idx_tuple[1], idx_tuple[0])
            self.graph[idx_tuple_reverse] = self.graph[idx_tuple]
        
        k0_update = []
        k1_update = []
        for k, v in self.graph.items():
            if (k[0] == idx_1) or (k[0] == idx_2): # old index still present !
                k0_update.append(k)        
            elif (k[1] == idx_1) or (k[1] == idx_2):
                k1_update.append(k)
        
        for k0 in k0_update:
            self.graph[(new_cluster_label, k0[1])] = self.graph.pop(k0)
        for k1 in k1_update:
            self.graph[(k1[0], new_cluster_label)] = self.graph.pop(k1)

    def merge_until_robust(self, X, cv_robust):

        self.init_n_cluster = len(np.unique(self.cluster_label))
        self.current_n_merge = 0
        
        self.history = []
    
        while True:
            all_robust = True
            worst_effect_cv = 10
            worst_edge = -1
            for edge, clf in self.graph.items():
                effect_cv = clf.cv_score - clf.cv_score_std
                if effect_cv < worst_effect_cv:
                    worst_effect_cv = effect_cv
                    worst_edge = edge
                if effect_cv < cv_robust:
                    all_robust = False
            
            if all_robust is False:
                n_cluster = self.init_n_cluster - self.current_n_merge - 1
                current_label = self.init_n_cluster + self.current_n_merge - 1

                merge_info(worst_edge[0], worst_edge[1], worst_effect_cv, current_label, n_cluster)
                
                # info before the merge -> this score goes with these labels
                self.history.append([worst_effect_cv, np.copy(self.cluster_label), deepcopy(self.nn_list)])
                
                #pos_idx0 = (self.cluster_label[self.idx_centers] == worst_edge[0])
                #pos_idx1 = (self.cluster_label[self.idx_centers] == worst_edge[1])
                
                ''' rho_0 = self.rho_idx_centers[pos_idx0]
                rho_1 = self.rho_idx_centers[pos_idx1]

                if rho_0 > rho_1:
                    tmp_idx = self.idx_centers[pos_idx0]
                    tmp_rho = rho_0
                else:
                    tmp_idx = self.idx_centers[pos_idx1]
                    tmp_rho = rho_1

                self.idx_centers[pos_idx0] = -20
                self.idx_centers[pos_idx1] = -20

                pos_del = self.idx_centers > -1

                # new "center" should go to end of list
                tmp_idx_center_array = np.zeros(len(self.idx_centers)-1,dtype=int)
                tmp_idx_center_array[:-1] = self.idx_centers[pos_del]
                tmp_idx_center_array[-1] = tmp_idx
                self.idx_centers = tmp_idx_center_array

                tmp_rho_array = np.zeros(len(self.rho_idx_centers)-1,dtype=float)
                tmp_rho_array[:-1] = self.rho_idx_centers[pos_del]
                tmp_rho_array[-1] = tmp_rho
                self.rho_idx_centers = tmp_rho_array '''
                
                self.merge_edge(X, worst_edge)
        
            else:
                break

        if len(self.idx_centers) == 1:
            self.history.append([1.0, np.copy(self.cluster_label), deepcopy(self.nn_list)])
        else:
            self.history.append([worst_effect_cv, np.copy(self.cluster_label), deepcopy(self.nn_list)])

    def print_edge_score(self):
        """ Print edge scores in sorted """

        score = []
        idx_list =[]
        for idx, s in self.edge_score.items():
            score.append(s[0]-s[1])
            idx_list.append(idx)
    
        asort = np.argsort(score)
        print("{0:<8s}{1:<8s}{2:<10s}".format('e1', 'e2', 'score'))
        for a in asort:
            print("{0:<8d}{1:<8d}{2:<10.4f}".format(idx_list[a][0], idx_list[a][1], score[a]))

def edge_info(edge_tuple, cv_score, std_score, min_score):
    edge_str = "{0:5<d}{1:4<s}{2:5<d}".format(edge_tuple[0]," -- ",edge_tuple[1])
    if cv_score > min_score:
        out = "[graph.py] : {0:<15s}{1:<15s}{2:<15s}{3:<7.4f}{4:<16s}{5:>6.5f}".format("robust edge ",edge_str,"score =",cv_score,"\t+-",std_score)
    else:
        out = "[graph.py] : {0:<15s}{1:<15s}{2:<15s}{3:<7.4f}{4:<16s}{5:>6.5f}".format("reject edge ",edge_str,"score =",cv_score,"\t+-",std_score)
    print(out)
    self.fout.write(out)
    

def merge_info(c1, c2, score, new_c, n_cluster):
    edge_str = "{0:5<d}{1:4<s}{2:5<d}".format(c1," -- ",c2)
    out = "[graph.py] : {0:<15s}{1:<15s}{2:<15s}{3:<7.4f}{4:<16s}{5:>6d}{6:>15s}{7:>5d}".format("merge edge ",edge_str,"score - std =",score,
    "\tnew label ->",new_c,'n_cluster=',n_cluster)
    print(out)
    self.fout.write(out)