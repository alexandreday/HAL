from fdc.classify import CLF
from fdc import FDC
import numpy as np
from copy import deepcopy
from collections import Counter, OrderedDict
from matplotlib import pyplot as plt
import pickle, time
from .tupledict import TupleDict
from .utility import FOUT

class VGraph:
    """ Validation graph class - builds a graph with nodes corresponding
    to clusters and draws edges between clusters that have a low validation score
    """
    def __init__(self, n_average = 10, cv_score = 0., test_size_ratio = 0.8, clf_type='rf', clf_args=None, n_edge =2):
        self.n_average = n_average
        self.cv_score_threshold = cv_score
        self.test_size_ratio = test_size_ratio
        self.clf_type = clf_type
        self.clf_args = clf_args
        self.cluster_label = None
        self.edge_score = OrderedDict()
        self.quick_estimate = 50 # should we replace this
        self.fout = FOUT('out.txt')
        self.n_edge = n_edge

    def fit(self, X, y_pred):  
        """ Constructs a low connectivity graph by joining clusters that have low edge scores.
        Each cluster is connected to n_edge other clusters.

        Parameters:
        ----------------

        X : array, shape = (n_sample, n_feature)
            Original space coordinates
        y_pred: array, shape = (n_sample,)
            Cluster labels for each data point
        n_edge: int, default = 2
            Number of edges for each node (worst ones)
        """

        y_unique = np.unique(y_pred)
        n_cluster = len(y_unique)
        n_iteration = (n_cluster*(n_cluster-1))/2
        n_edge = self.n_edge
        
        
        print('[vgraph.py]  Performing classification sweep over %i pairs of clusters'%n_iteration)
        self.cluster_label = np.copy(y_pred)

        n_average_pre = 3 # don't bother with this now, we just want a rough idea of what is good and what is bad.
        clf_args_pre = {'class_weight':'balanced', 'n_estimators': 10, 'max_features': 200}
        
        info = 'parameters:\t'+("n_average =%i"%n_average_pre)+'\t'+str(clf_args_pre)
        print(info)
        self.fout.write(info)
        score = {yu:{} for yu in y_unique} # dict of dict score[i][j] returns score for that edge

        # This is O(N^2) complexity in the number of clusters ...
        for i, yu1 in enumerate(y_unique): 
            for j, yu2 in enumerate(y_unique):
                if i<j:
                    idx_tuple = (yu1, yu2)
                    clf = self.classify_edge(idx_tuple, X, clf_args=clf_args_pre, n_average=n_average_pre)# quick_estimate = self.quick_estimate), can shortcut this ?
                    edge_info(idx_tuple, clf.cv_score, clf.cv_score_std, self.cv_score_threshold, fout=self.fout)
                    score[i][j] = clf.cv_score # since n_average is small, just use mean, no variance for now !
        
        edge_list = []
        score_list = []
        for cluster_idx in y_unique:
            key_tmp = list(score[cluster_idx].keys())
            v_tmp = list(score[cluster_idx].values())
            asort = np.argsort(v_tmp)
            for pos in asort[:n_edge]: ## ---------> For each node, get n_edges (worst ones) ==> here may want to look at a distribution or something like that
                edge_list.append((cluster_idx, key_tmp[pos]))
                score_list.append(v_tmp[pos])
    
        edge_list = np.array(edge_list)[np.argsort(score_list)] # resort end list 

        # How to select the edges ? Should you look at distribution and take a percentile, maybe <= =(
        print("[vgraph.py]    Edges that will be used ...")
        for edge in edge_list:
            i, j = edge
            print("[vgraph.py]    {0:<5d}".format(i),' ---- ',"{0:<5d}".format(j),'  =~  ',"%.4f"%score[i][j])

        # Now looping over those edges and making sure scores are accurately estimated (IMPORTANT)
        self.graph = TupleDict()
        self.nn_list = OrderedDict()
        self.edge_score = TupleDict()

        print('\n\n\n')
        print('[graph.py]    Performing deeper sweep over worst edges and coarse-graining from there:')

        for yu in y_unique:
            self.nn_list[yu] = set([])

        info = '[graph.py]    '+'Parameters:\t'+("n_average =%i"%self.n_average)+'\t'+str(self.clf_args)
        print(info)
        self.fout.write(info)

        # This is O(N) in the number of clusters
        for edge in edge_list:
            n1, n2 = edge
            self.nn_list[n1].add(n2)
            self.nn_list[n2].add(n1)
            
            clf = self.classify_edge(edge, X, n_average=self.n_average, clf_args = self.clf_args)
            self.edge_score[idx_tuple] = [clf.cv_score, clf.cv_score_std]

            # printing out results
            edge_info_update(edge, score[n1][n2], 0., clf.cv_score, clf.cv_score_std, self.cv_score_threshold, fout=self.fout)

            self.graph[idx_tuple] = clf
            
        return self
        
    def classify_edge(self, edge_tuple, X, quick_estimate = None, n_average=3, clf_args = None):
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
        
        return CLF(clf_type=self.clf_type, n_average=n_average, test_size=self.test_size_ratio, clf_args=clf_args).fit(Xsubset, ysubset)

    def merge_edge(self, X, edge_tuple):
        """ relabels data according to merging, and recomputing new classifiers for new edges """
        
        idx_1, idx_2 = edge_tuple
        pos_1 = (self.cluster_label == idx_1)
        pos_2 = (self.cluster_label == idx_2)
        new_cluster_label = self.current_max_label
        
        self.cluster_label[pos_1] = new_cluster_label   # updating labels !
        self.cluster_label[pos_2] = new_cluster_label   # updating labels !
        
        self.current_n_merge += 1
        self.current_max_label += 1 

        new_idx = []    # recompute classifiers for merged edge
        idx_to_del = set([])    # avoids duplicates

        for e in self.nn_list[idx_1]:
            if e<idx_1:
                idx_to_del.add((e, idx_1))
            else:
                idx_to_del.add((idx_1, e))
            new_idx.append(e)

        for e in self.nn_list[idx_2]:
            if e<idx_2:
                idx_to_del.add((e, idx_2))
            else:
                idx_to_del.add((idx_2, e))
            #idx_to_del.add((idx_2, e))
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
    
        for idxd in idx_to_del: # these edges have been reassigned ... 
            del self.graph[idxd]
        
        new_idx_set = set([])
        for ni in new_idx:
            new_idx_set.add((new_cluster_label, ni))

        for idx_tuple in new_idx_set:
            clf = self.classify_edge(idx_tuple, X, n_average=self.n_average, clf_args = self.clf_args)
            self.edge_score[idx_tuple] = [clf.cv_score, clf.cv_score_std]
            edge_info(idx_tuple, clf.cv_score, clf.cv_score_std, self.cv_score_threshold, fout=self.fout)
            self.graph[idx_tuple] = clf
            idx_tuple_reverse = (idx_tuple[1], idx_tuple[0])
        

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

        yunique = np.unique(self.cluster_label)
        self.init_n_cluster = len(yunique)
        self.current_max_label = np.max(yunique) + 1
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

                merge_info(worst_edge[0], worst_edge[1], worst_effect_cv, self.current_max_label, n_cluster, fout = self.fout)
                
                # info before the merge -> this score goes with these labels            
                self.history.append([worst_effect_cv, np.copy(self.cluster_label), deepcopy(self.nn_list)])
                self.merge_edge(X, worst_edge)
            else:
                self.history.append([worst_effect_cv, np.copy(self.cluster_label), deepcopy(self.nn_list)])
                break


    def print_edge_score(self, option = 0):
        """ Print edge scores in sorted """

        score = []
        idx_list =[]
        for idx, s in self.edge_score.items():
            if option == 0:
                score.append(s[0]-s[1])
            else:
                score.append(s[0])
            idx_list.append(idx)
    
        asort = np.argsort(score)
        print("{0:<8s}{1:<8s}{2:<10s}".format('e1', 'e2', 'score'))
        for a in asort:
            print("{0:<8d}{1:<8d}{2:<10.4f}".format(idx_list[a][0], idx_list[a][1], score[a]))

def edge_info(edge_tuple, cv_score, std_score, min_score, fout=None):
    
    edge_str = "{0:5<d}{1:4<s}{2:5<d}".format(edge_tuple[0]," -- ",edge_tuple[1])

    robust_or_not = "robust edge" if cv_score - std_score > min_score else "reject edge "

    out = "[vgraph.py]    {0:<15s}{1:<15s}{2:<15s}{3:<7.4f}{4:<16s}{5:>6.5f}".format(robust_or_not, edge_str, "score =", cv_score, "\t+-",std_score)

    print(out)

    if fout is not None:
        fout.write(out)

def edge_info_update(edge_tuple, cv_score_pre, std_score_pre, cv_score_post, std_score_post, min_score, fout=None):
    
    edge_str = "{0:5<d}{1:4<s}{2:5<d}".format(edge_tuple[0]," -- ",edge_tuple[1])

    robust_or_not = "robust edge" if cv_score_post - std_score_post > min_score else "reject edge "
    out ="[vgraph.py]    {0:<15s}{1:<15s}{2:<15s}{3:<7.4f}{4:^8s}{5:6.5f}{6:^10s}{7:<7.4f}{8:^8s}{9:6.5f}".format(robust_or_not, edge_str, "score change", 
    cv_score_pre, '+/-', std_score_pre,
    '\t>>>>\t', 
    cv_score_post, '+/-', std_score_post
    )

    print(out)

    if fout is not None:
        fout.write(out)
    
def merge_info(c1, c2, score, new_c, n_cluster, fout=None):
    edge_str = "{0:5<d}{1:4<s}{2:5<d}".format(c1," -- ",c2)
    out = "[vgraph.py]    {0:<15s}{1:<15s}{2:<15s}{3:<7.4f}{4:<16s}{5:>6d}{6:>15s}{7:>5d}".format("merge edge ",edge_str,"score - std =",score,
    "\tnew label ->",new_c,'n_cluster=',n_cluster)
    print(out)
    if fout is not None:
        fout.write(out)