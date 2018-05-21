from classify import CLF
from fdc import FDC
import numpy as np
from copy import deepcopy
from collections import Counter, OrderedDict
from matplotlib import pyplot as plt
import pickle, time
from tupledict import TupleDict
from utility import FOUT

class kNN_Graph:
    """ Validation graph class - builds a graph with nodes corresponding
    to clusters and draws edges between clusters that have a low validation score
    """
    def __init__(self, 
        n_bootstrap = 10, 
        cv_score = 0., 
        test_size_ratio = 0.8,
        n_sample_max = 1000,
        clf_type='svm', 
        clf_args=None,
        verbose = 1,
        n_edge =2,
        y_murky= None
        ):
        
        self.n_bootstrap = n_bootstrap
        self.cv_score_threshold = cv_score
        self.test_size_ratio = test_size_ratio
        self.clf_type = clf_type
        self.n_sample_max = n_sample_max
        self.recomputed = False

        if clf_args is None:
            self.clf_args = {'class_weight':'balanced'}
        else:
            self.clf_args = clf_args

        self.edge_score = OrderedDict()
        self.fout = None#/FOUT('out.txt')
        self.n_edge = n_edge
        self.y_murky = y_murky
        self.cout = graph_cout if verbose is 1 else lambda *a, **k: None
        print(self.__dict__)
        
    def fit(self, X, y_pred, n_bootstrap_shallow = 5):  
        """ Constructs a low connectivity graph by joining clusters that have low edge scores.
        Each cluster is connected to n_edge other clusters.

        Parameters:
        ----------------

        X : array, shape = (n_sample, n_feature)
            Original space coordinates
        y_pred: array, shape = (n_sample,)
            Cluster labels for each data point. Boundary points should be labelled by -1 ***********************
        n_edge: int, default = 2
            Number of edges for each node (worst ones)
        """

        y_unique = np.unique(y_pred) # keep -1 since we want to add their labels later on !
        y_unique = y_unique[y_unique > -1] # boundary terms are marked by -1 for now .

        n_cluster = len(y_unique)
        n_iteration = (n_cluster*(n_cluster-1))/2
        n_edge = self.n_edge
        
        self.cout('Performing classification sweep over %i pairs of clusters'%n_iteration)
        self.cout('Training on input of dimension %i, %i'%X.shape)
        
        self.y_pred = y_pred # shallow copy
        self.y_max = np.max(y_unique)

        if self.clf_type == 'rf':
            clf_args_pre = {'class_weight':'balanced', 'n_estimators': 30, 'max_features': min([X.shape[1],200])}
        elif self.clf_type == 'svm':
            clf_args_pre = {'class_weight':'balanced', 'kernel': 'linear'}
        
        info = 'CLF pre-parameters:\t'+("n_bootstrap = %i"%n_bootstrap_shallow)+',clf_args='+str(clf_args_pre)
        self.cout(info)

        # Looping over all possible edges 
        # This is O(N^2) complexity in the number of clusters ...
        self.score = TupleDict()
        for i, yu1 in enumerate(y_unique): 
            for j, yu2 in enumerate(y_unique):
                if i < j:
                    idx_tuple = (yu1, yu2)
                    clf = self.classify_edge(idx_tuple, X,self.y_pred, clf_args=clf_args_pre,n_bootstrap=n_bootstrap_shallow)
                    edge_info(idx_tuple, clf.cv_score, clf.cv_score_std, self.cv_score_threshold, fout=self.fout)
                    self.score[idx_tuple] = clf.cv_score - clf.cv_score_std # SCORE definition here

        # Sorting, keeping only worst edges for kNN 
        edge_list = []
        for i, yu1 in enumerate(y_unique):
            score_i = []
            idx_tmp = []
            for j, yu2 in enumerate(y_unique):
                if i != j:
                    score_i.append(self.score[(yu1, yu2)])
                    idx_tmp.append(yu2)
            asort = np.argsort(score_i)[:self.n_edge] # connectivity
            idx_tmp = np.array(idx_tmp)
            edge_list.append([yu1, idx_tmp[asort]])
    
        edge_info_raw(edge_list, self.score, cout = self.cout)
        del self.score

        ######################################
        ######################################

        # Now looping over those edges and making sure scores are accurately estimated (IMPORTANT)
        self.graph = TupleDict()
        self.cluster_idx = set(y_unique)

        self.cout("Performing deeper sweep over worst edges")
        info = 'CLF post-parameters:\t'+("n_bootstrap = %i"%self.n_bootstrap)+'\t'+str(self.clf_args)
        self.cout(info)

        # This is O(N) in the number of clusters
        for edge in edge_list:
            node_1, nn_node_1 = edge
            for node_2 in nn_node_1:
                idx_edge = (node_1, node_2)
                if idx_edge not in self.graph.keys():
                    clf = self.classify_edge(idx_edge, X, self.y_pred) # using constructor parameters here
                    self.graph[idx_edge] = clf
                    edge_info_update(idx_edge, self.graph, cout=self.cout) # print results
        
        self.compute_edge_score()
        self.compute_node_score()

        return self

    def compute_edge_score(self):
        """ 
            Computing edge weight based off classifier scores
        """
        # This should be recomputed everytime the graph is updated (not comput. expensive)
        
        self.edge = TupleDict()
        eps = 1e-10

        for yu in self.cluster_idx:
            score = []
            nn_yu = self.graph.get_nn(yu)
            for nn in nn_yu:
                clf = self.graph[(yu, nn)]
                self.edge[(yu,nn)] = clf.cv_score - clf.cv_score_std

    def compute_node_score(self):
        
        self.node = dict() # nodes with only one edge have gap of -1.

        for yu in self.cluster_idx:
            nn_yu = self.edge.get_nn(yu)
            edge_ij = []
            for nn in nn_yu:
                edge_ij.append(self.edge[(yu,nn)])
        
            if len(edge_ij) > 1:
                asort = np.argsort(edge_ij)
                gap = edge_ij[asort[1]] - edge_ij[asort[0]]
            else: # what to do if you have only one edge left ?
                gap = -1
            self.node[yu] = gap

    def find_next_merger(self):
        """ Finds the edge that should be merged next based on node score (gap)
        and edge score (edge with minimum score)
        Return
        -------
        edge_to_merge, score_edge, node_gap
        """
        # Go to node with largest gap. Merge it with it's worst edge
        # If node has only one connection ... what to do => nothing, if really bad, will merge with other node (since that one has many connections)
        # If all nodes have gap = -1 (only one pair left), stop
        node, gap = max(self.node.items(), key=lambda x:x[1])

        if gap < 0: # only two nodes left
            node_2 = list(self.edge.get_nn(node))[0]
            edge_to_merge = (node, node_2)
            score_edge = self.edge[(node, node_2)]
            gap = -1
        else:
            nn_yu = list(self.edge.get_nn(node))

            edge_ij = []
            for nn in nn_yu:
                    edge_ij.append(self.edge[(node,nn)])
            arg_minscore_edge = np.argmin(edge_ij)
            
            edge_to_merge = (node, nn_yu[arg_minscore_edge])
            score_edge = edge_ij[arg_minscore_edge]

        return edge_to_merge, score_edge, gap

    def merge_edge(self, edge_tuple, X, y_pred):
        # When merging, need to recompute scores for new edges.
        # Step 0. Find a new label
        self.y_pred = y_pred
        y_new = self.y_max+1
        self.y_max +=1

        node_1, node_2 = edge_tuple

        # Step 1. Relabel edge nodes
        pos_idx_1 = np.where(self.y_pred == node_1)[0]
        pos_idx_2 = np.where(self.y_pred == node_2)[0]
        self.y_pred[pos_idx_1] = y_new
        self.y_pred[pos_idx_2] = y_new

        if self.y_murky is not None:
            # Add back in intra (unpure) cluster elements
            pos_idx_1 = np.where(self.y_murky == node_1)[0]
            pos_idx_2 = np.where(self.y_murky == node_2)[0]

            self.y_pred[pos_idx_1] = y_new
            self.y_pred[pos_idx_2] = y_new
            self.y_murky[pos_idx_1] = -1
            self.y_murky[pos_idx_2] = -1

        # Step 2. Recompute edges
        neighbor_node_1 = self.graph.get_nn(node_1) - set([node_2])
        neighbor_node_2 = self.graph.get_nn(node_2) - set([node_1])
        node_list = list(neighbor_node_1.union(neighbor_node_2))

        for node in node_list:
            idx_new_edge = (y_new, node)
            clf = self.classify_edge(idx_new_edge, X, self.y_pred) # using constructor parameters here
            self.graph[idx_new_edge] = clf
            edge_info_update(idx_new_edge, self.graph, cout=self.cout) # print results

            if node in neighbor_node_1:
                del self.graph[(node_1, node)]
            if node in neighbor_node_2:
                del self.graph[(node_2, node)]
            
        del self.graph[(node_1, node_2)]

        self.cluster_idx.remove(node_1)
        self.cluster_idx.remove(node_2)
        self.cluster_idx.add(y_new)

        self.compute_edge_score()
        self.compute_node_score()
        
        return self


    def classify_edge(self, edge_tuple, X, y, clf_type=None, clf_args=None, 
        n_bootstrap=None, test_size_ratio=None, n_sample_max = None):
        """ Trains a classifier for cluster edge_tuple[0] and edge_tuple[1]

        Important attributes are (for CLF object):

            self.scaler_list -> [mu, std]

            self.cv_score -> mean cv score

            self.mean_train_score -> mean train score

            self.clf_list -> list of sklearn classifiers (for taking majority vote)
        
        Returns
        ---------
        CLF object (from classify.py). Object has similar syntax to sklearn's classifier syntax

        """
        if clf_type is None:
            clf_type = self.clf_type
        if clf_args is None:
            clf_args = self.clf_args
        if n_bootstrap is None:
            n_bootstrap = self.n_bootstrap
        if test_size_ratio is None:
            test_size_ratio = self.test_size_ratio
        if n_sample_max is None:
            n_sample_max = self.n_sample_max

        pos_subset = np.where((self.y_pred == edge_tuple[0]) | (self.y_pred == edge_tuple[1]))
        
        Xsubset = X[pos_subset] # original space coordinates
        ysubset = y[pos_subset] # labels

        return CLF(clf_type=clf_type, n_bootstrap=n_bootstrap, n_sample_max=n_sample_max, test_size=test_size_ratio, clf_kwargs=clf_args).fit(Xsubset, ysubset)
    
    def merge_until_robust(self, X_in, cv_robust, ratio_dict):
        """
        Merges cluster based on certainty metric

        X_in : inliers features in the original space
        
        ratio_dict : dict of cluster labels to list of boundary points for the cluster with info [majority_ratio_cluster_label, idx_in]

        --> why not just work with X_in instead ==> so you don't have to reinclude stuff ... !
        """ 

        yunique = np.unique(self.cluster_label) # this includes boundary points.
        yunique = yunique[yunique >=0] # remove boundary !
        self.init_n_cluster = len(yunique)
        self.current_max_label = np.max(yunique) + 1
        self.current_n_merge = 0
        self.history = []
        self.ratio_dict = deepcopy(ratio_dict)
        n_cluster = self.init_n_cluster

        while True:
            all_robust = True
            if n_cluster == 1:
                self.history.append([-1, np.copy(self.cluster_label), deepcopy(self.nn_list), -1, -1])
                break
            worst_effect_cv = 10
            worst_edge = -1
            score_list = []
            edge_list = []
            yunique = np.unique(self.cluster_label) # > how would you prepare this ?
            yunique = yunique[yunique >=0] # remove boundary <== <==

            score_dict = {yu:{} for yu in yunique}

            for edge, clf in self.graph.items():
                effect_score = clf.cv_score - clf.cv_score_std
                score_list.append(effect_score)
                edge_list.append(edge)
                score_dict[edge[0]][edge[1]] = effect_score
                score_dict[edge[1]][edge[0]] = effect_score
                if effect_score < worst_effect_cv:
                    worst_effect_cv = effect_score
                    worst_edge = (edge[0], edge[1])
            
            """ for n1, v in score_dict.items():
                for n2, score_value in v.items():
                    print((n1, n2),' = %.4f '%score_value, end='')
                    print('    ', end='')
                print('\n', end='') """

            certainty_node = {}
            # for each node just assign a score, and mark the other node that it should be merged with
            max_certainty = -1
            for cluster_idx, n_dict in score_dict.items():
                k, v = list(n_dict.keys()), list(n_dict.values())
                asort_edge = np.argsort(v) # we want largest value
                nn = k[asort_edge[0]] # node it should be merged with if certainty is high
                if len(asort_edge) > 1:
                    certainty_value = v[asort_edge[1]]-v[asort_edge[0]] # certainty value for each edge
                else:
                    # here there is only one edge left. So merge this edge if it has the lowest score of all.
                    idx_tmp = (cluster_idx, k) if cluster_idx < k else (k, cluster_idx)
                    if worst_edge == idx_tmp:
                        certainty_value = 10. # trick !

                # ----------------> 
                certainty_node[cluster_idx] = [certainty_value, nn]
                if certainty_value > max_certainty:
                    max_certainty = certainty_value
                    edge_merge = (cluster_idx, nn)

            asort = np.argsort(score_list)
            print('[vgraph.py]    Lowest scores:')
            for aa in asort[:5]:
                print(edge_list[aa], '\t\t', score_list[aa])

            worst_effect_cv = score_list[asort[0]]
            
            if worst_effect_cv < cv_robust:
                all_robust = False
            
            print('[vgraph.py]    Merging edge %i --- %i\t'%(edge_merge[0], edge_merge[1]),'with certainty=\t%.4f'%max_certainty)

            if all_robust is False:

                n_cluster = self.init_n_cluster - self.current_n_merge - 1
                n1, n2 = edge_merge

                merge_info(n1, n2, score_dict[n1][n2], self.current_max_label, n_cluster, fout = self.fout)
                
                n_cluster -= 1
                # info before the merge -> this score goes with these labels            
                self.edge_score

                self.history.append([score_dict[n1][n2], np.copy(self.cluster_label), deepcopy(self.nn_list), (n1,n2, self.current_max_label), deepcopy(self.graph[(n1,n2)])])
                
                #self.reinclude_bounday(X ===> here)

                self.merge_edge(X_in, edge_merge, ratio_dict) # this will modify cluster_labels and ratio_dict.
                # when merging edge should reinclude the boundary point ...
                # 1. this will modify X. 2. this will modify self.cluster_label. 3. This will modify scores (better !)
            
            else:
                self.history.append([worst_effect_cv, np.copy(self.cluster_label), deepcopy(self.nn_list), -1, -1])

                break

    def update_ratio_dict(self, edge_tuple, new_label):
        # edge_tuple : edge being merged
        # >>>> First updating cluster labels :

        i1, i2 = edge_tuple
        remaining_element = {i1:[], i2:[]}

        for ii in [i1,i2]: # go over each cluster of the edge boundaries
            idx_boundary = self.ratio_dict[ii] # if the point is on the boundary between the two being merge, include it in the merge.
            #print(ii, idx_boundary)
            if len(idx_boundary) > 0:

                cond_1 = (idx_boundary[:,0] == i1) & (idx_boundary[:,1] == i2) # is this condition ok ?
                cond_2 = (idx_boundary[:,1] == i1) & (idx_boundary[:,0] == i2)
                cond = cond_1 | cond_2

                idx_in_boundary = idx_boundary[cond, 2] # idx w.r.t. to .cluster_label
                # bug here when merging with the boundary --> implies that 

                tmp = idx_boundary[(cond == False)]# points that are merged with other clusters...
                remaining_element[ii] = tmp
                self.cluster_label[idx_in_boundary] = new_label # ok once this is done, need to update ratio dict as well ... for neighbors and remove cluster

        remain = []
        for k, v in remaining_element.items():
            for e in v:
                remain.append(e)
        if len(remain) > 0:
            self.ratio_dict[new_label] = np.vstack(remain) # updated boundary.
        else:
            self.ratio_dict[new_label] = [] # updated boundary.

        for k, v in self.ratio_dict.items():
            if len(v) > 0:
                pos = (v[:,0] == i1) | (v[:,0] == i2)
                v[pos,0] = new_label
                pos = (v[:,1] == i1) | (v[:,1] == i2)
                v[pos,1] = new_label
    
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
        
    def plot_kNN_graph(self, idx_pos, X=None, savefile=None):
        from plotlygraph import plot_graph
        #print(idx_pos)
        plot_graph(self.edge, idx_pos, self.node, X=X, title='k-NN graph', savefile=savefile, n_sample=20000)

def edge_info_raw(edge_list, score, cout=print):
    cout("Edges that will be used ...")
    for edge in edge_list:
        i, nn_i  = edge
        for j in nn_i:
            cout("{0:^5d}".format(i)+' ---- '+"{0:^5d}".format(j)+'  =~  '+"%.4f"%score[(i,j)])

def edge_info(edge_tuple, cv_score, std_score, min_score, fout=None, cout = print):
    
    edge_str = "{0:5<d}{1:4<s}{2:5<d}".format(edge_tuple[0]," -- ",edge_tuple[1])

    robust_or_not = "robust edge" if cv_score - std_score > min_score else "reject edge "

    out = "{0:<15s}{1:<15s}{2:<15s}{3:<7.4f}{4:<16s}{5:>6.5f}".format(robust_or_not, edge_str, "score =", cv_score, "\t+-",std_score)

    cout(out)

    if fout is not None:
        fout.write(out)

def edge_info_update(edge_tuple, graph, cout=print):
    
    edge_str = "{0:5<d}{1:4^s}{2:5<d}".format(edge_tuple[0]," -- ",edge_tuple[1])
    clf = graph[edge_tuple]
    cv, cv_std = clf.cv_score, clf.cv_score_std
    out = "{0:<20s}{1:<10.4f}{2:^10s}{3:<10.4f}".format(edge_str,cv,'+/-',cv_std)
    cout(out)
    
def merge_info(c1, c2, score, new_c, n_cluster, cout=print):
    edge_str = "{0:5<d}{1:4<s}{2:5<d}".format(c1," -- ",c2)
    out = "[vgraph.py]    {0:<15s}{1:<15s}{2:<15s}{3:<7.4f}{4:<16s}{5:>6d}{6:>15s}{7:>5d}".format("merge edge ",edge_str,"score - std =",score,
    "\tnew label ->",new_c,'n_cluster=',n_cluster)
    cout(out)

def graph_cout(s):
    print("[graph] %s"%s)



