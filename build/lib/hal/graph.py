from .classify import CLF
from .tree import TREE
import numpy as np
from copy import deepcopy
import pickle, time
from .tupledict import TupleDict
from .utility import FOUT, compute_cluster_stats

## == <==> 
#### GOAL TODAY ---> make this kNN graph more clear and better sort out options and structures, etc.
## == <==>

class SINGLE_EDGE:

    def __init__(self, i, j, clf):
        self.edge = (i, j)
        self.score = clf.cv_score
        self.score_median = clf.cv_score_median
        self.score_error = clf.cv_score_std
        self.score_error_iqr = clf.cv_score_iqr
    
    def LCB(self):
        # Lower confidence bound
        return self.score - self.score_error
    
    def LCB_robust(self):
        return self.score_median - self.score_error_iqr
    
    
class kNN_Graph:
    """ Validation graph class - builds a graph with nodes corresponding
    to clusters and draws edges between clusters that have a low validation score

    Parameters
    ------------------

    n_bootstrap: int (default = 10)
        Number of classifiers trained on each pair of clusters (the cv score is the average validation error)
    
    test_size_ratio: float (default = 0.8)
        Proportion of the data kept for testing
    
    n_sample_max: int (default = 1000)
        Number of 

    """
    def __init__(self, 
        n_bootstrap = 10,  
        test_size_ratio = 0.8,
        n_sample_max = 1000,
        clf_type='svm', 
        clf_args=None,
        verbose = 1,
        n_edge =2,
        y_murky= None,
        gap_min = 0.01
        ):
        
        self.n_bootstrap = n_bootstrap
        self.test_size_ratio = test_size_ratio
        self.clf_type = clf_type
        self.n_sample_max = n_sample_max
        self.recomputed = False
    
        if clf_args is None:
            if clf_type == 'svm':
                self.clf_args = {'kernel':'linear','class_weight':'balanced'}
            else:
                self.clf_args = {'class_weight':'balanced'}
        else:
            self.clf_args = clf_args

        self.fout = None#/FOUT('out.txt')
        self.n_edge = n_edge
        self.gap_min = gap_min # self-consistent way of determining this ?
        self.gap_option = "standard"
        self.y_murky = y_murky
        self.cluster_statistics = {} # node_id to median markers ... should be part of graph stuff ?
        self.merger_history = []
        self.cout = graph_cout if verbose is 1 else lambda *a, **k: None
        print("kNN-graph options:", self.__dict__)
        
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
            if self.clf_args is None:
                clf_args_pre = {'class_weight':'balanced', 'n_estimators': 30, 'max_features': min([X.shape[1],200])}
            else:
                clf_args_pre = self.clf_args
        elif self.clf_type == 'svm':
            if self.clf_args is None:
                clf_args_pre = {'class_weight':'balanced', 'kernel': 'linear'}
            else:
                clf_args_pre = self.clf_args
        elif self.clf_type == "nb":
            clf_args_pre = {}
        
        info = 'CLF pre-parameters:\t'+("n_bootstrap = %i"%n_bootstrap_shallow)+',clf_args='+str(clf_args_pre)

        self.cout(info)

        # Looping over all possible edges 
        # This is O(N^2) complexity, where N is number of clusters 
        self.score = TupleDict()

        for i, yu1 in enumerate(y_unique): 
            for j, yu2 in enumerate(y_unique):
                if i < j:
                    idx_tuple = (yu1, yu2)
                    clf = self.classify_edge(idx_tuple, X, self.y_pred, clf_args=clf_args_pre, n_bootstrap=n_bootstrap_shallow)
                    edge_info(idx_tuple, clf.cv_score_median, 0, fout=self.fout) #clf.cv_score_std, fout=self.fout)
                    self.score[idx_tuple] = clf.cv_score_median # - clf.cv_score_std # SCORE definition here

        # ======
        # Sorting, keeping only the k worst edges for each nodes
        # ====== =====================
        edge_list = []
        for i, yu1 in enumerate(y_unique):
            score_i = []
            idx_tmp = []
            for j, yu2 in enumerate(y_unique):
                if i != j:
                    score_i.append(self.score[(yu1, yu2)])
                    idx_tmp.append(yu2)
            asort = np.argsort(score_i)[:self.n_edge] # -- > > connectivity < < --
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

        # Compute basic cluster statistics (leaf nodes):
        for yu in np.unique(y_pred):
            self.cluster_statistics[yu] = compute_cluster_stats(X[self.y_pred==yu], len(self.y_pred))
        return self

    def compute_node_statistics(self, X, ypred, ynode):
        pos = (ypred == yu)
        median = np.median(X[pos],axis=0)
        std = np.std(X[pos],axis=0)
        size_ = np.count_nonzero(pos)
        self.cluster_statistics[ynode] = {"mu":median,"std":std, "size":size_, "ratio":size_/len(pos)}

    def compute_edge_score(self):
        """ 
            Computing edge weight based off classifier scores
        """
        # This should be recomputed everytime the graph is updated (not comput. expensive)
        
        self.edge_graph = TupleDict()
    
        eps = 1e-10

        for yu in self.cluster_idx:
            score = []
            nn_yu = self.graph.get_nn(yu)
            for nn in nn_yu:
                clf = self.graph[(yu, nn)]
                self.edge_graph[(yu, nn)] = SINGLE_EDGE(yu, nn, clf)

    def compute_node_score(self):
        # Computes edge 
    
        self.node = dict() # nodes with only one edge have gap of -1.

        for yu in self.cluster_idx:

            nn_idx_from_yu = list(self.edge_graph.get_nn(yu))
            edge_score_from_yu = []
            for nn in nn_idx_from_yu:
                edge_score_from_yu.append(self.edge_graph[(yu,nn)].LCB_robust())
        
            if len(edge_score_from_yu) > 1:
                asort=np.argsort(edge_score_from_yu)
                nn_from_yu = nn_idx_from_yu[asort[0]]
                nn2_from_yu = nn_idx_from_yu[asort[1]]

                gap = self.edge_graph[(yu, nn2_from_yu)].score_median - self.edge_graph[(yu, nn_from_yu)].score_median
                gap_error = self.edge_graph[(yu, nn2_from_yu)].score_error_iqr + self.edge_graph[(yu,nn_from_yu)].score_error_iqr
                gap_error_prop = gap - gap_error
                gap_LCB = self.edge_graph[(yu, nn2_from_yu)].LCB_robust() - self.edge_graph[(yu, nn_from_yu)].LCB_robust() # -> seems better behaved ?

            else: # what to do if you have only one edge left ?
                gap_LCB = -1
                gap_error_prop = -1

            if self.gap_option is "standard":
                self.node[yu] = gap_LCB
            else:
                self.node[yu] = gap_error_prop

    def find_next_merger(self):
        """ Finds the edge that should be merged next based on node score (gap)
        and edge score (edge with minimum score)

        Return
        -------
        edge_to_merge, score_edge, node_gap
        """
        #node_list, cv_scores = list(self.node.keys()), list(self.node.values())

        tmp = [[k, v.LCB_robust()] for k,v in self.edge_graph.items()]
        edge_tuples = np.array(tmp)[:,0]
        edge_scores = np.array(tmp)[:,1]
        n_edge_current = len(tmp)
        #edge_tuples = list(self.edge_graph.keys())

        # amongst worst edges -> take the node with the largest gap
        idx = np.argsort(edge_scores)[:max([15, int(0.1*n_edge_current)])] #(self.n_edge*3)] # worst edges indices

        # Step 1. Find largest gap, loop over nodes.
        gap = -10
        for i in idx:
            for n in edge_tuples[i]:
                if self.node[n] > gap:
                    gap = self.node[n]
                    node = n

        #print(self.node)

        # Step 2, merges nodes connected to node with largest gap !

        gap_array, idx_argsort, nn_idx_from_node, edge_score_from_node = self.compute_gaps(node)
        
        """ #, nn_idx_from_node)

        for nn in nn_idx_from_node:

            edge_score_from_node.append(self.edge_graph[(node ,nn)].score_median)
            edge_error_score_from_node.append(self.edge_graph[(node ,nn)].score_error)

        edge_score_from_node=np.array(edge_score_from_node)
        edge_error_score_from_node=np.array(edge_error_score_from_node)
        edge_LCB = edge_score_from_node - edge_error_score_from_node
    
        #------------- interchangeable lines ------------
        asort = np.argsort(edge_LCB)
        gap_array = np.diff(edge_LCB[asort]) # don't forget to sort first, (worst edges merged first) 
        ################################################# """

        gap_array = list(gap_array)
        gap_array.append(0.)
        
        hyper_edge_to_merge = [node, nn_idx_from_node[idx_argsort[0]]]
        score_edge = [edge_score_from_node[idx_argsort[0]]]
        if len(gap_array) > 1:
            gap_merge = gap_array[0]
        else:
            gap_merge = gap
        
        #print("all neighbors of %i"%node, nn_idx_from_node)
        #print(nn_idx_from_node[idx_argsort[-1]])
        #print(len(gap_array),'\t', len(idx_argsort))
        
        
        for i in range(1, len(gap_array)):
            if gap_merge > self.gap_min:
                #print("Reached max gap")
                break
            #print(idx_argsort[i])
            node_idx = nn_idx_from_node[idx_argsort[i]]
            #print(node_idx)
            hyper_edge_to_merge.append(node_idx)
            score_edge.append(edge_score_from_node[idx_argsort[i]])
            gap_merge += gap_array[i]
                
        return hyper_edge_to_merge, score_edge, gap_merge
    
    def compute_gaps(self, node):

        nn_idx_from_node = list(self.edge_graph.get_nn(node))
        
        edge_score = [] 
        edge_error = []

        for nn in nn_idx_from_node:

            """ if self.gap_option is "standard":
                edge_score.append(self.edge_graph[(node ,nn)].LCB())
                edge_error.append(self.edge_graph[(node ,nn)].score_error)
            else: """
            edge_score.append(self.edge_graph[(node ,nn)].LCB_robust())
            edge_error.append(self.edge_graph[(node ,nn)].score_error_iqr)
        
        edge_score=np.array(edge_score)
        edge_error=np.array(edge_error)

        asort = np.argsort(edge_score)
        gap_array = np.diff(edge_score[asort])
        gap_errors = edge_error[asort][:-1]+edge_error[asort][1:]

        if self.gap_option is "standard":
            return gap_array, asort, nn_idx_from_node, edge_score
        else:
            return gap_array-gap_errors, asort, nn_idx_from_node, edge_score


    def merge_edge(self, edge_tuple, X, y_pred):

        # Need to be able to merge hyper edge

        if len(edge_tuple) > 2: # compute multi-class classifier 
            clf_hyper_edge = self.classify_edge(edge_tuple, X, self.y_pred)
        else: # has already been computed, no need to recompute ...
            clf_hyper_edge = self.graph[tuple(edge_tuple)]

        # When merging, need to recompute scores for new edges.
        # Step 0. Find a new label

        self.y_pred = y_pred
        y_new = self.y_max+1
        self.y_max +=1

        for node_idx in edge_tuple:
            self.y_pred[self.y_pred == node_idx] = y_new

        # Step 1. Relabel edge nodes

        if self.y_murky is not None:
            # Add back in intra (unpure) cluster elements
            for node_idx in edge_tuple:
                pos_idx = np.where(self.y_murky == node_idx)[0]
                self.y_pred[pos_idx] = y_new
        
        nodes_to_be_merged = set(edge_tuple)
        neighboring_nodes = set([])
        for node_idx in edge_tuple:
            neighboring_nodes = neighboring_nodes.union(self.graph.get_nn(node_idx))
        
        neighboring_nodes = list(neighboring_nodes - nodes_to_be_merged)

        # Step 2. Recompute edges

        for node in neighboring_nodes:
            idx_new_edge = (y_new, node)
            clf = self.classify_edge(idx_new_edge, X, self.y_pred) # using constructor parameters here
            self.graph[idx_new_edge] = clf
            edge_info_update(idx_new_edge, self.graph, cout=self.cout) # print results

            for node_tbm in nodes_to_be_merged:
                if node in self.graph.get_nn(node_tbm):
                    del self.graph[(node_tbm, node)]
        
        self.merger_history.append([edge_tuple, y_new, deepcopy(clf_hyper_edge)]) # this saves the classifiers for later

        self.remove_edge_tuple(edge_tuple)

        for node_idx in edge_tuple:
            self.cluster_idx.remove(node_idx)

        self.cluster_idx.add(y_new)

        if len(self.graph.keys()) > 0:
            self.compute_edge_score()

            self.compute_node_score()

        self.cluster_statistics[y_new] = compute_cluster_stats(X[self.y_pred==y_new], len(self.y_pred))

        return self

    def remove_edge_tuple(self, edge_tuple):
        """ Deletes edges in graph (frees up memory) """
        for node_1 in list(edge_tuple):
            for node_2 in list(edge_tuple):
                if (node_1, node_2) in self.graph.keys():
                    del self.graph[(node_1, node_2)] # clearing memory

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

        pos_subset = np.zeros(len(self.y_pred),dtype=bool)
        for node_idx in edge_tuple:
            pos_subset = (pos_subset | (self.y_pred == node_idx))

        Xsubset = X[pos_subset] # original space coordinates
        ysubset = y[pos_subset] # labels ---

        return CLF(clf_type=clf_type, n_bootstrap=n_bootstrap, n_sample_max=n_sample_max, test_size=test_size_ratio, clf_kwargs=clf_args).fit(Xsubset, ysubset)
    
    def coarse_grain(self, X, y_pred):

        while len(self.graph) > 0: # merge until at most one edge is left #np.max(list(self.node.values())) > 0:
            edge, score, gap = self.find_next_merger()
            #print(self.graph)
            print('Merging edge\t', edge,'\t gap=',gap,'\t score=',score)
            self.merge_edge(edge, X, y_pred)
        
            print('\n\n')

    def build_tree(self, X, ypred_init):
        print("Building tree")
        self.tree = TREE(self.merger_history, self.cluster_statistics,self.clf_type,self.clf_args, ypred_init, test_size_ratio=self.test_size_ratio)
        self.tree.fit(X, self.y_pred)

    def predict(self, X, cv=0.5, option="fast", gap=None):
        print('-> Predicting on %i points'%len(X))
        return self.tree.predict(X, cv=cv, gap=gap, option=option)

    # Plotly graph
    #def plot_kNN_graph(self, idx_pos, X=None, savefile=None):
    #    from .plotlygraph import plot_graph
    #    plot_graph(self.edge, idx_pos, self.node, X=X, title='k-NN graph', savefile=savefile, n_sample=20000)

def edge_info_raw(edge_list, score, cout=print):
    cout("Edges that will be used ...")
    for edge in edge_list:
        i, nn_i  = edge
        for j in nn_i:
            cout("{0:^5d}".format(i)+' ---- '+"{0:^5d}".format(j)+'  =~  '+"%.4f"%score[(i,j)])

def edge_info(edge_tuple, cv_score, std_score, fout=None, cout = print):
    
    edge_str = "{0:5<d}{1:4<s}{2:5<d}".format(edge_tuple[0]," -- ",edge_tuple[1])

    out = "{0:<15s}{1:<15s}{2:<15s}{3:<7.4f}{4:<16s}{5:>6.5f}".format("[graph]", edge_str, "score =", cv_score, "\t+-",std_score)

    cout(out)

    if fout is not None:
        fout.write(out)

def edge_info_update(edge_tuple, graph, cout=print):
    
    edge_str = "{0:5<d}{1:4^s}{2:5<d}".format(edge_tuple[0]," -- ",edge_tuple[1])
    clf = graph[edge_tuple]
    cv, cv_std = clf.cv_score, clf.cv_score_std
    out = "{0:<20s}{1:<10.4f}{2:^10s}{3:<10.4f}".format(edge_str,cv,'+/-',cv_std)
    cout(out)
    
def graph_cout(s):
    print("[graph] %s"%s)



