#from .fdc import FDC
#from .classify import CLF
from fdc.classify import CLF
import numpy as np
import pickle
from scipy.cluster.hierarchy import dendrogram as scipydendroed
from scipy.cluster.hierarchy import to_tree
#from .hierarchy import compute_linkage_matrix
import copy
from collections import OrderedDict as OD
from collections import Counter

class TREENODE:

    def __init__(self, id_ = -1, parent = None, child = None, scale = -1):
        if child is None:
            self.child = [] # has to be list of TreeNode
        else:
            self.child = child
        self.scale = scale
        self.parent = parent
        self.id_ = id_

    def __repr__(self):
        return ("Node: [%s] @ s = %.3f" % (self.id_,self.scale)) 

    def is_leaf(self):
        return len(self.child) == 0

    def get_child(self, id_ = None):
        if id_ is None:
            return self.child
        else:
            for c in self.child:
                if c.get_id() == id_:
                    return c

    def get_child_id(self):
        if len(self.child) == 0:
            return []
        else:
            return [c.id_ for c in self.child]

    def get_scale(self):
        return self.scale

    def get_id(self):
        return self.id_

    def add_child(self, treenode):
        self.child.append(treenode)
    
    def remove_child(self, treenode):
        self.child.remove(treenode)
        
    def get_rev_child(self):
        child = self.child[:]
        child.reverse()
        return child 
    
class TREE:
    """ Contains all the hierachy and information concerning the clustering """
    
    def __init__(self, merge_and_clf, clf_args, test_size_ratio=0.8):
        # merge_and_clf = list([idx_1, idx_2, edge_clf])
        self.merge_and_clf = merge_and_clf
        self.clf_args = clf_args
        self.test_size_ratio = test_size_ratio

    def fit(self, X): 
        """ X is the original space (boundary + pure) 
        """
        #[score_dict[n1][n2], np.copy(self.cluster_label), deepcopy(self.nn_list),(n1,n2, self.current_max_label),deepcopy(self.graph[(n1,n2)])])
        ypred = self.merge_and_clf[-1][1]
        n_merge = len(self.merge_and_clf)

        clf_top =  CLF(clf_type='rf', n_average=5, test_size=self.test_size_ratio, clf_args=self.clf_args).fit(X, ypred)
        y_unique = np.unique(ypred)

        self.node_dict = OD()
        self.clf_dict = OD()
        self.root = TREENODE(id_ = max(y_unique)+1, scale=clf_top.cv_score)
        
        self.node_dict[self.root.get_id()] = self.root
        self.clf_dict[self.root.get_id()] = clf_top

        for yu in y_unique:
            c_node = TREENODE(id_=yu, parent=self.root)
            self.root.add_child(c_node)
            self.node_dict[c_node.get_id()] = c_node

        # building full tree here
        for nm in range(n_merge-2,0,-1): # already checked last "merge", 
            idx_1, idx_2, idx_merge = self.merge_and_clf[nm][3]
            #print(idx_1,idx_2, idx_merge)
            clf = self.merge_and_clf[nm][4]
            self.clf_dict[idx_merge] = clf
            self.node_dict[idx_merge].scale = clf.cv_score

            c_node = TREENODE(id_ = idx_1, parent = self.node_dict[idx_merge])
            self.node_dict[c_node.get_id()] = c_node
            self.node_dict[idx_merge].add_child(c_node)

            c_node = TREENODE(id_ = idx_2, parent = self.node_dict[idx_merge])
            self.node_dict[c_node.get_id()] = c_node
            self.node_dict[idx_merge].add_child(c_node)
        
    def predict(self, X, cv=0.9):
        print('Predicting on %i points'%len(X))
        #import time
        c_node = self.root
        ypred=-1*np.ones(len(X), dtype=int)
        for i, x in enumerate(X):
            if (i+1) % 1000 == 0:
                print("[tree.py]   Prediction done on %i points"%(i+1))
            c_node = self.root
            score = c_node.scale
            while score > cv :
                new_id = self.clf_dict[c_node.get_id()].predict([x], option='fast')
                c_node = self.node_dict[new_id[0]]
                score = c_node.scale
            ypred[i] = c_node.parent.get_id()

        return ypred

    def feature_path_predict(self, x, cv=0.9):

        c_node = self.root
        feature_important = []
        c_node = self.root
        score = c_node.scale

        while score > cv :
            new_id = self.clf_dict[c_node.get_id()].predict([x], option='fast')
            feature_important.append(self.clf_dict[c_node.get_id()].feature_importance())
            c_node = self.node_dict[new_id[0]]
            score = c_node.scale

        return c_node.parent.get_id(), feature_important
        
    def save(self, name=None):
        """ Saves current model to specified path 'name' """
        if name is None:
            name = self.make_file_name()
        fopen = open(name,'wb')
        pickle.dump(self,fopen)
        fopen.close()
        
    def load(self, name=None):
        if name is None:
            name = self.make_file_name()

        self.__dict__.update(pickle.load(open(name,'rb')).__dict__)
        return self

    def make_file_name(self):
        t_name = "clf_tree.pkl"
        return t_name

def str_gate(marker, sign):
    if sign < 0. :
        return marker+"-"
    else:
        return marker+"+"

def apply_map(mapdict, k):
    old_idx = k
    while True:
        new_idx = mapdict[old_idx]
        if new_idx == -1:
            break
        old_idx = new_idx
    return old_idx


def float_equal(a,b,eps = 1e-6):
    if abs(a-b) < 1e-6:
        return True
    return False


def get_scale(Z, c_1, c_2):
    for z in Z:
        if (z[0],z[1]) == (c_1,c_2) or (z[0],z[1]) == (c_2,c_1):
            return z[2]
    return -1
        
def breath_first_search(root):
    """
    Returns
    -------
    node_list : list of node id contained in root 
    """

    stack = [root]
    node_list = []
    # breath-first search
    while stack:
        current_node = stack[0]
        stack = stack[1:]
        node_list.append(current_node.get_id()) 

        if not current_node.is_leaf():
            for node in current_node.get_child():
                stack.append(node)

    return node_list
    
def find_idx_cluster_in_root(model, node):
    """ Finds the original (noise_threshold = init) clusters contains in the node
    Returns the index of the terminal nodes contained in node.
    """
    node_list = np.array(breath_first_search(node)) #list of terminal nodes contained in node. 
    n_initial_cluster = len(model.hierarchy[0]['idx_centers']) # map out what is going on here .
    # recall that the cluster labelling is done following the dendrogram convention (see scipy)
    return np.sort(node_list[node_list < n_initial_cluster]) # subset of initial clusters contained in the subtree starting at node