from .classify import CLF
import numpy as np
import pickle
import copy, time
from collections import OrderedDict as OD

class TREENODE:

    def __init__(self, id_ = -1, parent = None, child = None, scale = -1):
        if child is None:
            self.child = [] # has to be list of TreeNode
        else:
            self.child = child
        self.scale = scale
        self.parent = parent
        self.id_ = id_
        self.info = {} # extra information !

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
    
    def __init__(self, merge_history, cluster_statistics, clf_type, clf_args, test_size_ratio=0.8):
        #merger_history = list of  [edge, clf]

        self.merge_history = merge_history
        self.cluster_statistics = cluster_statistics
        self.clf_args = clf_args
        self.clf_type = clf_type
        self.test_size_ratio = test_size_ratio

    def fit(self, X, y_pred, n_bootstrap=10): 
        """ 

        """
        #[score_dict[n1][n2], np.copy(self.cluster_label), deepcopy(self.nn_list),(n1,n2, self.current_max_label),deepcopy(self.graph[(n1,n2)])])
        
        pos = (y_pred > -1)
        y_unique = np.unique(y_pred)
        idx_subset = np.where(pos)

        clf_root = CLF(clf_type=self.clf_type, n_bootstrap=n_bootstrap, test_size=self.test_size_ratio, clf_kwargs=self.clf_args).fit(X[idx_subset], y_pred[idx_subset])
        self.root = TREENODE(id_ = y_unique[-1]+1 , scale=clf_root.cv_score)

        # cluster statistics for the root 
        self.cluster_statistics[self.root.get_id()]  = {"mu":np.median(X[idx_subset], axis=0),
                                                        "std":np.std(X[idx_subset],axis=0),
                                                        "size":len(idx_subset),
                                                        "feature":[]}

        self.node_dict = OD()
        self.clf_dict = OD()
        
        self.node_dict[self.root.get_id()] = self.root
        self.clf_dict[self.root.get_id()] = clf_root

        for yu in y_unique:
            c_node = TREENODE(id_=yu, parent=self.root)
            self.root.add_child(c_node)
            self.node_dict[c_node.get_id()] = c_node

        # building full tree here
        for edge, y_new, clf in reversed(self.merge_history):
            idx_1, idx_2 = edge

            self.clf_dict[y_new] = clf
            self.node_dict[y_new].scale = clf.cv_score

            for idx in edge:
                c_node = TREENODE(id_ = idx, parent = self.node_dict[y_new])
                self.node_dict[c_node.get_id()] = c_node
                self.node_dict[y_new].add_child(c_node)
        
        return self
        
    def predict(self, X_, cv = 0.9, option='fast'):
        """
        Prediction on the original space data points
        """

        if X_.ndim == 1:
            X = X_.reshape(-1,1)
        else:
            X = X_

        print('Predicting on %i points'%len(X))

        stack = [self.root]
        ypred = self.clf_dict[stack[0].get_id()].predict(X, option=option)

        while stack:
            child = self.node_dict[stack[0].get_id()].child # node list (not integers)
            stack = stack[1:]
            for c in child:
                if c.scale > cv: # c.scale (unpropagated scores)
                    stack.append(c)
                    pos = (ypred == c.get_id())
                    ypred[pos] = self.clf_dict[c.get_id()].predict(X[pos], option=option)
        return ypred

    def plot_tree(self, X, cv, out="nested"):
        
        self.compute_feature_importance_dict(X)
        self.compute_node_info()

        print(self.node)
        exit()

        stack = [self.root]
        if out is "nested":
            while stack:
                child = self.node_dict[stack[0].get_id()].child
                stack = stack[1:]
                for c in child:
                    if c.scale > cv: # node division has high enough probability
                        stack.append(c)

    """ def add_node_nested(name,info,median_markers,feature_importance,cv):
        return {'cv':cv,'name':name,'info':info,'median_markers':median_markers,'feature_importance':feature_importance} """

    def compute_feature_importance_dict(self, X):
        """ --> this only works for random forest <- """
        assert self.clf_type == "rf", "Need to use random forest ('rf' option)"
        self.feature_importance_dict = {}

        for node_id, clf in self.clf_dict.items():
            importance_matrix = np.vstack([c.feature_importances_ for c in clf.clf_list])
            scores = np.mean(importance_matrix, axis=0)
            std_scores = np.std(importance_matrix, axis=0)
            self.feature_importance_dict[node_id] = [scores, std_scores]

    def compute_node_info(self):

        for node_id, node in self.node_dict.items():

            node.info = {
                'cv': node.scale, # need to update this to reflect the fact that depth propagates errors
                'leaf': (len(node.child) == 0),
                'name': node_id,
                'extra': [],
                'median_marker': self.cluster_statistics[node_id]
            }

            if node.info['leaf'] is True:
                node.info['feature_importance'] = []
                node.info['children_id'] = []
            else:
                node.info['feature_importance'] = self.feature_importance_dict[node_id]
                node.info['children_id'] = [c.id_ for c in node.child]
 
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