from .classify import CLF
import numpy as np
import pickle
import os
import copy, time
from collections import OrderedDict as OD
from .utility import compute_cluster_stats
import json


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

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
    
    def __init__(self, merge_history, cluster_statistics, clf_type, clf_args, y_pred_init, test_size_ratio=0.8):
        #merger_history = list of  [edge, clf]

        self.merge_history = merge_history
        self.cluster_statistics = cluster_statistics # computed in merging, just need to add root
        self.y_pred_init = y_pred_init # for computing f1-scores
        self.clf_args = clf_args
        self.clf_type = clf_type
        self.test_size_ratio = test_size_ratio

    def fit(self, X, y_pred, n_bootstrap=10): 
        """ Fits hierarchical model"""

        from .plotting import cluster_w_label

        #cluster_w_label(X,y_pred)
        #exit()
        
        pos = (y_pred > -1)
        y_unique = np.unique(y_pred[pos]) # fit only on the non-outliers
        idx_subset = np.where(pos)[0]
        y_new_tmp = y_unique[-1]+1

        clf_root = CLF(clf_type=self.clf_type, n_bootstrap=n_bootstrap, test_size=self.test_size_ratio, clf_kwargs=self.clf_args).fit(X[idx_subset], y_pred[idx_subset])
        self.root = TREENODE(id_ = y_new_tmp , scale=clf_root.cv_score)
    
        # cluster statistics for the root 
        self.cluster_statistics[self.root.get_id()]  = compute_cluster_stats(X[idx_subset], len(y_pred))

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

        self.merge_history.append([list(y_unique), y_new_tmp, copy.deepcopy(clf_root)])

        # constructs the structures that have the information about the tree 

        ### Left it here ... how to update this thing ?
        self.compute_feature_importance_dict(X)
        self.compute_node_info()
        self.find_idx_in_each_node()

        return self
        
    def predict(self, X_, cv = 0.9, option='fast'):
        """
        Prediction on the original space data points

        """

        if X_.ndim == 1:
            X = X_.reshape(-1,1)
        else:
            X = X_

        stack = [self.root]
        ypred = self.clf_dict[stack[0].get_id()].predict(X, option=option)

        while stack:
            child = self.node_dict[stack[0].get_id()].child # node list (not integers)
            stack = stack[1:]
            for c in child:
                if c.scale > cv: # c.scale (unpropagated scores)
                    stack.append(c)
                    idx = np.where(ypred == c.get_id())[0]
                    if len(idx) > 0:
                        ypred[idx] = self.clf_dict[c.get_id()].predict(X[idx], option=option)
                    
        return ypred

    """ def compute_f1_score(self, X, node_id, ypred_init):
        # to relabel points according to nodes ... just start from ypred init and apply mergers
        ytmp = np.copy(ypred_init)
        stack = [node_id]
        leaf_list = []

        while stack:
            current_node = stack[0]
            stack = stack[1:]

            if self.node_dict[current_node].info['leaf']:
                leaf_list.append(current_node)
            else:
                for c in self.node_dict[current_node].info['children_id']:
                    stack.append(c)
        
        pos = np.zeros(len(ypred_init), dtype=bool)
        leaf_list = np.array(leaf_list, dtype=int)

        for leaf in leaf_list:
            pos = (pos | (ytmp == leaf))

        idx_true = np.where(pos)[0]
        idx_false = np.where(pos ^ np.ones(len(ypred_init), dtype=bool))[0]

        idx_true_sample = np.random.choice(idx_true, size=min([200,len(idx_true)]), replace=False)
        idx_false_sample = np.random.choice(idx_false, size=min([200,len(idx_false)]), replace=False)

        ypred_true = self.predict(X[idx_true_sample], cv=-1)
        ypred_false = self.predict(X[idx_false_sample], cv=-1)

        for leaf in leaf_list:
            ypred_true[ypred_true == leaf] = node_id
            ypred_false[ypred_false == leaf] = node_id
        
        TP = np.count_nonzero(ypred_true == node_id)/len(ypred_true)   # true positives
        FN = np.count_nonzero(ypred_true != node_id)/len(ypred_true) # false negative
        FP = np.count_nonzero(ypred_false == node_id)/len(ypred_false) # false positives

        R = TP/(TP + FN)
        P = TP/(TP + FP)

        F1 = 0.5*(R*P)/(R+P)

        return F1  """
        
    def plot_tree(self, Xtsne, idx, feature_name): # plot tree given the specified cv score 
        """ Construct a nested dictionary representing the tree along with principal information
        and exports the nested dictornary in a json file (to be read by javascript program)
        """

        
        dashboard_information = {'feature_name':feature_name,'nestedTree':{}}
        
        self.construct_nested_dict(dashboard_information['nestedTree'], self.root.id_)

        if not os.path.exists("js"):
            os.makedirs("js")

        with open('js/tree.json','w') as f:
            f.write(json.dumps(dashboard_information, cls=MyEncoder))
        
        with open('js/idx_merge.json','w') as f:
            tmp = {str(k) : list(map(int,node.info['idx_merged'])) for k, node in self.node_dict.items()}
            f.write(json.dumps(tmp,cls=MyEncoder))

        with open('js/tsne.json','w') as f:
            f.write(json.dumps({"x":list(Xtsne[:,0]),"y":list(Xtsne[:,1]),"idx":list(map(int,idx))},cls=MyEncoder))
        
    def construct_nested_dict(self, nested_dict, node_id):
        ## Need a more structured approach here, but for now it's ok 
        node = self.node_dict[node_id]
        nested_dict["name"] = str(node_id)
        nested_dict["info"] = "size=%.4f, cv=%.3f"%(self.cluster_statistics[node_id]["ratio"],node.info["cv"])
        nested_dict["median_markers"] = list(node.info["median_marker"]["mu"])
        nested_dict["std_markers"] = list(node.info["median_marker"]["std"])
        if node.info["cv"] < 0:
            nested_dict["cv"] = "leaf"
        elif node.info["cv"] > 0.99:
            nested_dict["cv"] = "1.0"
        else:
            nested_dict["cv"] = ("%.3f"%node.info["cv"])[1:]
        nested_dict["f1"] = 0 #"%.3f"%node.info["f1"] # this is not quite working

        if len(node.info["children_id"]) == 0:
            nested_dict["feature_importance"]=[0]*len(nested_dict["median_markers"])
            return nested_dict
        else:
            nested_dict["feature_importance"]=list(node.info["feature_importance"][0]) # second element is std
            nested_dict["children"] = []
            for node_id_2 in node.info["children_id"]:
                nested_dict["children"].append(self.construct_nested_dict({}, node_id_2))
            return nested_dict

    def compute_feature_importance_dict(self, X):
        """ -> feature importance computation <--  (valid for rf and linear svm) """
        self.feature_importance_dict = {}

        if self.clf_type == "rf": #"Need to use random forest ('rf' option)"
            for node_id, clf in self.clf_dict.items():
                importance_matrix = np.vstack([c.feature_importances_ for c in clf.clf_list])
                scores = np.mean(importance_matrix, axis=0)
                std_scores = np.std(importance_matrix, axis=0)
                self.feature_importance_dict[node_id] = [scores, std_scores]
        elif self.clf_type == "svm":
            for node_id, clf in self.clf_dict.items():
                importance_matrix = np.vstack([c.coef_ for c in clf.clf_list]) # linear weights ... 
                scores = np.mean(importance_matrix, axis=0)
                std_scores = np.std(importance_matrix, axis=0)
                self.feature_importance_dict[node_id] = [scores, std_scores]
        else:
            for node_id, clf in self.clf_dict.items():
                self.feature_importance_dict[node_id] = [np.ones(X.shape[1]),np.ones(X.shape[1])]
    
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
        #print(node.info[)

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

    def find_idx_in_each_node(self):
        for node in self.node_dict.values():
            if node.is_leaf():
                node.info['idx_merged'] = [node.get_id()]
            else:
                node.info['idx_merged'] = []

        for mh in self.merge_history: # start from the bottom
            idx_merged, new_idx, _  = mh
            node = self.node_dict[new_idx]

            for idx in idx_merged:
                tmp = self.node_dict[idx].info['idx_merged']
                if tmp: # if not empty, concatenate lists
                    node.info['idx_merged'] += tmp 

            #self.node_dict[new_idx].info['idx_merged']






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
