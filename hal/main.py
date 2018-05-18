from fdc import FDC
import numpy as np
from .vgraph import VGraph
from .tupledict import TupleDict
import pickle
from collections import Counter
import os
from .purify import DENSITY_PROFILER

class VAC:
    """Validated agglomerative clustering"""

    def __init__(self, density_clf = None, outlier_ratio=0.2, nn_pure_ratio=0.9, min_size_cluster=20):
        """pass in a density classifier, need to be able to get labels and compute a density map
        """
        if density_clf is None:
            self.density_clf = FDC()
        else:
            self.density_clf = density_clf
        self.outlier_ratio = outlier_ratio
        self.nn_pure_ratio = nn_pure_ratio
        self.min_size_cluster = min_size_cluster
        self.ypred = None

        print(self.__dict__)
        # dict of cluster labels -> idx, cluster with largest overlap, ratio of overlap

    def purify(self, X):
        """
        Tries to purify clusters by removing outliers, boundary terms and small clusters
        """

        self.dp_profile= DENSITY_PROFILER(
            self.density_clf,
            outlier_ratio=self.outlier_ratio, 
            nn_pure_ratio=self.nn_pure_ratio, 
            min_size_cluster=self.min_size_cluster
        )

        self.dp_profile.fit(X)
        self.ypred = self.dp_profile.y

        #plotting.cluster_w_label(Xtsne, y)
        #ypred = model_DP.y
        #model_DP.check_purity(y) # this can be used to test t-SNE and initial set-up ...
        #model_DP.describe()
        #plotting.cluster_w_label(Xtsne, ypred)
    
    def fit_kNN_graph(self, X, y, n_average = 10, n_edge = 2, clf_type='svm', clf_args = None):

        self.kNN = kNN_graph()
        




    def fit_raw_graph(self, X_inlier, y_inlier_pred, n_average = 10, n_edge = 2, clf_args = None):
        self.VGraph = VGraph(clf_type=self.clf_type, n_average = n_average, clf_args=clf_args, n_edge = n_edge, test_size_ratio=0.8)
        self.VGraph.fit(X_inlier, y_inlier_pred)
        
    def fit_robust_graph(self, X_inlier, cv_robust = 0.99):
        """ Takes worst edges found using raw_graph method and performs retraining on those edges 
        with more expressive classifiers ( = more costly computationally)
        """
        self.VGraph.merge_until_robust(X_inlier, cv_robust, self.boundary_ratio)

    def identify_boundary(self, nn_list):
        """ Iterates over all cluster and marks "boundary" points """

        idx_all = np.arange(len(self.label_sets[('all','inliers')]))
        y_mask = self.mask_boundary_cluster(nn_list)

        self.idx_sets[('inliers','boundary')] = idx_all[(y_mask == -1)]
        self.idx_sets[('inliers','pure')] = idx_all[(y_mask != -1)]
    
    def mask_boundary_cluster(self, nn_list):
        # check for points have that multiple points in their neighborhood
        # that do not have the same label as them

        # construct list of arrays (mix types) the following form :
        # {cluster_main : [idx_cluster_secondary, idx_wrt_in]}
        # 
        y_mask = np.copy(self.label_sets[('all','inliers')])

        n_sample = len(y_mask)
        y_unique = np.unique(y_mask)

        for cluster_number in y_unique:
            pos = (y_mask == cluster_number) # checking the neighborhood of all members of these clusters
            # here maybe we want a larger neighborhood size
            nn_cluster = nn_list[pos][:,1:] # nn of cluster members, dim = (n_cluster_member, nh_size-1)
            idx_sub = np.arange(n_sample)[pos] # idx for pure + boundary (removed outliers)
            n_neighbor = len(nn_cluster[0])
    
        # we want to access boundary ratios in the following ez way
        # dict of cluster labels to boundary points
        # boundary points have a list of ratios [dict again cuz you need to know with respect to which cluster that is]

            idx_unpure = []
            boundary_ratios = []

            for i, nn_idx in enumerate(nn_cluster): 
                # For each point in the cluster, compute purity ratio (based on it's neighbors) 
                neighbor_labels = self.label_sets[('all','inliers')][nn_idx]
                count_nl = Counter(neighbor_labels)
                k, v = list(count_nl.keys()), list(count_nl.values())
                sort_count = np.argsort(v)
                kmax = k[sort_count[-1]] # most common cluster

                count_nl = {k: v / n_neighbor for k, v in count_nl.items()} # keep track of those only for boundary terms
            
                # Notes to me: when merging two clusters -> recover overlapping boundary
                # remaining boundary should be added to the new cluster. i.e. every cluster has a boundary
                ratio = count_nl[cluster_number]
                if ratio < self.nn_pure_ratio:      # is a boundary term
                    idx_unpure.append(idx_sub[i]) # w.r.t. to inliers fdc labels
                    # implies there is an overlap
                    boundary_ratios.append([k[sort_count[-1]], k[sort_count[-2]], idx_sub[i]])  # the index here is w.r.t. only to pure+boundary set
                    
            self.boundary_ratio[cluster_number] = np.array(boundary_ratios, dtype=int)
            # For every cluster there are multiple boundary points.
            # This is stored as an array : dict(cluster_number) -> np.array([2nd cluster to merge with, idx])
            # idx is boundary + pure (no outliers here)

            #-> for every cluster. There neighboring cluster with largest 
            # when merging clusters later on, just look at self.boundary_ratio[cluster_number]
            # then loop over the points of the cluster. If points have a kmax == other cluster, then remove those, and use idx
            # for forming new cluster. Reindex self.boundary_ratio with new cluster label. 
            
            if len(idx_unpure) > 0:
                y_mask[np.array(idx_unpure)] = -1 #masking boundary terms
        
        return y_mask

    def save(self, name=None):
        """ Saves current model to specified path 'name' """
        if name is None:
            name = self.make_file_name()
        fopen = open(name, 'wb')
        pickle.dump(self, fopen)
        fopen.close()
        
    def load(self, name=None):
        if name is None:
            name = self.make_file_name()
        if os.path.isfile(name):
            self.__dict__.update(pickle.load(open(name,'rb')).__dict__)
            return True
        return False

    def printvac(self, s):
        print('[vac.py]    %s'%s)

    def make_file_name(self):
        t_name = "clf_vgraph.pkl"
        return t_name
    
    def edge_info(self,option=0):
        self.VGraph.print_edge_score(option = option)
    