from fdc import FDC
import numpy as np
from .vgraph import VGraph
from .tupledict import TupleDict
import pickle
from collections import Counter

class VAC:

    def __init__(self, density_clf = None, outlier_ratio=0.2, nn_pure_ratio=0.9):
        """
        pass in a density classifier, need to be able to get labels and compute a density map
        """
        
        if density_clf is None:
            self.density_clf = FDC()
        else:
            self.density_clf = density_clf
        self.outlier_ratio = outlier_ratio
        self.nn_pure_ratio = nn_pure_ratio
        self.cluster_label = None
        self.idx_centers = None
        self.boundary_ratio = {} # dict of idx (w.r.t to inliers) to dict of ratios.
        # dict of cluster labels -> idx, cluster with largest overlap, ratio of overlap

    def get_pure_idx(self, X, eta=0.1):

        """  Determines outliers and boundary points from X (low-dim points)

        self.idx_pure : idx of the pure points for the original array
        self.boundary : idx of the boundary points for the original array

        Returns
        -----------

        idx_in, idx_boundary, idx_out

        """
        n_sample = len(X)

        # Compute "outliers" based on density
        self.density_clf.fit_density(X)
        rho = self.density_clf.rho
        asort = np.argsort(rho)
        
        n_out = int(self.outlier_ratio*n_sample)
        idx_inliers = np.sort(asort[n_out:])
        idx_low_rho = np.sort(asort[:n_out])

        self.density_clf.reset()
        self.density_clf.eta = 0.0

        # Refit density model on remaining data points:
        self.density_clf.fit(X[idx_inliers])
        self.density_clf.coarse_grain(np.linspace(0.,eta,10))
        self.cluster_label = self.density_clf.cluster_label # this is important // labels for later ...
        
        # Mark boundary point 
        idx_in_boundary, idx_in_pure = self.identify_boundary(self.density_clf.nn_list) # computed idx of inliers that are 'pure'
        self.idx_boundary = idx_inliers[idx_in_boundary] # w.r.t to original data points
        idx_pure = idx_inliers[idx_in_pure] # w.r.t to original data points

        # Finally ... remove clusters that are too small [those should go in the outlier category, i.e. they will be post-classified]        
        cluster_label_pure = self.cluster_label[idx_in_pure] # (>_<) labels
        nh_size = self.density_clf.nh_size

        # Here it comes ... 
        count = Counter(cluster_label_pure)
        idx_big = []
        idx_small = []

        for k, v in count.items():
            if v >= nh_size:
                idx_big.append(np.where(cluster_label_pure == k)[0])
            else:
                idx_small.append(np.where(cluster_label_pure == k)[0])
        
        # Large enough clusters
        print("[vac.py]    Removing %i clusters since they are too small (< nh_size) ..."% len(idx_small))
        assert len(idx_big) > 0, 'Assert false, no cluster is large enough, consider changing purity and outlier ratios !'
        
        # ----------------> 
        idx_in_pure_large = np.hstack(idx_big)

        self.cluster_pure_label = cluster_label_pure[idx_in_pure_large] # labels of points that will be used for classification 
        self.cluster_boundary_label = self.cluster_label[idx_in_boundary]

        self.idx_in = idx_pure[idx_in_pure_large] # (1st set --> <--) 
    
        if len(idx_small) > 0:
            idx_in_pure_small = np.hstack(idx_small)
            self.idx_out = np.hstack((idx_low_rho, idx_pure[idx_in_pure_small]))
        else:
            self.idx_out = idx_low_rho


        # --------- boundaries would need to be remerged -------> 

        print("[vac.py]    There are %i clusters remaining"% len(np.unique(self.cluster_pure_label)))
        return self.idx_in, self.idx_boundary, self.idx_out
        
    def get_purify_result(self):
        idx_out = self.idx_final_out
        idx_in =  self.idx_final
        cluster_label = self.cluster_label_final

        return self.idx_final, self.cluster_label_final, self.idx_final_out
        
    def fit_raw_graph(self, X_original, y_pred, n_average = 10, edge_min=0.9, clf_args = None):
        self.VGraph = VGraph(clf_type='rf', n_average = n_average, edge_min=edge_min, clf_args=clf_args)
        self.VGraph.fit(X_original, y_pred)
        self.cluster_label = y_pred
        self.save() # saving at this point
        
    def fit_robust_graph(self, X_original, cv_robust = 0.99):
        self.load()
        self.VGraph.merge_until_robust(X_original, cv_robust)
        self.save(name="robust.pkl")

    def identify_boundary(self, nn_list):
        """ Iterates over all cluster and marks "boundary" points """

        y_mask = np.copy(self.cluster_label)
        y_unique = np.unique(self.cluster_label)
        idx_all = np.arange(len(self.cluster_label))

        for yu in y_unique:
            self.mask_boundary_cluster(y_mask, yu, nn_list)
    
        idx_in_boundary = idx_all[(y_mask == -1)]
        idx_in_pure = idx_all[(y_mask != -1)]
        return idx_in_boundary, idx_in_pure

    def mask_boundary_cluster(self, y_mask, cluster_number, nn_list):
        # check for points have that multiple points in their neighborhood
        # that do not have the same label as them

        n_sample = len(y_mask)
        pos = (y_mask == cluster_number)
        nn = nn_list[pos][:,1:] # nn of cluster members
        idx_sub = np.arange(n_sample)[pos]
        n_neighbor = len(nn[0])
    
        # we want to access boundary ratios in the following ez way
        # dict of cluster labels to boundary points
        # boundary points have a list of ratios [dict again cuz you need to know with respect to which cluster that is]
    
        r1 = [] # purity ratios
        idx_unpure = []
        
        boundary_ratios = []
        for i, n in enumerate(nn):
            l1 = self.cluster_label[n]
            count_l1 = Counter(l1)
            #print(count_l1)
            kmax = max(count_l1.items(), key=lambda k: count_l1[k[0]])[0]
            #print(kmax)l
            count_l1 = {k: v / n_neighbor for k, v in count_l1.items()} # keep track of those only for boundary terms
        
            # Notes to me: when merging two clusters -> recover overlapping boundary
            # remaining boundary should be added to the new cluster. i.e. every cluster has a boundary
            ratio = count_l1[cluster_number]
            if ratio < self.nn_pure_ratio:      # is a boundary term
                idx_unpure.append(idx_sub[i])
                boundary_ratios.append([idx_sub[i], kmax, count_l1[kmax]])
                # ------ --------- ------------ ----------------------- -------------------- 
        
        self.boundary_ratio[cluster_number] = boundary_ratios
        y_mask[np.array(idx_unpure)] = -1 #masking boundary terms

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
        self.__dict__.update(pickle.load(open(name,'rb')).__dict__)
        return self

    def make_file_name(self):
        t_name = "clf_vgraph.pkl"
        return t_name
    
    def edge_info(self,option=0):
        self.VGraph.print_edge_score(option = option)
    