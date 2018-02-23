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

    def purify(self, X):
        """  Determines outliers and boundary points from X (low-dim points)

        self.idx_pure : idx of the pure points for the original array
        self.boundary : idx of the boundary points for the original array

        Returns idx for original data along with labels

        idx_in, cluster_label, idx_out

        """

        # Notation for idx stuff ... names seperated by underscore "_" specifies the subset

        self.n_sample = len(X)

        # Compute "outliers" based on density
        self.density_clf.fit_density(X)
        rho = self.density_clf.rho
        asort = np.argsort(rho)
        
        self.n_out = int(self.outlier_ratio*self.n_sample)
        self.X_in = X[asort[self.n_out:]] # well defined clusters
        self.X_out = X[asort[:self.n_out]] # will be classified at the very end

        self.idx_inliers = asort[self.n_out:]
        self.idx_outiers = asort[:self.n_out]

        self.density_clf.reset()

        # Refit density model on remaining data points:
        self.density_clf.fit(self.X_in)
        self.cluster_label = self.density_clf.cluster_label
        self.nn_list = self.density_clf.nn_list

        # Mark boundary point
        self.identify_boundary()
        self.idx_boundary = self.idx_inliers[self.idx_in_boundary]
        self.idx_pure = self.idx_inliers[self.idx_in_pure]
        
        # Remaining data set -- maybe we can avoid doing those copies here, but for now memory is not an issue
        self.X_pure = self.X_in[self.idx_in_pure] # remaining data points
        self.cluster_label_pure = self.cluster_label[self.idx_in_pure] # labels

        # Finally ... remove clusters that are too small 
        nh_size = self.density_clf.nh_size

        # Here it comes ...
        count = Counter(self.cluster_label_pure)
        idx_tmp = []
        n_remove = 0
        for k, v in count.items():
            if v > nh_size:
                idx_tmp.append(np.where(self.cluster_label_pure == k)[0])
            else:
                n_remove+=1
        
        # ... large enough clusters
        print("[vac.py]   Removing %i clusters since they are too small (< nh_size) ..."%n_remove)
        self.idx_in_pure_large = np.hstack(idx_tmp)

        # ... complementary set of indices (for later post-classification)
        self.idx_final = self.idx_pure[self.idx_in_pure_large]
        self.idx_final_out = np.setdiff1d(np.arange(self.n_sample), self.idx_final)
        self.cluster_label_final = self.cluster_label_pure[self.idx_in_pure_large]

        idx_out = self.idx_final_out
        idx_in =  self.idx_final
        cluster_label = self.cluster_label_final

        return idx_in, cluster_label, idx_out
    def get_purify_result(self):
        idx_out = self.idx_final_out
        idx_in =  self.idx_final
        cluster_label = self.cluster_label_final

        return self.idx_final, self.cluster_label_final, self.idx_final_out
        
    def fit_raw_graph(self, X_original, y_pred, n_average = 10, edge_min=0.9, clf_args = None):
        self.VGraph = VGraph(clf_type='rf', n_average = n_average, edge_min=edge_min, clf_args=clf_args)
        self.VGraph.fit(X_original, y_pred)
        self.save() # saving at this point
        
    def fit_robust_graph(self, X_original, cv_robust = 0.99):
        self.load()
        self.VGraph.merge_until_robust(X_original_final, cv_robust)
        self.save(name="tmp.pkl")

    def identify_boundary(self):
        """ Iterates over all cluster and marks "boundary" points """

        y_mask = np.copy(self.cluster_label)
        y_unique = np.unique(self.cluster_label)
        idx_all = np.arange(len(self.cluster_label))

        for yu in y_unique:
            self.mask_boundary_cluster(y_mask, yu)
    
        self.idx_in_boundary = idx_all[(y_mask == -1)]

        self.idx_in_pure = idx_all[(y_mask != -1)]

    def mask_boundary_cluster(self, y_mask, cluster_number):
        # check for points have that multiple points in their neighborhood
        # that do not have the same label as them

        ratio = self.nn_pure_ratio
        n_sample = len(y_mask)
        pos = (y_mask == cluster_number)
        nn = self.nn_list[pos][:,1:] # nn of cluster members
        idx_sub = np.arange(n_sample)[pos]

        r1 = [] # purity ratios
        for n in nn:
            l1 = self.cluster_label[n]
            r1.append(np.count_nonzero(l1 == cluster_number)/len(l1))
        
        r1 = np.array(r1)

        idx_unpure = idx_sub[(r1 < self.nn_pure_ratio)]

        y_mask[idx_unpure] = -1 #masking boundary terms

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
    
    def edge_info(self):
        self.VGraph.print_edge_score()
    