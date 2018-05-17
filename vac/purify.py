import numpy as np
from fdc import FDC
"""
What this class does:
    Finds low density points (yi = -1)
    Finds overlapping boundaries (yi = -2)
    Finds small clusters (yi = -3)
    Finds high density clusters (yi >= 0)
"""

class DENSITY_PROFILER:
    def __init__(self, density_model:FDC, outlier_ratio = 0.2, nn_pure_ratio = 0.99, min_size_cluster = 0):
        self.density_model = density_model
        self.outlier_ratio = outlier_ratio
        self.nn_pure_ratio = nn_pure_ratio
        self.min_size_cluster= min_size_cluster # this should be on the order of the number of features
        self.y = None

    def fit(self, X):
        self.mark_density_profile(X).mark_murky_points().mark_small_cluster()

    def get_cluster_label():
        return self.y

    def mark_lowdensity_points(self, X):
        """
        X : array, shape = (-1, 2) 
            low-dimensional array
        """
        assert X.shape[1] < 5, "Working only for low-dimensional arrays"

        n_sample = len(X)
        rho = self.density_model.fit(X).rho # finds density based clusters
        self.y = np.copy(self.density_model.cluster_label)
        rho_argsort = np.argsort(rho)
        self.idx_lowD = rho_argsort[:int(outlier_ratio*n_sample)]
        
        # Density outliers 
        self.y[idx_lowD] = -1
        
        return self

    def mark_murky_points(self):

        self.density_model.nn_list
        counts = np.empty(len(y), dtype=float)
        n_sample = len(y)

        for i in range(n_sample): # could cythonize this ... but no need for that now
            counts[i] = np.count_nonzero(self.density_model.nn_list[i] == self.y[i])
        
        counts /= len(self.density_model.nh_size) # between [0, 1]
        
        self.idx_murky = np.where(counts < self.nn_pure_ratio)[0]
        self.y[self.idx_murky] = -2 # assign murky
        self.y[self.idx_lowD] = -1 # reassign lowD

        return self
    
    def mark_small_cluster(self):
        y_unique = np.unique(self.y)[2:] # removes -1 and -2 labels
        counts = []
        for y_u in y_unique:
            counts.append(np.count_nonzero(self.y = y_u))
        counts = np.array(counts)
        
        y_small = y_unique[counts < self.min_size_cluster]
        self.idx_small = []
        

        for y in y_small:
            self.idx_small[np.where(self.y == y)[0]]
        self.idx_small = np.hstack(self.idx_small)

        self.y[self.idx_small] = -3
        return self
    

    





