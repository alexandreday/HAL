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
        self.min_size_cluster= min_size_cluster

    def fit(self, X):
        y, idx_lowD = self.mark_density_profile(X)
        
    def mark_lowdensity_points(self, X):
        """
        X : array, shape = (-1, 2) 
            low-dimensional array
        """
        assert X.shape[1] < 5, "Working only for low-dimensional arrays"

        n_sample = len(X)
        rho = self.density_model.fit(X).rho # finds density based clusters
        y = np.copy(self.density_model.cluster_label)
        rho_argsort = np.argsort(rho)
        idx_lowD = rho_argsort[:int(outlier_ratio*n_sample)]
        
        # Density outliers 
        y[idx_lowD] = -1
        
        return y, idx_lowD

    def mark_murky_points(self, X, y):

        self.density_model.nn_list
        np.count_non
        # Run of all points, che




