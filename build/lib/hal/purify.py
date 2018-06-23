import numpy as np
from fdc import FDC

class DENSITY_PROFILER:
    """
    What this class does:
        Finds low density points (yi = -1)
        Finds overlapping boundaries (yi = -2)
        Finds small clusters (yi = -3)
        Finds high density clusters (yi >= 0)

    Just call DENSITY_PROFILER.fit(X)
    """

    def __init__(self, density_model:FDC, outlier_ratio = 0.1, nn_pure_ratio = 0.95, min_size_cluster = 0):
        self.density_model = density_model
        self.outlier_ratio = outlier_ratio
        self.nn_pure_ratio = nn_pure_ratio
        self.y_murky = None
        self.min_size_cluster= min_size_cluster # this should be on the order of the number of features
        self.y = None

    def fit(self, X):
        self.mark_lowdensity_points(X).mark_murky_points().mark_small_cluster()
        print("[purify.py]    # of clusters after purification : %i"%np.count_nonzero(np.unique(self.y) > -1))
        return self

    def get_cluster_label():
        return self.y

    def mark_lowdensity_points(self, X):
        """
        X : array, shape = (-1, 2) 
            low-dimensional array
        """
        assert X.shape[1] < 5, "Working only for low-dimensional arrays"

        n_sample = len(X)
        eta = self.density_model.eta
        
        self.density_model.eta = 0.
        rho = self.density_model.fit(X).rho # finds density based clusters
        self.density_model.coarse_grain(np.linspace(0, eta, 30)) # stops when max number of clusters is reached 

        self.y = np.copy(self.density_model.cluster_label)
        rho_argsort = np.argsort(rho)
        self.idx_lowD = rho_argsort[:int(self.outlier_ratio*n_sample)]
        
        # Density outliers 
        self.y[self.idx_lowD] = -1
        
        return self

    def mark_murky_points(self):

        self.density_model.nn_list
        n_sample = len(self.y)
        counts = np.empty(n_sample, dtype=float)
    
        for i in range(n_sample): # could cythonize this ... but no need for that now
            y_tmp =np.copy(self.y[self.density_model.nn_list[i]])
            y_tmp = y_tmp[(y_tmp != -1)]
            if len(y_tmp) == 0:
                counts[i] = 10. # not murky, just outlier
            else:
                counts[i] = np.count_nonzero(y_tmp == self.y[i])/len(y_tmp)
        
        self.idx_murky = np.where(counts < self.nn_pure_ratio)[0]
        self.y_murky = -1*np.ones(n_sample, dtype=int)
        self.y_murky[self.idx_murky] = np.copy(self.y[self.idx_murky]) # these are copied for later mergers 
        self.y[self.idx_murky] = -2 # assign murky
        self.y[self.idx_lowD] = -1 # reassign lowD

        return self
    
    def mark_small_cluster(self):
        y_unique = np.unique(self.y)
        y_unique = y_unique[y_unique > -1]
        counts = []
        for y_u in y_unique:
            counts.append(np.count_nonzero(self.y == y_u))
        counts = np.array(counts)
        
        y_small = y_unique[counts < self.min_size_cluster]
        self.idx_small = []
        
        if len(y_small) > 0:
            for y in y_small:
                self.idx_small.append(np.where(self.y == y)[0])
            self.idx_small = np.hstack(self.idx_small)
            self.y[self.idx_small] = -3
        
        return self

    ################# INFORMATIVE FUNCTIONS #######################
    
    def describe(self, cout = print):
        import pandas as pd

        des = [] 
        y_unique = np.unique(self.y)
        for yu in y_unique:
            c = np.count_nonzero(self.y == yu)
            des.append([yu, c, c/len(self.y)])
        
        cout(pd.DataFrame(des, columns=['label','count','ratio']).to_string(index=False))

    def check_purity(self, ytrue, plot=True): # Only if you have access to the true labels

        from collections import OrderedDict
        y_unique = np.unique(self.y)
        y_unique = y_unique[y_unique > -1]
        n_unique = len(y_unique)
        cluster_entropy = OrderedDict()
        for yu in y_unique: # want to check entropy of the distribution 
            ytrue_sub = ytrue[self.y == yu]
            y_unique_sub, counts = np.unique(ytrue_sub, return_counts=True)
            pc = counts/len(ytrue_sub)
            S = -1*np.sum(pc*np.log(pc)) # entropy
            cluster_entropy[yu] = S

        if plot is True:
            from matplotlib import pyplot as plt
            plt.bar(np.arange(n_unique), cluster_entropy.values(), width=0.4)
            plt.xticks(np.arange(n_unique), y_unique, rotation=45, fontsize=8)
            plt.ylabel('Entropy')
            plt.xlabel('Cluster label')
            cS_value = np.array(list(cluster_entropy.values()))
            plt.title('Total purity = %.3f, nCluster=%i'%(np.sum(cS_value),n_unique))
            plt.tight_layout()
            plt.show()
        
        return np.mean(cS_value), np.sum(cS_value), cluster_entropy

        
        
