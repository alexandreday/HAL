from fdc import FDC

class VAC:

    def __init__(self, X, density_clf = None, outlier_ratio=0.2, nn_pure_ratio=0.9):
        
        self.X = X
        if density_clf is None:
            self.density_clf = FDC()
        else:
            self.density_clf = density_clf
        self.outlier_ratio = outlier_ratio
        self.nn_pure_ratio = nn_pure_ratio
        self.n_sample = len(X)

    def fit(self):
        
        # compute "outliers" based on density
        self.density_clf.fit_density(self.X)
        rho = self.density_clf.rho
        asort = np.argsort(rho)


        self.n_out = int(self.outlier_ratio*self.n_sample)
        self.X_in = self.X[asort[self.n_out:]] # well defined clusters
        self.X_out = self.X[asort[:self.n_out]] # will be classified at the very end

        




        
    

