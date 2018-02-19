from fdc import FDC

class VAC:

    def __init__(self, density_clf = None, outlier_ratio=0.2, nn_pure_ratio=0.9):
        
        if density_clf is None:
            self.density_clf = FDC()
        else:
            self.density_clf = density_clf
        self.outlier_ratio = outlier_ratio
        self.nn_pure_ratio = nn_pure_ratio
        self.n_sample = len(X)

    def fit(self, X):
        self.n_sample = len(X)
        
        # compute "outliers" based on density
        self.density_clf.fit_density(X)
        rho = self.density_clf.rho
        asort = np.argsort(rho)


        self.n_out = int(self.outlier_ratio*self.n_sample)
        self.X_in = X[asort[self.n_out:]] # well defined clusters
        self.X_out = X[asort[:self.n_out]] # will be classified at the very end

        self.density_clf.reset()

        # refit density model on remaining data points:
        self.density_clf.fit(X_in)
        self.cluster_label = self.density_clf.cluster_label
        self.nn_list = self.density_clf.nn_list

        self.identify_boundary()


    def identify_boundary(self):

        y_mask = np.copy(self.cluster_label)
        y_unique = np.unique(self.cluster_label)
        idx_all = np.arange(len(self.cluster_label)

        for yu in y_unique:
            self.mask_boundary_cluster(y_mask, yu)
    
        self.boundary = idx_all[(y_mask == -1)]
        self.pure = idx_all[(y_mask != -1)]

    def mask_boundary_cluster(self, y_mask, cluster_number):
        # check for points have that multiple points in their neighborhood
        # that do not have the same label as them

        ratio = self.nn_pure_ratio
        pos = (y_mask == cluster_number)
        nn = self.nn_list[pos][:,1:]

        r1 = [] # purity ratios
        for n in nn:
            l1 = self.cluster_label[n]
            r1.append(np.count_nonzero(l1 == cluster_number)/len(l1))
        
        r1 = np.array(r1)

        y_mask[pos & (r1 < self.nn_pure_ratio)] = -1 #







        
    

