from .graph import l
from .tupledict import TupleDict
from .tree import TREE
from .utility import make_file_name, print_param, find_position_idx_center, make_hash_name
from .purify import DENSITY_PROFILER
from .plotjs import runjs

from fdc import FDC

from fitsne import FItSNE
from sklearn.preprocessing import RobustScaler

from collections import Counter
import numpy as np
import pickle, os

class HAL():
    """HAL-x : clustering via Hierarchial Agglomerative Learning.
    Construct FFT t-SNE embedding and identify pure clusters. Constructs
    a hierarchical classifer based on coarse-graining a k-NN graph
        
    Parameters
    -------------
    
    outlier_ratio: float (default=0.2)
        Ratio of the lowest density points that are considered outliers
    
    nn_pure_ratio: float (default=0.99)
        Ratio of the neighborhood that has to be in the same cluster to be considered pure
    
    min_size_cluster: int (default=25)
        Minimum size of a cluster
    
    perplexity: float (default=30)
        t-SNE perplexity - sets the effective neighborhood size
    
    n_iteration_tsne: int (default = 1000)
        Number of t-SNE iteration

    late_exag: int (default=800)
        Iteration at which to turn on late exageration
    
    alpha_late: float (default=2.0)
        Exageration factor used

    tsne_type: str (default='fft')
        Type of t-SNE used (for now only fft is available)
    
    n_cluster_init: int (default=30)
        Number of initial cluster to start with (approx.)

    seed : int (default =0)
        Random seed
    
    nh_size : int (default='auto')
        Neighborhood size for density clustering

    ...
    """
    # {'class_weight':'balanced','n_estimators': 50, 'max_features': min([X_down_sample.shape[1],200])}

    def __init__(self,
        outlier_ratio=0.05,
        nn_pure_ratio=0.0,
        min_size_cluster=25,
        preprocess_option=None,
        perplexity = 40,    
        n_iteration_tsne =  1000,
        late_exag = 1000, # default is no late  exageration
        tsne_type = 'fft', # default is FFTW t-SNE
        alpha_late = 2.0,
        n_cluster_init = 30,
        seed = 0,
        warm_start = False,
        nh_size = "auto",
        eta = 2.0,
        fdc_test_ratio_size = 0.8,
        run_tSNE = True, # if not True, put in a file name for reading
        root = "info_hal", # default directory where information will be dumped
        n_jobs = 0, # All available processors will be used
        n_clf_sample_max = 500,
        clf_type = 'svm',
        clf_args = None,
        n_bootstrap = 30,
        clf_test_size_ratio = 0.8,
        n_edge_kNN = 4,
        verbose = 1
    ):
        # Preprocessing:
        if preprocess_option is None:
            self.preprocess_option={"whiten":False,"zscore":True}
        else:
            self.preprocess_option = preprocess_option
            if "whiten" not in self.preprocess_option.keys():
                self.preprocess_option["whiten"]=False
            if "zscore" not in self.preprocess_option.keys():
                self.preprocess_option["zscore"]=True
            
        # t-SNE parameters
        self.perplexity = perplexity
        self.n_iteration_tsne = n_iteration_tsne
        self.n_jobs = n_jobs
        self.tsne_type = tsne_type

        self.late_exag = late_exag
        self.alpha_late = alpha_late

        # Purification parameters 
        self.outlier_ratio = outlier_ratio
        self.nn_pure_ratio = nn_pure_ratio
        self.min_size_cluster = min_size_cluster
        self.density_cluster = None
        
        # Density clustering parameters
        self.nh_size = nh_size
        self.eta = eta
        self.fdc_test_ratio_size = fdc_test_ratio_size
        self.n_cluster_init = n_cluster_init

        # kNN graph parameters (=> probably I don't want any of this <=)
        self.clf_test_size_ratio = clf_test_size_ratio
        self.n_bootstrap = n_bootstrap
        self.clf_type = clf_type
        self.n_clf_sample_max = n_clf_sample_max
        
        if clf_args is None:
            if clf_type == 'svm':
                self.clf_args = {'kernel':'linear','class_weight':'balanced'}
            else:
                self.clf_args = {'class_weight':'balanced'}
        else: 
            self.clf_args = clf_args
        self.n_edge_kNN = n_edge_kNN

        # Misc. ==> need to find a way to fix that for fitsne ...
        self.seed = seed

        if not os.path.exists(root):
            os.makedirs(root)

        if root[-1] != "/":
            root+="/"

        self.root = root
        self.tsne = run_tSNE
        self.warm_start = warm_start

        self.file_name = {}
        self.file_name['tsne'] = make_hash_name(self.__dict__, file='tsne')
        self.file_name['fdc'] = make_hash_name(self.__dict__, file='fdc')
        self.file_name['kNN_precoarse'] = make_hash_name(self.__dict__, file='kNN_precoarse')
        self.file_name['kNN_coarse'] = make_hash_name(self.__dict__, file='kNN_coarse')
        self.file_name['hal'] = make_hash_name(self.__dict__, file='hal')


    def fit(self, data):
        """ Clustering and fitting random forest classifier ...
        Processing steps:
            1. zscore data
            2. t-SNE data
            3. zscore t-SNE data
            4. Find pure clusters
            5. fit random forest, and coarse-grain, until desired level
            6. predict on data given cv score

        Returns:

        self
        """

        """ if clf_args is None:
            clf_args = {'class_weight':'balanced','n_estimators': 50, 'max_features': min([data.shape[1],200])} """

        np.random.seed(self.seed)

        self.n_feature = data.shape[1]

        X_preprocess = self.preprocess(data, **self.preprocess_option)

        # run t-SNE
        X_tsne = self.run_tSNE(X_preprocess)
   
        # purifies clusters
        self.density_cluster = FDC(
            nh_size=self.nh_size,
            eta=self.eta,
            test_ratio_size=self.fdc_test_ratio_size,
            n_cluster_init=self.n_cluster_init
        )

        self.purify(X_tsne)

        self.dp_profile.describe()

        self.ypred_init = np.copy(self.ypred) # important for later

        self.fit_kNN_graph(X_preprocess, self.ypred)

        self.coarse_grain_kNN_graph(X_preprocess, self.ypred) # coarse grain

        self.construct_model(X_preprocess) # links all classifiers together in a hierarchical model

        return self


    def fit_kNN_graph(self, X, ypred):
        # Left it here ... need to update this to run graph clustering
        if check_exist(self.file_name['kNN_precoarse'], self.root) & self.warm_start:
            self.kNN_graph = pickle.load(open(self.root+self.file_name['kNN_precoarse'],'rb'))
            return self

        self.kNN_graph = kNN_Graph(
            n_bootstrap = self.n_bootstrap,
            test_size_ratio = self.clf_test_size_ratio,
            n_sample_max=self.n_clf_sample_max,
            clf_type = self.clf_type,
            clf_args = self.clf_args,
            n_edge = self.n_edge_kNN,
            y_murky = self.dp_profile.y_murky
        )

        self.kNN_graph.fit(X, ypred, n_bootstrap_shallow=5)
        pickle.dump(self.kNN_graph, open(self.root+ self.file_name['kNN_precoarse'],'wb'))
        return self

    def coarse_grain_kNN_graph(self, X, ypred):

        if check_exist(self.file_name['kNN_coarse'], self.root) & self.warm_start:
            self.kNN_graph = pickle.load(open(self.root+self.file_name['kNN_coarse'],'rb'))
        else:
            self.kNN_graph.coarse_grain(X, ypred)
            pickle.dump(self.kNN_graph, open(self.root+ self.file_name['kNN_coarse'],'wb'))
        return self
        
    def construct_model(self, X):
        if check_exist(self.file_name['hal'], self.root) & self.warm_start:
            self.kNN_graph = pickle.load(open(self.root+self.file_name['hal'],'rb'))
        else:
            self.kNN_graph.build_tree(X, self.ypred_init)
            pickle.dump(self.kNN_graph, open(self.root+self.file_name['hal'],'wb'))

    def load(self, s=None):
        if s is None:
            self.kNN_graph = pickle.load(open(self.root+self.file_name['hal'],'rb'))
        else:
            return pickle.load(open(self.root+self.file_name[s],'rb'))

    def predict(self, X, cv=0.5, preprocess_option="same", option="fast"):
        if preprocess_option is "same":
            print("Preprocessing with same methods as during training\t", self.preprocess_option)
            X_preprocess = self.preprocess(X, **self.preprocess_option, verbose=False)
        else: # other options could be implemented 
            print("Preprocessing with methods\t", preprocess_option)
            X_preprocess = self.preprocess(X, **preprocess_option, verbose=False)
        
        return self.kNN_graph.predict(X_preprocess, cv=cv, option=option) # predict on full set !

    def preprocess(self, X, whiten=False, zscore = True, verbose=True):
        if verbose:
            print("Preprocessing data, whiten = %s, zscore = %s"%(str(whiten), str(zscore)))
        X_tmp = X
        from sklearn.decomposition import PCA
        if whiten is True:
            X_tmp = PCA(whiten=True).fit_transform(X)
        if zscore is True:
            X_tmp = RobustScaler().fit_transform(X_tmp)
        return X_tmp

    def possible_clusters(self, cv):
        return np.sort(self.kNN_graph.tree.possible_clusters(cv=cv))

    def feature_importance(self, cluster_idx):
        p_id = self.kNN_graph.tree.node_dict[cluster_idx].parent.get_id()
        return self.kNN_graph.tree.feature_importance_dict[p_id]
    
    def feature_median(self, cluster_idx):
        return self.kNN_graph.tree.node_dict['median_marker']['mu']

    def plot_tree(self, feature_name = None):
        """ Renders a dashboard with the hierarchical tree
        and bar charts of feature information for each cluster
        """
        Xtsne = pickle.load(open(self.root+self.file_name['tsne'],'rb'))
        
        if feature_name is None:
            feature_name_ = list(range(self.n_feature))
        else:
            assert len(feature_name) == self.n_feature, "Feature name list must have the same number of element as the number of features"
            feature_name_ = feature_name

        self.kNN_graph.tree.plot_tree(Xtsne, self.ypred_init, feature_name_)
        runjs('js/')

    def cluster_w_label(self, X_tsne, y, rho=None, **kwargs):
        from .plotting import cluster_w_label
        if rho =="auto":
            idx_center = find_position_idx_center(X_tsne, y, np.unique(y), self.density_cluster.rho)
            cluster_w_label(X_tsne, y, idx_center, **kwargs)
        else:
            cluster_w_label(X_tsne, y,  *kwargs)
        
    def purify(self, X):
        """
        Tries to purify clusters by removing outliers, boundary terms and small clusters
        """
        if check_exist(self.file_name['fdc'], self.root) & self.warm_start:
            self.dp_profile = pickle.load(open(self.root+self.file_name['fdc'],'rb'))
            self.density_cluster = self.dp_profile.density_model
            print_param(self.dp_profile.__dict__)
            self.ypred = self.dp_profile.y
            return

        if self.density_cluster is None:
            self.density_cluster = FDC()

        self.dp_profile = DENSITY_PROFILER(
            self.density_cluster,
            outlier_ratio=self.outlier_ratio, 
            nn_pure_ratio=self.nn_pure_ratio, 
            min_size_cluster=self.min_size_cluster
        )
        print_param(self.dp_profile.__dict__)

        self.dp_profile.fit(RobustScaler().fit_transform(X))
        
        self.ypred = self.dp_profile.y

        del self.dp_profile.density_model.nn_dist # No need to save this ..., can always be recomputed extremely fast
        del self.dp_profile.density_model.nn_list
        del self.dp_profile.density_model.density_model
        del self.dp_profile.density_model.density_graph

        pickle.dump(self.dp_profile, open(self.root+ self.file_name['fdc'],'wb'))

    def run_tSNE(self, X):
        """
        Performs t-SNE dimensional reduction using a FFT based C-implementation of Barnes-Hut t-SNE (very fast)
        Optionally, one can specify the number of paralell jobs to run from the :n_jobs: parameter in the constructor.
        Make sure X is normalized too !
        
        Options
        --------
        self.param['tsne']:
            - True : runs t-SNE
            - 'auto' : checks wether file exists, if not runs t-SNE

        Returns
        -------
        X_tsne: array, shape = (-1,2)
            t-SNE embedding

        """
        print('[pipeline.py]    Running t-SNE for X.shape = (%i,%i)'%X.shape)
    
        tsnefile = self.file_name['tsne']
       
        if check_exist(tsnefile, self.root) & self.warm_start:
            return pickle.load(open(self.root+tsnefile,'rb'))
        else:
            assert self.tsne_type == 'fft' # for now just use this one
            Xtsne = FItSNE(
                np.ascontiguousarray(X.astype(np.float)),
                start_late_exag_iter=self.late_exag, late_exag_coeff=self.alpha_late,
                max_iter = self.n_iteration_tsne,
                perplexity= self.perplexity,
                rand_seed=self.seed
            )
            pickle.dump(Xtsne, open(self.root+ tsnefile,'wb')) # Saving data in with useful name tag
            print('[HAL]    t-SNE data saved in %s' % tsnefile)
            return Xtsne
    
def check_exist(file_name, root = ""):
    return os.path.exists(root+file_name)
