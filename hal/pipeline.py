from .graph import kNN_Graph
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
    Construct FFT t-SNE embedding and identif pure clusters. Constructs
    a hierarchical classifer based on coarse-graining a k-NN graph
        
    Parameters
    -------------
    late_exag: int (default=800)
        Iteration at which to turn on late exageration
    
    alpha_late: float (default=2.0)
        Exageration factor used

    preprocess_option: dict (default = {'whiten':False, 'zscore':True})
        Preprocessing applied to the data. The resulting preprocessed data
        is used for training by the classifier. Note that the t-SNE will use the raw input
        data without further transformations.
    
    clf_type: str (default = 'svm')
        Type of classifier used for determining out-of-sample accuracy. Options
        are ['svm','rf','nb'], for support vector machine, random forest and naive bayes

    clf_args: dict (default = case dependent):
        Classifiers parameters. For svm, linear kernels are used by default
        For rf the maximum number of features used in each split is 200.
        All populations are weigh balanced in the cost function.
    
    clf_test_size_ratio: float (default = 0.8):
        Split ratio (training is 0.2 and testing is 0.8 by defeault)used for each fold for used in the bagging estimates. 

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

    tsne_type: str (default='fft')
        Type of t-SNE used (for now only fft is available)
    
    n_cluster_init: int (default=30)
        Number of initial cluster to start with (approx.)

    seed : int (default =0)
        Random seed
    
    nh_size : int (default='auto')
        Neighborhood size for density clustering

    warm_start: bool (default = False)
        Wether to reuse previously computed data from the pipeline. The default storage location is info_hal directory.

    """

    def __init__(self, # parameters are roughly in order of importance or usefuless
        late_exag = 1000, # default is no late  exageration
        alpha_late = 2.0,
        preprocess_option=None,
        n_cluster_init = 20,
        n_clf_sample_max = 500,
        clf_type = 'svm',
        clf_args = None,
        warm_start = False,
        root = "info_hal", # default directory where information will be dumped
        clf_test_size_ratio = 0.8,

        perplexity = 50,    
        n_iteration_tsne =  1000,
        outlier_ratio=0.05,
        nn_pure_ratio=0.0,
        gap_min=0.01,   
        min_size_cluster=50,
        tsne_type = 'fft', # default is FFTW t-SNE
        bh_angle = 0.5,
        seed = 0,
        nh_size = "auto",
        file_name_prefix = None,
        eta = 1.5,
        fdc_test_ratio_size = 0.8,
        run_tSNE = True, # if not True, put in a file name for reading
        n_job = "auto", # All available processors will be used
        n_bootstrap = 30,
        n_edge_kNN = 4, #-> check potential bug with number too small (2)
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
        self.tsne_type = tsne_type
        self.bh_angle = bh_angle

        self.late_exag = late_exag
        self.alpha_late = alpha_late

        # Purification parameters 
        self.n_job = n_job
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
        self.gap_min = gap_min
        
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
        self.file_name_prefix = file_name_prefix

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

        # standardized data for training
        X_preprocess = self.preprocess(data, **self.preprocess_option)

        # run t-SNE
        # when running t-SNE, use raw data, it's to the user to provide his data in a correct format.
        X_tsne = self.run_tSNE(data)
   
        # purifies clusters
        self.density_cluster = FDC(
            nh_size=self.nh_size,
            eta=self.eta,
            test_ratio_size=self.fdc_test_ratio_size,
            n_cluster_init=self.n_cluster_init,
            n_job=self.n_job
        )

        # Density clustering & finding outliers
        self.purify(X_tsne)
        self.dp_profile.describe()
        self.ypred_init = np.copy(self.ypred) # important for later

        # Fitting kNN graph
        self.fit_kNN_graph(X_preprocess, self.ypred)
        
        # Coarse graining kNN graph
        self.coarse_grain_kNN_graph(X_preprocess, self.ypred) # coarse grain

        # Constructing predictive model
        self.construct_model(X_preprocess) # links all classifiers together in a hierarchical model

        return self


    def fit_kNN_graph(self, X, ypred):
        """ Fits a kNN graph to the density clusters. Initially 
        performs a full (K^2) soft sweep (n_bootstrap small) to identify edges with potentially bad 
        edges. Then performs a deeper sweep for every cluster worst k edges.
        """

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
            y_murky = self.dp_profile.y_murky,
            gap_min= self.gap_min
        )

        self.kNN_graph.fit(X, ypred, n_bootstrap_shallow=5)
        pickle.dump(self.kNN_graph, open(self.root+ self.file_name['kNN_precoarse'],'wb'))
        return self

    def coarse_grain_kNN_graph(self, X, ypred):
        """ Merges edges with worst gaps in a semi-greedy fashion (see source code for definition of what this means !)
        """
        if check_exist(self.file_name['kNN_coarse'], self.root) & self.warm_start:
            self.kNN_graph = pickle.load(open(self.root+self.file_name['kNN_coarse'],'rb'))
        else:
            self.kNN_graph.coarse_grain(X, ypred)
            pickle.dump(self.kNN_graph, open(self.root+ self.file_name['kNN_coarse'],'wb'))
        return self
        
    def construct_model(self, X):
        """ Constructing classifier tree model """
        if check_exist(self.file_name['hal'], self.root) & self.warm_start:
            #print(pickle.load(open(self.root+self.file_name['hal'],'rb')))
            [self.robust_scaler, self.kNN_graph] = pickle.load(open(self.root+self.file_name['hal'],'rb'))
            
        else:
            self.kNN_graph.build_tree(X, self.ypred_init)
            pickle.dump([self.robust_scaler, self.kNN_graph], open(self.root+self.file_name['hal'],'wb'))

    def load(self, s=None):
        if s is None:
            assert check_exist(self.file_name['hal'], self.root), "hal model not saved for specified parameters"
            [self.robust_scaler, self.kNN_graph] = pickle.load(open(self.root+self.file_name['hal'],'rb'))
        else:
            assert check_exist(self.file_name[s], self.root), "%s model not saved for specified parameters"%s
            return pickle.load(open(self.root+self.file_name[s],'rb'))

    def predict(self, X, cv=0.5, gap=None, preprocess_option="same", option="fast"):
        if preprocess_option is "same":
            print("Preprocessing with same methods as during training\t", self.preprocess_option)
            X_preprocess = self.preprocess(X, **self.preprocess_option, verbose=False)
        else: # other options could be implemented 
            print("Preprocessing with methods\t", preprocess_option)
            X_preprocess = self.preprocess(X, **preprocess_option, verbose=False)
        
        return self.kNN_graph.predict(X_preprocess, cv=cv, option=option, gap=gap) # predict on full set <- !

    def preprocess(self, X, whiten=False, zscore = True, verbose=True):
        if verbose:
            print("Preprocessing data, whiten = %s, zscore = %s"%(str(whiten), str(zscore)))
        X_tmp = X
        from sklearn.decomposition import PCA
        if whiten is True:
            X_tmp = PCA(whiten=True).fit_transform(X)
        if zscore is True:
            if hasattr(self, 'robust_scaler'):
                X_tmp = self.robust_scaler.transform(X_tmp)
            else:
                self.robust_scaler = RobustScaler()
                X_tmp = self.robust_scaler.fit_transform(X_tmp)
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

    def cluster_w_label(self, X_tsne, y, rho="auto", **kwargs):
        from .plotting import cluster_w_label_plotly
        cluster_w_label_plotly(X_tsne, y)
        
        """ from .plotting import cluster_w_label
        if rho =="auto":
            self.dp_profile = self.load('fdc')
            idx_center = find_position_idx_center(X_tsne, y, np.unique(y), self.dp_profile.density_model.rho)
            cluster_w_label(X_tsne, y, idx_center, **kwargs)
        else:
            cluster_w_label(X_tsne, y,  **kwargs) """
        
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
            self.density_cluster = FDC(atol=0.001,rtol=0.00001)

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
            if self.run_tSNE:
                Xtsne = FItSNE(
                    np.ascontiguousarray(X.astype(np.float)),
                    theta = self.bh_angle,
                    start_late_exag_iter=self.late_exag, late_exag_coeff=self.alpha_late,
                    max_iter = self.n_iteration_tsne,
                    perplexity= self.perplexity,
                    rand_seed=self.seed
                )
            else:
                Xtsne = X #simple trick for now, a bit memory wasteful but not a big deal.
            pickle.dump(Xtsne, open(self.root+ tsnefile,'wb')) # Saving data in with useful name tag
            print('[HAL]    t-SNE data saved in %s' % tsnefile)
            return Xtsne
    
def check_exist(file_name, root = ""):
    return os.path.exists(root+file_name)
