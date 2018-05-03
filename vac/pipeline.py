import numpy as np
from .vgraph import VGraph
from .tupledict import TupleDict
import pickle
from collections import Counter
from .main import VAC
from .tree import TREE
from .utility import make_file_name
from tsne_visual import TSNE
from fdc import FDC, plotting
import numpy as np
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

def quick_name(root, object_name, para, ext=".pkl):
    return root + object_name+"_"+para+".pkl"

class CLUSTER():
    """Validated agglomerative clustering [NEED A BETTER NAME]
        
    Parameters
    -------------
    [update coming soon]
    
    """
    # {'class_weight':'balanced','n_estimators': 50, 'max_features': min([X_down_sample.shape[1],200])}

    def __init__(self,
        outlier_ratio=0.2,
        nn_pure_ratio=0.99, 
        min_size_cluster=0,
        perplexity = 50,
        n_iteration_tsne =  1000,
        n_cluster_init = 30,
        seed = 0,
        nh_size = 40,
        eta = 1.5,
        test_ratio_size = 0.8,
        run_tSNE = True, # if not True, put in a file name for reading
        plot_inter = True,
        root = "",
        try_load = True
    ):
        self.param = {}
        
        # t-SNE parameters
        self.param['perplexity'] = perplexity
        self.param['n_iteration_tsne'] = n_iteration_tsne

        # Purification parameters 
        self.param['outlier_ratio'] = outlier_ratio
        self.param['nn_pure_ratio'] = nn_pure_ratio
        self.param['min_size_cluster'] = min_size_cluster
        
        # 
        self.param['n_cluster_init'] = n_cluster_init
        self.param['seed'] = seed
        self.param['root'] = root

        # Density clustering parameters
        self.param['nh_size'] = nh_size
        self.param['eta'] = eta
        self.param['test_ratio_size'] = test_ratio_size

        self.run_tSNE = run_tSNE 
        self.plot_inter = plot_inter 
        self.try_load = try_load

    def fit(self, data, clf_args = None):
        """ Clustering and fitting random forest classifier ...
        Processing steps:

        1. zscore data
        2. t-SNE data
        3. zscore t-SNE data
        """

        if clf_args is None:
            clf_args = {'class_weight':'balanced','n_estimators': 50, 'max_features': min([data.shape[1],200])}

        param = self.param
        np.random.seed(param['seed'])
        root = param['root']
        run_tSNE = self.run_tSNE
        plot_inter = self.plot_inter

        # zscoring data
        ss = StandardScaler()
        X_zscore = ss.fit_transform(data)
        info_str = make_file_name(param)

        ######################### dimensional reduction via t-SNE ###########################

        if run_tSNE is True:
            model_tsne = TSNE(perplexity=param['perplexity'], n_iter=param['n_iter_tsne'])
            X_tsne =  StandardScaler().fit_transform(model_tsne.fit_transform(X_zscore))
            tsnefile = param['root'] + 'tsne_'+info_str+'.pkl'
            print('t-SNE data saved in %s' % tsnefile)
            pickle.dump(X_tsne, open(tsnefile,'wb'))
        elif run_tSNE == 'auto':
            tsnefile = param['root'] + 'tsne_'+info_str+'.pkl'
            X_tsne = pickle.load(open(tsnefile,'rb'))
        else:
            X_tsne = pickle.load(open(self.run_tSNE,'rb'))

        ######################### Density clustering ###########################

        model_fdc = FDC(
            nh_size=param['nh_size'],
            eta=param['eta'], 
            test_ratio_size=param['test_ratio_size'],
            n_cluster_init=param['n_cluster_init']
            )

        model_vac = VAC(
            density_clf = model_fdc,
            outlier_ratio=param['outlier_ratio'],
            nn_pure_ratio=param['nn_pure_ratio'],
            min_size_cluster=param['min_size_cluster']
        )

        idx_pure_big, idx_pure_small, idx_out, idx_boundary = model_vac.get_pure_idx(X_tsne)

        if plot_inter is True:
            print('Plotting inliers and outliers')
            ytmp = -1*np.ones(len(X_tsne),dtype=int)
            ytmp[vac.idx_sets[('all','inliers')]] = vac.density_clf.cluster_label
            plotting.cluster_w_label(X_tsne, model_vac.label_sets[('pure','big')])
            print('Plotting pure data and boundaries')
            plotting.cluster_w_label(X_tsne[idx_pure_big], model_vac.label_sets[('pure','big')])

        idx_pure_big, idx_pure_small, idx_out, idx_boundary = model_vac.get_purify_result()

        # OK this is a mess, but at least there are no bug, need to clean this up ... 

        idx_train_pure = model_vac.idx_sets[('all','pure')][model_vac.idx_sets[('pure','big')]]
        idx_train_bound = model_vac.idx_sets[('all','boundary')]
        idx_train = np.sort(np.hstack([idx_train_pure, idx_train_bound]))
        label_pure_big = model_vac.label_sets[('pure','big')]
        y_pred = -1*np.ones(len(idx_train), dtype=int)

        for i, e in enumerate(idx_train_pure):
            y_pred[np.where(idx_train == e)[0]] = label_pure_big[i]

        x_train = X_zscore[idx_train]

        print("---> Initial estimates")

        ######## Raw graph #############
        loading_worked = False
        if self.try_load is True:
            loading_worked = model_vac.load(quick_name(root,'raw',info_str))
        
        if loading_worked is False:
            model_vac.fit_raw_graph(x_train, y_pred, n_average = 30, clf_args = clf_args, n_edge = 4)
            model_vac.save(quick_name(root,'raw',info_str))
        
        ######## Coarse grained graph #############
        loading_worked = False
        if self.try_load is True:
            loading_worked = model_vac.load(quick_name(root,'robust', info_str))
        
        if loading_worked is False:
            model_vac.fit_robust_graph(x_train, cv_robust = 0.99)
            model_vac.save(quick_name(root, 'robust', info_str))

        ######## Tree random forest classifer graph #############
        print(' == >> Fitting tree << == ')

        mytree = TREE(model_vac.VGraph.history, clf_args)
        mytree.fit(x_train)
        pickle.dump(mytree, open(quick_name(root, 'tree', info_str),'wb'))

        # classifying tree, can predict on new data that is normalized beforehand
        # When running on new data, use mytree.predict(ss.transform(X)) to get labels !

        return mytree, ss 
