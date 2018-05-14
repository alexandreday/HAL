import numpy as np
from .vgraph import VGraph
from .tupledict import TupleDict
import pickle
from collections import Counter
from .main import VAC
from .tree import TREE
from .utility import make_file_name
from fdc import FDC, plotting
import numpy as np
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import os

def quick_name(root, object_name, para, ext=".pkl"):
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
        angle = 0.5,
        tsne_type = 'fit',
        n_cluster_init = 30,
        seed = 0,
        nh_size = 40,
        eta = 1.5,
        test_ratio_size = 0.8,
        run_tSNE = True, # if not True, put in a file name for reading
        plot_inter = True,
        root = "",
        try_load = True,
        n_jobs = 0, # All available processors will be used
        random_seed = None
    ):
        self.param = {}
        
        # t-SNE parameters
        self.param['perplexity'] = perplexity
        self.param['n_iteration_tsne'] = n_iteration_tsne
        self.param['angle'] = angle
        self.param['tsne_type'] = tsne_type

        # Purification parameters 
        self.param['outlier_ratio'] = outlier_ratio
        self.param['nn_pure_ratio'] = nn_pure_ratio
        self.param['min_size_cluster'] = min_size_cluster
        
        # 
        self.param['n_cluster_init'] = n_cluster_init
        self.param['seed'] = seed
        self.param['root'] = root
        self.param['n_jobs'] = n_jobs

        # Density clustering parameters
        self.param['nh_size'] = nh_size
        self.param['eta'] = eta
        self.param['test_ratio_size'] = test_ratio_size

        self.param['tsne'] = run_tSNE
        self.plot_inter = plot_inter 
        self.try_load = try_load

        self.file_name = {}
        info_str = make_file_name(self.param)
        self.file_name['raw'] = quick_name(root, 'raw', info_str)
        self.file_name['robust'] = quick_name(root, 'robust', info_str)
        self.file_name['tree'] = quick_name(root, 'tree', info_str)
        self.file_name['tsne'] = self.param['root'] + 'tsne_perp=%i_niter=%i.pkl'%(self.param['perplexity'], self.param['n_iteration_tsne'])
        #print(self.file_name)

    def fit(self, data, clf_args = None):
        """ Clustering and fitting random forest classifier ...
        Processing steps:
            1. zscore data
            2. t-SNE data
            3. zscore t-SNE data
            4. Find pure clusters
            5. fit random forest, and coarse-grain, until desired level
            6. predict on data given cv score

        Returns
        -------
        tree classifier, zscore scaler
        """

        if clf_args is None:
            clf_args = {'class_weight':'balanced','n_estimators': 50, 'max_features': min([data.shape[1],200])}

        param = self.param
        np.random.seed(param['seed'])
        root = param['root']
        run_tSNE = self.run_tSNE
        plot_inter = self.plot_inter

        # zscoring data
        self.ss = StandardScaler()

        X_zscore = self.ss.fit_transform(data)

        X_tsne = self.run_tSNE(X_zscore)

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

        # check this part now
        idx_pure_big, idx_pure_small, idx_out, idx_boundary = model_vac.get_pure_idx(X_tsne)

        if plot_inter is True:
            print('[pipeline.py]   Plotting inliers and outliers')
            ytmp = -1*np.ones(len(X_tsne),dtype=int)
            ytmp[model_vac.idx_sets[('all','inliers')]] = model_vac.density_clf.cluster_label
            plotting.cluster_w_label(X_tsne, ytmp)
            print('[pipeline.py]   Plotting pure data and boundaries')
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

        # ===================================================
        # ===================================================

        x_train = X_zscore[idx_train] # points to train on

        print("---> Initial estimates")

        ######## Raw graph #############
        loading_worked = False
        if self.try_load is True:
            loading_worked = model_vac.load(self.file_name['raw'])
        
        if loading_worked is False:
            model_vac.fit_raw_graph(x_train, y_pred, n_average = 30, clf_args = clf_args, n_edge = 4)
            model_vac.save(self.file_name['raw'])
        
        ######## Coarse grained graph #############
        loading_worked = False
        if self.try_load is True:
            loading_worked = model_vac.load(self.file_name['robust'])
        
        if loading_worked is False:
            model_vac.fit_robust_graph(x_train, cv_robust = 0.99)
            model_vac.save(self.file_name['robust'])

        ######## Tree random forest classifer graph #############
        print('[pipeline.py]    == >> Fitting tree << == ')
        self.tree = TREE(model_vac.VGraph.history, clf_args)
        self.tree.fit(x_train)
        pickle.dump([self.tree, self.ss], open(self.file_name['tree'],'wb'))

        # classifying tree, can predict on new data that is normalized beforehand
        # When running on new data, use mytree.predict(ss.transform(X)) to get labels !
        return self.tree, self.ss 

    def load_clf(self, fname=None):
        if fname is not None:
            self.tree, self.ss = pickle.load(open(tree_file_name,'rb'))
        else:
            self.tree, self.ss = pickle.load(open(self.file_name['tree'],'rb'))
    
    def predict(self, X, cv=0.9, option='fast'):
        """
        Standardizes according to training set rescaling and then predicts given the cv-score specified
        """
        return self.tree.predict(self.ss.transform(X), cv=cv, option=option)   

    def run_tSNE(self, X):
        """
        Performs t-SNE dimensional reduction using a FFT based C-implementation of Barnes-Hut t-SNE (very fast)
        Optionally, one can specificy number of paralell jobs to run from the :n_jobs: parameter in the constructor.
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

        if self.param['tsne_type'] == 'fit': # makes it easy to switch back to tsne_visual
            import fitsne
            kwargs_ = {
                'nthreads':self.param['n_jobs'],
                'perplexity':self.param['perplexity'], 
                'max_iter':self.param['n_iteration_tsne'], 
                'theta':self.param['angle'],
                'rand_seed': self.param['seed']
            }
            if (kwargs_['rand_seed'] is None) : kwargs_['rand_seed'] = -1 
        else:
            from tsne_visual import TSNE
            model_tsne = TSNE(
                perplexity=self.param['perplexity'],
                n_iter=self.param['n_iteration_tsne'],
                verbose=self.param['verbose'],
                angle=self.param['angle'],
                random_state=self.param['seed']
            )

        if self.param['tsne'] is True: # Run t-SNE embedding
            if self.param['tsne_type'] == 'fit':
                X_tsne = StandardScaler().fit_transform(fitsne.FItSNE(np.ascontiguousarray(X.astype(np.float)), **kwargs_))
            else:
                X_tsne =  StandardScaler().fit_transform(model_tsne.fit_transform(X))

            print('t-SNE data saved in %s' % tsnefile)
            pickle.dump(X_tsne, open(tsnefile,'wb')) # Saving data in with useful name tag

        elif self.param['tsne'] == 'auto': # if files exist, will read it.
            if os.path.isfile(tsnefile):
                X_tsne = pickle.load(open(tsnefile,'rb'))
            else:
                if self.param['tsne_type'] == 'fit':
                    X_tsne = StandardScaler().fit_transform(fitsne.FItSNE(np.ascontiguousarray(X.astype(np.float)), **kwargs_))
                else:
                    X_tsne =  StandardScaler().fit_transform(model_tsne.fit_transform(X))
                print('t-SNE data saved in %s' % tsnefile)
                pickle.dump(X_tsne, open(tsnefile,'wb'))
        else:
            assert False
        
        return X_tsne
