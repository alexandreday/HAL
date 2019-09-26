'''
Created on Jan 16, 2017

@author: Alexandre Day

Description : 
    Module for performing cross-validation and classification 
'''

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import ShuffleSplit
import numpy as np
from collections import Counter
import itertools
import time
from copy import deepcopy

def most_common(lst):
    return max(set(lst), key=lst.count)

class CLF:
    """ Implements a classifier for hierarchical clustering

    Parameters:
    ----------
    clf_type : str
        Type of cluster, either 'svm' or 'rf'
    clf_kwargs : optional keyword arguments for the classifier    
    """

    def __init__(self, clf_type='svm', n_bootstrap=10, train_size = 0.8, n_sample_max = 1000, clf_kwargs=None):
        self.clf_type = clf_type
        self.n_bootstrap = n_bootstrap
        self.n_sample_max = n_sample_max
        self.train_size = train_size

        if clf_kwargs is None:
            self.clf_kwargs = {}
        else:
            self.clf_kwargs = clf_kwargs

        if self.clf_type == 'svm':
                self.clf = SVC(**self.clf_kwargs)

        elif self.clf_type == 'rf':
                if 'max_features' not in self.clf_kwargs.keys():
                    self.clf_kwargs['max_features'] = min([int(X.shape[1]/2), 100])
                if 'n_estimators' not in self.clf_kwargs.keys():
                    self.clf_kwargs['n_estimators'] = 20
                self.clf = RandomForestClassifier(**self.clf_kwargs)
        elif self.clf_type == 'nb':
                self.clf = GaussianNB()#**self.clf_kwargs)
        else:
            assert False

        """ if clf_kwargs is None:
            if clf_type == 'svm':
                self.clf_kwargs = {'kernel':'linear','class_weight':'balanced'}
            else:
                self.clf_kwargs = {'class_weight':'balanced'}
        else:
            self.clf_kwargs = clf_kwargs """

        self.trained = False
        self.cv_score = 1.0
        self.y_unique = None
        self.class_count = None
        self.idx_pos = {}
        
    #@profile
    def fit(self, X, y):
        """ Fit clf to data.

        Parameters
        ------------
        X: array, shape = (n_sample, n_feature)
            your data

        y: array, shape = (n_sample, 1)
            your labels
        
        Other parameters
        ------------
        self.n_bootstrap : int
            number of classifiers to train (will then take majority vote)
        
        self.test_size: float
            ratio of test size (between 0 and 1). 

        Return
        -------
        self, CLF object

        """
        #print('Training %s with n_bootstrap=%i, with nsample=%i'%(self.clf_type, self.n_bootstrap, len(X)))

        self.trained = True

        predict_score = []; training_score = []; clf_list = []; xtrain_scaler_list = [];
        zero_eps = 1e-6

        self.y_unique = np.unique(y) # different labels
        assert len(self.y_unique) > 1, "Cluster provided only has a unique label, can't classify !"
        
        idx_bootstrap_split = self.idx_train_test_split(y, train_size=self.train_size, n_split=self.n_bootstrap, n_sample_max=self.n_sample_max)
        #print(len(idx_bootstrap_split[0][0]),'\t', len(idx_bootstrap_split[0][1]))
        for s in range(self.n_bootstrap):

            idx_train, idx_test = idx_bootstrap_split[s]
            xtrain, xtest, ytrain, ytest = X[idx_train], X[idx_test], y[idx_train], y[idx_test] 
            #= self.train_test_split(X, y)
            
            std = np.std(xtrain, axis = 0)    
            std[std < zero_eps] = 1.0 # get rid of zero variance data.
            mu, inv_sigma = np.mean(xtrain, axis=0), 1./std

            xtrain = (xtrain - mu)*inv_sigma # zscoring the data cd

            xtest = (xtest - mu)*inv_sigma
            
            # train classifier
            self.clf.fit(xtrain, ytrain)
            
            # predict on train set
            t_score = self.clf.score(xtrain, ytrain) 
            
            training_score.append(t_score)

            # predict on validation set # maybe this is a noisy estimate ?
            p_score = self.clf.score(xtest[:500], ytest[:500])

            predict_score.append(p_score)

            clf_list.append(deepcopy(self.clf))

            xtrain_scaler_list.append([mu, inv_sigma])
        
        #print(predict_score)
        self.scaler_list = xtrain_scaler_list # scaling transformations (zero mean, unit std)
        self.cv_score = np.mean(predict_score)
        self.cv_score_median = np.median(predict_score)
        self.cv_score_iqr = np.percentile(predict_score, 80) - self.cv_score_median
        self.cv_score_std = np.std(predict_score)  
        self.mean_train_score = np.mean(training_score)
        self.std_train_score = np.std(training_score)
        self.clf_list = clf_list # > > classifier list for majority voting !
        self.n_sample_ = len(y)
        self.idx_pos.clear()

        return self
    
    def set_params(self, param_dict):
        self.clf.set_params(**param_dict)

    def grid_search_optimize(self, X, y, grid_dict, objective = None, verbose=True):
        """ Using specified value of parameters to sweep over
        returns the CLF object with the best cross-validation
        score on those parameters

        Parameters 
        ---------
        grid_dict : dict
            Dictionary of parameters to iterate. Example : grid_dict = {'C':[0.01,0.1,1,10]} 
        """
        if objective is None:
            objective = lambda x : x.cv_score

        vParIterables = list(grid_dict.values())
        vParNames = list(grid_dict.keys())

        best_score = -1

        for values in list(itertools.product(*vParIterables)):
            current_param = dict(zip(vParNames, values))
            self.set_params(current_param)
            self.fit(X, y)
            current_score = objective(self)

            if verbose:
                print(current_param,'\t', current_score)

            if current_score>best_score:
                best_score = current_score
                clf_best = deepcopy(self)
                param_best = deepcopy(current_param)
        if verbose:
            print('Best objective is %.4f with parameters %s'%(best_score, str(param_best)))

        return clf_best 



    def predict(self, X, option='fast'):
        """Returns labels for X (-1, 1)"""
        if option is 'fast':
            mu, inv_sigma = self.scaler_list[0] # choose here median
            return self.clf_list[0].predict(inv_sigma*(X-mu))

        if self.clf_type == 'trivial':
            self._n_sample = len(X)
            return np.zeros(len(X))

        assert self.trained is True, "Must train model first !" 

        # col is clf, row are different data points
        n_clf = len(self.clf_list)
        vote = []

        for i in range(n_clf):
            clf = self.clf_list[i]
            mu, inv_sigma = self.scaler_list[i]
            vote.append(clf.predict(inv_sigma*(X-mu)))

        vote = np.vstack(vote).T
        # row are data, col are clf
    
        y_pred = []
        for x_vote in vote: # majority voting here !
            y_pred.append(most_common(list(x_vote)))

        return np.array(y_pred)#.reshape(-1,1)

    def score(self, X, y):
        y_pred = self.predict(X).flatten()
        return np.count_nonzero(y_pred == y)/len(y)
    
    def optimize_clf(self, idx_folds, param_sweep):
        vParNames = list(param_sweep.keys())
        vParIterables = list(param_sweep.values())
        for values in list(itertools.product(*vParIterables)):
            param = dict(zip(vParNames, values))
            self.param = param
            # <==> evaluate classifier here <==> ... 

    def idx_train_test_split(self, y, train_size = 0.8, n_sample_max = 1000, n_train_min=20, n_split=1): 
        """ 
        Splitting function of uneven populations (can be extremely unbalanced)
        
        Split the unique values in y into a training and test set with
        proportion given by test_size. Also takes into account that for each
        category in both the training and test set, the number of samples cannot exceed
        n_sample_max. This is used to drastically speed-up the error process.
        The number of splits (different folds is specified by n_split)

        Parameters 
        -----------
        y: numpy array (shape = (n_sample))
            labels to split

        train_size: float (default = 0.8)
            Ratio size of the test set
        
        n_sample_max: int (default = 1000)
            Maximum number of samples for either the training or test set
        
        n_split: int (default = 1)
            Number of splits to compute.

        Return 
        -----------
        idx_per_split_concat: list of lenght n_split
            Each element of the returned list is [idx_train, idx_test], the location of the training samples and test samples to use
        """

        test_size = 1. - train_size

        y_unique = np.unique(y)
        idx_pos = {}
        class_count = {}
        n_yu_test = {} # stores the number of examples with category yu to put into the testing set
        n_yu_train = {} # stores the number of examples with category yu to put into the training set

        # the goal here is to make sure you always enough samples from each category ...
        for i, yu in enumerate(y_unique):
            idx_pos[yu] = np.where(y==yu)[0]
            n_yu = len(idx_pos[yu])
            n_test = int(test_size*n_yu)
            n_train = int(train_size*n_yu)

            if n_train < n_train_min: # force a 80/20 train-test split
                n_test = int(0.2*n_yu)
                n_train = int(0.8*n_yu)
            
            if n_train > n_sample_max: # too many samples
                n_train = n_sample_max

            n_yu_test[yu] = n_test
            n_yu_train[yu] = n_train

        idx_split = []
        for n in range(n_split):
            idx_train = []
            idx_test =[]
            for i, yu in enumerate(y_unique):
                np.random.shuffle(idx_pos[yu])
                n_train = n_yu_train[yu]
                n_test = n_yu_test[yu]
                idx_train += list(idx_pos[yu][:n_train])
                idx_test += list(idx_pos[yu][n_train:(n_train+n_test)])
            idx_split.append([idx_train, idx_test])

        return idx_split