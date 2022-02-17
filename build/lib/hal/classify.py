from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import ShuffleSplit
import numpy as np
from collections import Counter
import time

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

    def __init__(self, clf_type='svm', n_bootstrap=10, test_size = 0.8, n_sample_max = 1000, clf_kwargs=None):
        self.clf_type = clf_type
        self.n_bootstrap = n_bootstrap
        self.n_sample_max = n_sample_max
        self.test_size = test_size
        if clf_kwargs is None:
            if clf_type == 'svm':
                self.clf_kwargs = {'kernel':'linear','class_weight':'balanced'}
            else:
                self.clf_kwargs = {'class_weight':'balanced'}
        else:
            self.clf_kwargs = clf_kwargs

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

        s=time.time()

        self.trained = True
        
        if self.clf_type == 'svm':
                clf = SVC(**self.clf_kwargs)
        elif self.clf_type == 'rf':
                if 'max_features' not in self.clf_kwargs.keys():
                    self.clf_kwargs['max_features'] = min([int(X.shape[1]/2), 100])
                if 'n_estimators' not in self.clf_kwargs.keys():
                    self.clf_kwargs['n_estimators'] = 20
                clf = RandomForestClassifier(**self.clf_kwargs)
        elif self.clf_type == 'nb':
                clf = GaussianNB()#**self.clf_kwargs)
        else:
            assert False

        predict_score = []; training_score = []; clf_list = []; xtrain_scaler_list = [];
        zero_eps = 1e-6

        self.y_unique = np.unique(y) # different labels
        assert len(self.y_unique) > 1, "Cluster provided only has a unique label, can't classify !"
        
        dt = 0

        idx_bootstrap_split = self.idx_train_test_split(y, test_size =self.test_size, n_split=self.n_bootstrap, n_sample_max=self.n_sample_max)
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
            #s2 = time.time()
            clf.fit(xtrain, ytrain)
            #dt += (time.time() - s2)
    
            # predict on train set
            t_score = clf.score(xtrain, ytrain) 
            
            training_score.append(t_score)

            # predict on test set # maybe this is a noisy estimate ?
            # 
            p_score = clf.score(xtest[:500], ytest[:500])
            predict_score.append(p_score)

            clf_list.append(clf)
            xtrain_scaler_list.append([mu, inv_sigma])
        
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

        #print('Done ALL in %.4f with training of %.4f'%(time.time()-s, dt))
        return self


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
    
    def train_test_split(self, X, y):
        if self.y_unique is None:
            self.y_unique = np.unique(y)
        if len(self.idx_pos) ==0:
            for yu in self.y_unique:
                self.idx_pos[yu] = np.where(y==yu)[0]
        
        self.n_class = len(self.y_unique)
        if self.class_count is None:
            self.class_count = {yu:len(self.idx_pos[yu]) for yu in self.y_unique}
        
        idx_train=[]; idx_test = [];
        train_ratio = 1.-self.test_size
        n_max = self.n_sample_max

        for yc in self.y_unique: # ===
            
            n_yc = self.class_count[yc]
            np.random.shuffle(self.idx_pos[yc])

            if n_yc > n_max:
                idx_train.append(self.idx_pos[yc][:int(train_ratio*n_max)])
                idx_test.append(self.idx_pos[yc][int(train_ratio*n_max):n_max])
            else:
                nsplit = int(train_ratio*n_yc)
                if nsplit == 0:
                    idx_train.append(self.idx_pos[yc][:int(0.5*n_yc)])
                    idx_test.append(self.idx_pos[yc][int(0.5*n_yc):n_yc])
                else:
                    idx_train.append(self.idx_pos[yc][:nsplit])
                    idx_test.append(self.idx_pos[yc][nsplit:n_yc])
        
        idx_train = np.hstack(idx_train)
        idx_test = np.hstack(idx_test)

        return X[idx_train],X[idx_test],y[idx_train],y[idx_test]


    def idx_train_test_split(self, y, test_size = 0.8, n_sample_max = 1000, n_split=1): 
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

        test_size: float (default = 0.8)
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

        train_size = 1. - test_size

        y_unique, counts = np.unique(y, return_counts=True)
        idx_pos = {}
        class_count = {}
        for i, yu in enumerate(y_unique):
            idx_pos[yu] = np.where(y==yu)[0]
            class_count[yu] = counts[i]

        idx_per_split = [[] for i in range(n_split)]
        
        for yc in y_unique:
            n_yc = class_count[yc]
            if n_yc < 200: 
                # For small clusters take 50/50 splits
                n_yc_test = min([int(0.5 * n_yc), n_sample_max])
                n_yc_train = min([int(0.5 * n_yc), n_sample_max])
            else:
                # For large clusters take specified cluster splits
                n_yc_test = min([int(test_size * n_yc), n_sample_max])
                n_yc_train = min([int(train_size * n_yc), n_sample_max])

            skf = ShuffleSplit(n_splits=n_split, test_size = n_yc_test, train_size = n_yc_train)

            for enum_idx, idx_split in enumerate(skf.split(idx_pos[yc])):
                p = idx_pos[yc][idx_split[0]], idx_pos[yc][idx_split[1]] # train and test set splits
                idx_per_split[enum_idx].append(p)
        
        idx_per_split_concat = []
        for i in range(n_split):
            idx_train=[]
            idx_test = []
            for p in idx_per_split[i]:
                idx_train.append(p[0])
                idx_test.append(p[1])
            idx_train=np.concatenate(idx_train)
            idx_test=np.concatenate(idx_test)
            
            np.random.shuffle(idx_train)
            np.random.shuffle(idx_test)
            idx_per_split_concat.append([idx_train, idx_test])

        return idx_per_split_concat