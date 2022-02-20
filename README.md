# Hierarchical Agglomerative Learning (HAL)
Package for performing clustering for high-dimensional data. This packages uses heavily scikit-learn and FFT accelerated t-SNE.

# System requirement
* Has been tested on latest version of OS X and Linux
* (OPTIONAL) The dynamical plotting requires Chrome, Safari or Firefox (without *ad blockers* !).
# Requirement:
Python 3.6 or later versions.

# Installing (once)
Activate an [Anaconda](https://conda.io/docs/user-guide/tasks/manage-environments.html) Python 3 environment
```
source activate YOUR_CONDA_ENVIRONMENT
conda config --add channels conda-forge
conda install cython numpy fftw scipy
pip install hal-x
```
# Updating
For future versions of the package, you can upgrade using:
```
pip install hal-x --upgrade
```
# Small example
```
from hal import HAL  # this imports the class HAL() 
from sklearn.datasets import make_blobs
from hal import metric
import numpy as np

# Setting random seed, in case you want to re-run example but keep saved data in info_hal/ 
np.random.seed(0)

# Generate some data. 
X,ytrue = make_blobs(n_samples=10000,n_features=12,centers=10) # 10 gaussians in 12 dimensions, 10000 data points

# The HAL constructor has many optional parameters (documentation coming soon)
model = HAL(clf_type='svm', warm_start=False) # using linear SVMs (fastest) for agglomeration. Other options are 'rf' and 'nb' (random forest, and naive bayes)

# builds model -> will save data in file info_hal
model.fit(X)

# rendering of results using javascript (with optional feature naming)
feature_name = ['feat_%i'%i for i in range(12)]

# Now that your model is fitted, let's visualize the clustering hierarchy. This will give us an idea of how to choose the cv score for the final prediction
model.plot_tree(feature_name = feature_name)

# In the visualization, we see that cv above ~ 0.86 will yield perfect clustering
# In order to generate the corresponding final clustering labels, use the predict function
ypred95 = model.predict(X, cv=0.95) # Predict with score of 0.95
ypred50 = model.predict(X, cv=0.5) # Predict with score of 0.5

# You can check the accuracy of your predictions against the true labels using the convenient metric functions:
print("Normalized mutual information score: %.4f"%metric.NMI(ypred95, ytrue))
print("Normalized mutual information score: %.4f"%metric.NMI(ypred50, ytrue))

# The fitted model information is in directory info_hal. To reload that information for later use, just:
model.load()

# To load t-SNE coordinates:
model.load('tsne')
```

