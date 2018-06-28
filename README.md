# Hierarchical Agglomerative Learning (HAL)
Package for performing clustering for high-dimensional data. This packages uses heavily scikit-learn and fft accelerated t-SNE. 

# Requirement:
Python 3.6 or later versions.

# Installing (once)
Activate an [Anaconda](https://conda.io/docs/user-guide/tasks/manage-environments.html) Python 3 environment
```
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

# Generate some data. 
X,y = make_blobs(10000,12,10) # 10 gaussians in 12 dimensions, 10000 data points

# The HAL constructor has many optional parameters, documentation coming soon
model = HAL(clf_type='svm') # using linear SVMs (fastest) for agglomeration. Other options are 'rf' and 'nb' (random forest, and naive bayes)

# builds model -> will save data in file info_hal
model.fit(X)

# rendering of results using javascript
model.plot_tree()

# Now that your model is fitted, can predict on data (either new or old), using a cross-validation score of 0.95
ypred = model.predict(X, cv=0.95)

# The fitted model information is in directory info_hal. To reload that information for later use, just:
model.load()
```
