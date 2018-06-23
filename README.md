# Hierarchical Agglomerative Learning (HAL)
Package for performing clustering for high-dimensional data. This packages uses heavily scikit-learn and fft accelerated t-SNE. 

# Installing (once)
Activate an [Anaconda](https://conda.io/docs/user-guide/tasks/manage-environments.html) Python 3 environment
```
conda config --add channels conda-forge
conda install cython numpy fftw
pip install hal-x
```
# Updating
Again from your Anaconda Python 3 environment:
```
pip install hal-x --upgrade
```
# Example of use
```
from hal import HAL
from sklearn.datasets import make_blobs

# generate some data
X,y = make_blobs(10000,12,10) # 10 blobs in 12 dimensions, 10000 data points

model = HAL(clf_type='rf')

# builds model and outputs intermediate plots/results
model.fit(X)

# predict new labels

ypred = HAL.predict(X)
```
