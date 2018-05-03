from sklearn.datasets import make_blobs
import numpy as np
from matplotlib import pyplot as plt
import pickle
from vac import CLUSTER

def main():
    
    np.random.seed(0)
    X, y = make_blobs(n_samples=5000, n_features=30, centers=10)
    model = CLUSTER(root='b1_results/', run_tSNE='auto')
    
    tree, scaler = model.fit(X)
    ypred = tree.predict(scaler.transform(X))

    root = 'b1_results/'
    pickle.dump(ypred, open(root+'ypred.pkl','wb'))

if __name__ == "__main__":
    main()