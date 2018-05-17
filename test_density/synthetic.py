from sklearn.datasets import make_blobs
from vac import DENSITY_PROFILER
from fdc import plotting, FDC
import numpy as np

def main():
    np.random.seed(0)
    X, y =  make_blobs(5000, n_features=2, centers = 10)
    model_fdc = FDC()
    model_DP = DENSITY_PROFILER(model_fdc).fit(X)
    y = model_DP.y
    plotting.cluster_w_label(X, y)


if __name__ == "__main__":
    main()