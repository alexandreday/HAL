import os
import numpy as np

class FOUT:
    def __init__(self, fname='tmp.txt'):
        self.fname = 'tmp.txt'
        if os.path.isfile(fname):
            os.system("rm %s"%self.fname) # removes file if it exist

    def write(self, mystring): # this will append
        fout = open(self.fname,'a')
        fout.write(mystring+'\n')
        fout.close()
    
def make_file_name(param):
    out = [
        ['perplexity', 'perp',int],
        ['n_iteration_tsne', 'niter', int],
        ['outlier_ratio', 'outratio', float],
        ['nn_pure_ratio', 'pureratio', float],
        ['min_size_cluster', 'minsize', int],
        ['nh_size', 'nhsize', int],
        ['eta', 'eta', float],
        ['fdc_test_ratio_size', 'testfdcsize', float]
    ]
    info_str = ''
    for e in out:
        v = param[e[0]]
        n = e[1]
        t = e[2]

        if t is int:
            info_str += '%s=%i_'%(n,int(v))
        else:
            info_str += '%s=%.2f_'%(n, float(v))

    return info_str[:-1]

def print_param(my_dict):
    for k, v in my_dict.items():
        print("[HAL] {0:<20s}{1:<4s}{2:<20s}".format(str(k),":",str(v)))
    print('\n')

def find_position_idx_center(X_tsne, ypred, idx_center, rho):
    """
    Returns dict of idx_center to cartesian Xtsne coordinates
    """
    assert len(X_tsne) == len(ypred)

    idx_center_pos = {}
    for idx in idx_center:
        pos = np.where(ypred == idx)[0]
        pos_center = np.argmax(rho[pos])
        idx_center_pos[idx] = X_tsne[pos[pos_center]]

    return idx_center_pos