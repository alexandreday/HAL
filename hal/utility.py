import os, sys
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

def compute_cluster_stats(Xsubset, size):
    median = np.median(Xsubset,axis=0)
    std = np.std(Xsubset,axis=0)
    return  {"mu":median, "std":std, "size":len(Xsubset), "ratio":len(Xsubset)/size}

def float_fmt(param_name, nbr, decimal_place = 2):
    nbr_fmt = ("{:0<%s.%sf}"%(decimal_place+2, decimal_place)).format(nbr);
    return param_name + "=" + nbr_fmt

def int_fmt(param_name, nbr, max_value = 1000):
    assert nbr < max_value, "Wrong integer parameter value, must be smaller than %i"%max_value
    n_max = len(str(max_value))
    nbr_fmt = str(nbr).zfill(n_max)
    return param_name + "=" + nbr_fmt

def str_fmt(param_name, my_str, width = 10):
    assert len(my_str)<width, "string parameter is too big"

    
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

def decode():
    file_name = sys.argv[1]
    #check file name is in ok format
    ls_parameters = file_name.split('_')
    param_dict = {}
    for param in ls_parameters:
        name, value = param.split("=")
        param_dict[name] = format_str(name, value)
    print("----> File information <------- \n")
    print(param_dict)
    return param_dict

