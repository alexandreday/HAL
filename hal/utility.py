import os

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
        ['test_ratio_size', 'testfdcsize', float]
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
