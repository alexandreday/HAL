'''
Created on Jan 16, 2017

@author: Alexandre Day
'''

import numpy as np
import matplotlib as mpl
#mpl.use('TkAgg') # makes it better, so doesn't have to deal with framework, etc.
from matplotlib import pyplot as plt

import matplotlib.patheffects as PathEffects
from fdc.mycolors import COLOR_PALETTE

def set_nice_font(size = 18, usetex=False):
    font = {'family' : 'serif', 'size'   : size}
    plt.rc('font', **font)
    if usetex is True:
        plt.rc('text', usetex=True)

def cluster_w_label(X, y, Xcluster=None, show=True, savefile = None, fontsize =15, psize = 20, title=None, w_label = True, figsize=None,
     dpi=200, alpha=0.7, edgecolors=None, cp_style=1, w_legend=False, outlier=True):

    if figsize is not None:
        plt.figure(figsize=figsize)
    y_unique_ = np.unique(y)

    palette = COLOR_PALETTE(style=cp_style)
    idx_centers = []
    ax = plt.subplot(111)
    all_idx = np.arange(len(X))
    
    if outlier is True:
        y_unique = y_unique_[y_unique_ > -1]
    else:
        y_unique = y_unique_
    n_center = len(y_unique)

    for i, yu in enumerate(y_unique):
        pos=(y==yu)
        Xsub = X[pos]
        plt.scatter(Xsub[:,0], Xsub[:,1], c=palette[i], s=psize, alpha=alpha, edgecolors=edgecolors, label = yu)
        
        if Xcluster is None:
            Xmean = np.mean(Xsub, axis=0)
            idx_centers.append(all_idx[pos][np.argmin(np.linalg.norm(Xsub - Xmean, axis=1))])

    if outlier is True:
        color_out = {-3 : '#ff0050', -2 : '#9eff49', -1 : '#89f9ff'}
        for yi in [-3, -2, -1]:
            pos = (y == yi)
            if np.count_nonzero(pos) > 0:
                Xsub = X[pos]
                plt.scatter(Xsub[:,0], Xsub[:,1], c=color_out[yi], s=psize, rasterized=True, alpha=alpha, marker="2",edgecolors=edgecolors, label = yi)
            

    if w_label is True:
        if Xcluster is not None:
            centers = Xcluster
        else:
            centers = X[idx_centers]
        for xy, i in zip(centers, y_unique) :
            # Position of each label.
            txt = ax.annotate(str(i),xy,
            xytext=(0,0), textcoords='offset points',
            fontsize=fontsize,horizontalalignment='left', verticalalignment='left'
            )
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=5, foreground="w"),
                PathEffects.Normal()])
        
    
    xmin,xmax = plt.xlim()
    ymin,ymax = plt.ylim()
    dx = xmax - xmin
    dy = ymax - ymin
    plt.xticks([])
    plt.yticks([])
    
    if title is not None:
        plt.title(title,fontsize=fontsize)
    if w_legend is True:
        plt.legend(loc='best')

    plt.tight_layout()
    if savefile is not None:
        if dpi is None:
            plt.savefig(savefile)
        else:
            plt.savefig(savefile,dpi=dpi)

    if show is True:
        """ plt.draw()
        plt.pause(0.5) """
        plt.show()
        plt.clf()
        plt.close()
        
    return ax

def plot_graph(X_tsne, Aij, node_label, node_center, node_value):
    return
    """ def plot_kNN_graph(self, X_tsne):
        idx_center = find_position_idx_center(X_tsne, self.ypred, np.unique(self.ypred), self.density_cluster.rho)
        self.kNN_graph.plot_kNN_graph(idx_center, X=X_tsne) """
        # plotting intermediate results
        #if self.plot_inter is True:
        #    cluster_w_label(X_tsne, self.ypred)
        #if self.plot_inter is True:
        #    self.plot_kNN_graph(X_tsne)

def cluster_w_label_plotly(X, y, size=4):

    import plotly.plotly as py
    import plotly.graph_objs as go
    import plotly.offline as of
    from plotly.graph_objs import Figure, Data, Layout
    
    palette = COLOR_PALETTE(style=1)

    data = []
    for i, yu in enumerate(np.unique(y)):
        pos = (y == yu)
        trace = go.Scattergl(
        x = X[pos,0],
        y = X[pos,1],
        mode = 'markers',
        name = 'cluster '+str(yu),
        marker = dict(
            color = palette[i],
            line = dict(width = 0),
            size=size
        )
        )
        data.append(trace)
    
    fig=Figure(data=Data(data),layout=Layout(
    font = {'family' : 'helvetica, sans serif', 'size'   : 12},
    hovermode='closest',
        legend ={
            'font':{
                'size':14
                }
        }
    )
    )
    #py.iplot(data, filename='compare_webgl')

    of.plot(fig)

