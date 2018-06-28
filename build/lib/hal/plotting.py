'''
Created on Jan 16, 2017

@author: Alexandre Day
'''

import numpy as np
import matplotlib as mpl
mpl.use('TkAgg') # makes it better, so doesn't have to deal with framework, etc.
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
        plt.scatter(Xsub[:,0], Xsub[:,1],c=palette[i], s=psize, alpha=alpha, edgecolors=edgecolors, label = yu)
        
        if Xcluster is not None:
            Xmean = Xcluster[i]
        else:
            Xmean = np.mean(Xsub, axis=0)
        #Xmean = np.mean(Xsub,axis=0)
        idx_centers.append(all_idx[pos][np.argmin(np.linalg.norm(Xsub - Xmean, axis=1))])

    if outlier is True:
        color_out = {-3 : '#ff0050', -2 : '#9eff49', -1 : '#89f9ff'}
        for yi in [-3, -2, -1]:
            pos = (y == yi)
            if np.count_nonzero(pos) > 0:
                Xsub = X[pos]
                plt.scatter(Xsub[:,0], Xsub[:,1], c=color_out[yi], s=psize, rasterized=True, alpha=alpha, marker="2",edgecolors=edgecolors, label = yi)
            

    if w_label is True:
        centers = X[idx_centers]
        for xy, i in zip(centers, y_unique) :
            # Position of each label.
            txt = ax.annotate(str(i),xy,
            xytext=(0,0), textcoords='offset points',
            fontsize=fontsize,horizontalalignment='center', verticalalignment='center'
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
