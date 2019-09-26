'''
Created on Sep 26, 2019

@author: Alexandre Day

Description : 
    Class for embedders and providing different methods for embedding data
'''

import error as err

class EMBEDDER()

    def __init__(self, method='tsne', parameters=None):
        self.method='tsne'
        self.parameters=parameters
        if parameters == None:
            if self.method == 'tsne':
                self.parameters = {
                    "perplexity":40
                }
            elif self.method == 'umap':
                self.parameters = {
                    "perplexity":40
                }
            else:
                err.throw("Only 'method' allowed are ['tsne','umap']")

        if self.method == 'tsne' and parameters == None:
            self.parameters = {
               perplexity:40
            }
        
    def fit(self, X):
        pass
