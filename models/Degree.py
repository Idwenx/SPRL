import numpy as np
import torch

class Degree(object):
    def __init__(self, k:int = 20):
        self.k=k
        
    def select_action(self, graph_rep, nf, mask):
        seedset = self.rank[:self.k]
        return torch.as_tensor(seedset, dtype=torch.long)
    
    def init_degree(self, out_degree):
        self.degree = out_degree
        self.rank = np.argsort(-out_degree)
    
    def load(self, filename, lr=None):
        pass