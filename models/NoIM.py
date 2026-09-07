import torch

class NoIM(object):
    def select_action(self, graph_rep, nf, mask):
        return torch.LongTensor([])
    
    def load(self, filename, lr=None):
        pass