#!/usr/bin/python3
import numpy as np

class Node:
    """
    A node can be an item, a user or a feature. Each node represents a 
    random variable taking values in the range stored in the attribute self.range 
    """

    def __init__(self, index):
        
        self.index = index
        self.support = None

    def set_probs(self, prob):
        # self.values != prob.support:
        # raise("The support of the distribution isn't compatible with the node's")
        self.prob = prob

class User(Node):
    def __init__(self, index, cb=False, cf=False):
        super().__init__(index)
        self.support = np.arange(0, 10 + 1)
        self.is_cb_node = cb
        self.is_cf_node = cf

class Item(Node):
    def __init__(self, index):
        super().__init__(index)
        self.support = np.arange(0, 1 + 1)
    
    def set_features(self, features):
        self.features = features

class Feature(Node):
    def __init__(self, index):
        super().__init__(index)
        self.support = np.arange(0, 1 + 1)


