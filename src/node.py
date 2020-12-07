#!/usr/bin/python3
import numpy as np
from probabilityDistribution import ProbabilityDistribution

class Node:
    """
    A node can be an item, a user or a feature. Each node represents a 
    random variable taking values in the range stored in the attribute self.range 
    """

    def __init__(self, index):
        self.index = index
        self.support = None
        self.probs = []

    def add_probability(self, consequence, probability_values):
        distribution = ProbabilityDistribution(self, consequence, probability_values)
        self.probs.append(distribution)

class User(Node):
    def __init__(self, index, cb=False, cf=False):
        super().__init__(index)
        self.support = np.arange(0, 10 + 1)        
        self.is_cb_node = cb
        self.is_cf_node = cf

class Item(Node):
    def __init__(self, index, is_target=False):
        super().__init__(index)
        self.support = np.arange(0, 1 + 1)
        self.is_target = is_target

    def set_features(self, features):
        self.features = features

class Feature(Node):
    def __init__(self, index):
        super().__init__(index)
        self.support = np.arange(0, 1 + 1)


