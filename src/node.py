#!/usr/bin/python3
import numpy as np
from probabilityDistribution import ProbabilityDistribution

class Node:
    """
    A node can be an item, a user or a feature. Each node represents a 
    random variable taking values in the range stored in the attribute self.support 
    """

    def __init__(self, index):
        self.index = index
        self.prob = None

    def add_probability(self, probability_distribution):
        self.prob = probability_distribution

    def get_prob(self, sample):
        return self.prob.get_prob(sample)


class User(Node):
    def __init__(self, index, cb=False, cf=False):
        super().__init__(index)
        self.is_cb = cb
        self.is_cf = cf
        self.rating = {}
        self.support = np.arange(0, 10 + 1) # ratings go from 0 to 10. 0 is only used when
                                            # the user hasn't rated   
    
    def get_rating(self, matrix_S, song):
        """
        Gets rating for a song given the scoring matrix
        """
        return matrix_S.loc[matrix_S['user_id'] == self.index][song.index].values[0]

class Item(Node):
    def __init__(self, index, is_target=False):
        super().__init__(index)
        self.support = np.arange(0, 1 + 1)
        self.is_target = is_target
    
    def set_as_target(self, is_target=True):
        self.is_target = is_target

class Feature(Node):
    def __init__(self, index):
        super().__init__(index)
        self.support = np.arange(0, 1 + 1)


