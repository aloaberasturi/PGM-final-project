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
        self.support = None
        self.probs = []

    def get_prob(self, sample, consequent):
        prob = [p for p in self.probs if (p.consequent == consequent)][0]
        return prob.get_prob(sample)

    def add_probability(self, consequent, probability_values):
        distribution = ProbabilityDistribution(self, consequent, probability_values)
        self.probs.append(distribution)

class User(Node):
    def __init__(self, index, cb=False, cf=False):
        super().__init__(index)
        self.is_cb = cb
        self.is_cf = cf
        self.rating = {}
        self.support = np.arange(1, 10 + 1) # ratings go from 1 to 10. 0 is only used when
                                            # the user hasn't rated yet  
    
    def set_rating(self, matrix_S, song):
        """
        Sets rating for the target song
        """
        rating = matrix_S.loc[matrix_S['user_id'] == self.index][song.index].values[0]
        self.rating[song.index] = rating
    
    def get_rating(self, song):
        return self.rating[song.index]

class Item(Node):
    def __init__(self, index, is_target=False):
        super().__init__(index)
        self.support = np.arange(0, 1 + 1)
        self.is_target = is_target

    def set_features(self, features):
        self.features = features
    
    def set_as_target(self, is_target=True):
        self.is_target = is_target

class Feature(Node):
    def __init__(self, index):
        super().__init__(index)
        self.support = np.arange(0, 1 + 1)


