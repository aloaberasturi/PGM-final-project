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
        self.probs = []

    def add_probability(self, probability_distribution):
        probability_distribution.check_integrity()
        self.probs.append(probability_distribution)

    def get_prob(self, sample, evidence):
        prob = [p for p in self.probs][0]# if p.evidence == evidence][0]
        return prob.get_prob(sample)

    
    def add_sample(self, sample, prob_value, evidence):
        try:
            prob_distribution = self.get_prob(sample, evidence)
        except IndexError:
            prob_distribution = ProbabilityDistribution(self, evidence=evidence)
            prob_distribution.add_sample(sample, prob_value)
        self.add_probability(prob_distribution)

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
        Gets rating for the target song
        """
        return matrix_S.loc[matrix_S['user_id'] == self.index][song.index].values[0]


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


