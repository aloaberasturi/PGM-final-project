#!/usr/bin/python3
import numpy as np

class Node:
    """
    A node can be an item, a user or a feature. Each node represents a 
    random variable taking values in the range stored in the attribute self.range 
    """

    def __init__(self):
        pass

    def set_values(self, max_value):
        self.range = np.arange(0, max_value)

    def set_probs(self, prob):
        self.prob = prob

