#!/usr/bin/python3
from probabilityDistribution import ProbabilityDistribution
from node import Feature, Item, User
import numpy as np
import math
import topologyUtils

"""
This module contains the necessary utilities to perform inference in inference.py
"""

def propagate_downwards(sink_nodes, matrix_S, graph, evidence, layer):
    """
    Propagates information from parent nodes to children nodes.

    Parameters:
    -----------
    sink_nodes: set of children nodes on which to perform the inference
    matrix_S: pd.DataFrame
    graph: Graph
    evidence: node
            Depending on the node type, a set of features, items or users nodes
    layer: str
            A string with the name of the level on which we perform the inference
    """
    for node in sink_nodes:
        if node.prob != None:
            return
        probs = []
        if layer == 'features-items':
            p = theorem_1(graph, matrix_S, node, 1, evidence) 
            if abs(1.0 - p) < 0.00000001:
                p = 1.0
            probs.extend([1.0 - p, p])
        else: # layer items-users or users-a_cf
            for s in node.support:
                p = theorem_1(graph, matrix_S, node, s, evidence)
                probs.append(p)

        probability = ProbabilityDistribution(node, evidence, probabilities=probs)
        node.add_probability(probability) 

def initiate_features_probs(features, graph, evidence):
    """
    A function that initializes the probabilities of the features

    Parameters:
    -----------
    features: list
            A list of nodes
    graph: Graph
    evidence: list
            A list of nodes 
    """
    for feature in graph.feature_nodes:
        if feature in evidence:
            probs = [0.0, 1.0]

        else: # if F is not a parent of Ij, its probability is a-priori
            a_priori = a_priori_probability(graph, uniform=True, feature_node=None)
            if abs(1.0 - a_priori) < 0.00000001:
                a_priori = 1.0
            probs = [1.0 - a_priori, a_priori]

        probability = ProbabilityDistribution(feature, evidence=evidence, probabilities=probs)
        feature.add_probability(probability)

def initiate_u_plus_probs(u_plus, graph, matrix_S, evidence):
    """
    A function to instantiate the probabilities of users in U+

    Parameters:
    -----------
    u_plus: list
            A list of user nodes in U+
    graph: Graph
    matrix_S: pd.DataFrame
    evidence: list
            A list of nodes

    """
    for user in u_plus:
        target_item = graph.get_target_item()
        try:
            rating = int(user.get_rating(matrix_S, target_item))
        except ValueError: 
            print('This user does not belong in U+!')
        probability_values = np.zeros(len(user.support)).tolist()
        probability_values[rating] = 1.0
        probability = ProbabilityDistribution(user, evidence=evidence, probabilities=probability_values)
        user.add_probability(probability)


def theorem_1(graph, matrix_S, x, s, evidence):
    """
    Function implementing Theorem 1. 
    
    Parameters
    ----------
    graph : Graph
    matrix_S: pd.DataFrame
    x: Node
    s: int
        x's state
    evidence: list

    Returns
    -------
    int: 
        Probability of x being in state s given the evidence P(x_s|ev)    
    """
    prob = 0.0
    for y in graph.get_parents(x):
        for k in y.support: 
            prob += w(y, k, x, s, graph, matrix_S) * y.get_prob(k)
    return prob

def a_priori_probability(graph, uniform=True, feature_node=None):    
    """
    An auxiliary function to initiate the probability distributions in initiate_features_probs()
    """
    if uniform:
        return 1.0 / len(graph.feature_nodes)

    elif (not uniform and feature_node): 
        m = len(graph.item_nodes)
        n_k = len([e.x for e in graph.feature_item_edges if e.x.index == feature_node.index])
        numerator = n_k + 0.5
        denominator = m + 1.0
        return numerator / denominator


def w(y, k, x, s, graph, matrix_S):

    """
    Function that returns the weight for a given pair of nodes

    Parameters
    ----------
    y: Node
    k: int
        y's state
    x: Node
    s: int
        x's state
    graph: Graph
    matrix_S: pd.DataFrame
    
    Returns
    -------
    w: double
    """

    if (isinstance(y, Feature) and isinstance(x, Item)):
        if k == 1:
            n_k = len(graph.get_children(y))
            m = len(graph.item_nodes)
            return ( (1.0 / m_operator(x, graph)) * math.log((m / n_k) + 1.0))
        return 0.0        

    elif (isinstance(y, Item) and isinstance(x, User)):
        I_u = len(graph.get_parents(x))
        rating = x.get_rating(matrix_S, y)

        if k == 1:
            if s == rating:
                return 1.0 / I_u

            elif (s != rating):
                return 0.0

        if k == 0:
            if s == 0:
                return 1.0 / I_u

            else:
                return 0.0

    elif (isinstance(y, User) and isinstance(x, User)):      
        norm = normalize(x, graph, matrix_S)
        x_row = matrix_S.loc[matrix_S['user_id'] == x.index].squeeze().drop('user_id')
        y_row = matrix_S.loc[matrix_S['user_id'] == y.index].squeeze().drop('user_id') 
        r_sim = topologyUtils.compute_similarity(y_row, x_row) / norm
        p = probability_star(x, s, y, k, matrix_S)

        if (s != 0 and k != 0):
            return r_sim * p

        elif (s == 0 and k == 0):
            return r_sim

        else: return 0.0

def m_operator(node, graph):
    """
    Auxiliary function for the computation of weights in feature-item layer
    """
    sum = 0.0
    for feature in graph.get_parents(node):
        n_k = len(graph.get_children(feature))
        m = len(graph.item_nodes)
        sum += math.log((m / n_k) + 1)
    return sum 

def normalize(a_cf, graph, matrix_S):
    """
    Auxiliary function for the computation of weights in user-ac_f layer. See eq.7)
    """
    parents = graph.get_parents(a_cf)
    sum = 0.0
    for u in parents:
        a_row = matrix_S.loc[matrix_S['user_id'] == a_cf.index].squeeze().drop('user_id')
        u_row = matrix_S.loc[matrix_S['user_id'] == u.index].squeeze().drop('user_id')
        sum += topologyUtils.compute_similarity( u_row, a_row)
    return sum


def probability_star(a, s, u, t, matrix_S):
    """
    Auxiliary function for the computation of weights in user-ac_f layer. See eq.7)
    """
    a_row = matrix_S.loc[matrix_S['user_id'] == a.index].squeeze().drop('user_id')
    u_row = matrix_S.loc[matrix_S['user_id'] == u.index].squeeze().drop('user_id')
    is_commonly_scored = [True if (i!=0 and j!=0) else False for i, j in zip(a_row.values.tolist(), u_row.values.tolist())]
    i_n_as = a_row[is_commonly_scored].values.tolist()
    i_n_ut = u_row[is_commonly_scored].values.tolist()

    count = 0.0
    for i,j in zip (i_n_as, i_n_ut):
        if (i == s and j == t):
            count +=1

    n_ut_as = count
    n_ut = i_n_ut.count(t)

    numerator = n_ut_as + (1.0 / max(a.support))
    denominator = n_ut + 1.0
    prob = numerator / denominator
    return prob