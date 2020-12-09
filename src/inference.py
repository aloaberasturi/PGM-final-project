#!/usr/bin/python3
from probabilityDistribution import ProbabilityDistribution
from node import Feature, Item, User
import numpy as np
import math
import utils

def perform_inference(graph, matrix_D, matrix_S, item_instantiation=True):

    # *************** Content-based propagation ***************

    # 1.a) If (evcb == Ij): // Item instantiation (see Fig. 3a)
    if item_instantiation:
        ev_cb = graph.get_target_item()

    #   1.a.1) set Pr(ij,1jev) = 1 
    #   (Prob(item = not relevant) is 0.0.  Prob(item = relevant) is 1.0)
        ev_cb.add_probability(ProbabilityDistribution(ev_cb, evidence=ev_cb, probabilities=[0.0, 1.0]))
                
    #   1.a.2) Compute Pr(Fk|ev) using Theorem 2
        for f in graph.feature_nodes:
            prob = theorem_2(graph, f, matrix_S, evidence=ev_cb)
        probability = ProbabilityDistribution(f, evidence=ev_cb, probabilities=[1.0 - prob, prob])
        f.add_probability(probability)

    # 1.b) else:
    elif not item_instantiation:
        ev_cb = graph.get_target_features()

    #   1.b.1) for each Fk that is a parent of Ij set Pr(Fk = 1|ev) = 1.// Features Inst.
        for feature in graph.feature_nodes:
            if feature in ev_cb:
                probs = [0.0, 1.0]

            else: # if F is not a parent of Ij, its probability is a-priori
                a_priori = a_priori_probability(graph, uniform=True, feature_node=None)
                if abs(1.0 - a_priori) < 0.00000001:
                    a_priori = 1.0
                probs = [1.0 - a_priori, a_priori]

            probability = ProbabilityDistribution(feature, evidence=ev_cb, probabilities=probs)
            feature.add_probability(probability)

    #   1.b.2) Propagate to items using Theorem 1.  
        items = graph.item_nodes
        propagate(items, matrix_S, graph, ev_cb, layer='features-items')


    #   1.b.3) Propagate to Acb and Ui using Theorem 1.
        for item in items:
            users = graph.get_children(item) 
            if users:
                propagate(users, matrix_S, graph, ev_cb, layer='items-users') #bug here


    # *************** Collaborative propagation ***************

    # For each Uk+ set Pr(Uk = rk,j|evcf) = 1.// Collaborative evidence
    ev_cf = graph.get_u_plus(matrix_S)

    for user in ev_cf:
        target_item = graph.get_target_item()
        try:
            rating = int(user.get_rating(target_item))
        except ValueError: 
            print('This user does not belong to U+!')

        probability_values = np.zeros(len(user.support)).tolist()
        probability_values[rating - 1] = 1.0
        probability = ProbabilityDistribution(user, evidence=ev_cf, probabilities=probability_values)
        user.add_probability(probability)

    # Propagate to Acf node using Theorem 1.// (see Fig. 3c)

    # Combine content-based and collaborative likelihoods at hybrid node Ah

    # Select the predicted rating.

def propagate(source_nodes, matrix_S, graph, evidence, layer):

    for node in source_nodes:
        probs = []
        if layer == 'features-items':
            p = theorem_1(graph, matrix_S, node, 1, evidence) 
            if abs(1.0 - p) < 0.00000001:
                p = 1.0
            probs.extend([1.0 - p, p])
        else: 
            for s in node.support:
                p = theorem_1(graph, matrix_S, node, s, evidence)
                probs.append(p)
        probability = ProbabilityDistribution(node, evidence, probabilities=probs)
        node.add_probability(probability) 

def theorem_1(graph, matrix_S, x, s, evidence):
    """
    Function implementing Theorem 1
    
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

    parents = graph.get_parents(x)
    prob = 0.0
    for y in parents:
        for k in y.support: 
            prob += w(y, k, x, s, graph, matrix_S) * y.get_prob(k, evidence)
    return prob

def theorem_2(graph, f, matrix_S, evidence):
    if f in graph.get_parents(evidence):
        p = a_priori_probability(graph)
        prob = [1.0 - p, p]

    else:
        p = a_priori_probability(graph)
        weights = w(f, 1, evidence, 1, graph, matrix_S)
        sum = 0.0
        for item_feature in graph.get_parents(evidence):
            sum += w(item_feature, 1, evidence, 1, graph, matrix_S) * a_priori_probability(item_feature, 1)        
        prob = (p + weights * p * (1.0 - p)) / sum    

    return prob


def a_priori_probability(graph, uniform=True, feature_node=None):    
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
        au_row = matrix_S.loc[matrix_S['user_id'] == x.index].squeeze()
        I_u = float((au_row.drop('user_id') != 0.0).sum())
        rating = x.get_rating(matrix_S, y)

        if k == 1:
            if s == rating:
                print('acierta!! rating: %i' % rating)
                return 1.0 / I_u

            elif (s != rating):
                return 0.0

        if k == 0:
            if s == 0:
                return 1.0 / I_u

            else:
                return 0.0

    elif (isinstance(y, User) and isinstance(x, User)):
        # x is a_cf
        x_parents = graph.get_parents(x)
        norm = normalize(x_parents, x, matrix_S)
        x_row = matrix_S.loc[matrix_S['user_id'] == x.index].squeeze().drop('user_id')
        y_row = matrix_S.loc[matrix_S['user_id'] == y.index].squeeze().drop('user_id')
        r_sim = utils.compute_similarity(y_row,x_row) / norm
        p = probability_star(x,s,y,k,matrix_S)
        if ( (k >= 1) and (s >= 1) ):
            return r_sim * p

        elif ((s==0) and (k==0)):
            return r_sim

        elif ((s==0)):
            return 0

        elif ((k==0)):
            return 0

def normalize(parents, a_cf, matrix_S):
    sum = 0.0
    for u in parents:
        a_row = matrix_S.loc[matrix_S['user_id'] == a_cf.index].squeeze().drop('user_id')
        u_row = matrix_S.loc[matrix_S['user_id'] == u.index].squeeze().drop('user_id')
        sum += utils.compute_similarity(a_row, u_row)
    return sum

def m_operator(node, graph):
    sum = 0.0
    for feature in graph.get_parents(node):
        n_k = len(graph.get_children(feature))
        m = len(graph.item_nodes)
        sum += math.log((m / n_k) + 1)
    return sum

def probability_star(a, s, u, t, matrix_S):
    a_row = matrix_S.loc[matrix_S['user_id'] == a.index].squeeze().drop('user_id')
    u_row = matrix_S.loc[matrix_S['user_id'] == u.index].squeeze().drop('user_id')
    is_common_score = [True if (i!=0 and j!=0) else False for i, j in zip(a_row.values.tolist(), u_row.values.tolist())]
    n_u_a = sum(is_common_score)
    n_u_t = u_row[is_common_score].values.tolist().count(t)
    numerator = (n_u_a + 1) /  max(a.support) # I THINK that '#r' in eq (7) is this max
    denominator = (n_u_t + 1)
    return numerator / denominator