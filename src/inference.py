#!/usr/bin/python3
from probabilityDistribution import ProbabilityDistribution
import numpy as np
from node import Feature, Item, User
import math
import utils

def perform_inference(graph, matrix_D, matrix_S, item_instantiation=True):

    # *************** Content-based propagation ***************

    # 1.a) If (evcb == Ij): // Item instantiation (see Fig. 3a)
    if item_instantiation:
        ev_cb = graph.get_target_item()

    #   1.a.1) set Pr(ij,1jev) = 1 
        ev_cb.add_probability(consequent=None, probability_values=[0.0, 1.0]) # Prob(item = not relevant) is 0.0.  Prob(item = relevant) is 1.0
                
    #   1.a.2) Compute Pr(Fk|ev) using Theorem 2

    # 1.b) else:
    else:
        ev_cb = graph.get_target_features()

    #   1.b.1) for each Fk that is a parent of Ij set Pr(Fk = 1|ev) = 1.// Features Inst.
        consequent = None
        for feature in ev_cb:
            probability_values = [0.0, 1.0]
            feature.add_probability(consequent, probability_values)

    # Propagate to items using Theorem 1.


    # Propagate to Acb and Ui using Theorem 1.
    # OJO!! propagar solo a los elementos en acb U u-!!!
    for i in graph.user_nodes:
        for u in graph.user_nodes:
            if u.is_cf:
                theorem_1(graph, matrix_S, u, 3, i, 1, ev_cb)

    # *************** Collaborative propagation ***************

    # For each Uk+ set Pr(Uk = rk,j|evcf) = 1.// Collaborative evidence
    ev_cf = graph.get_u_plus(matrix_S)
    consequent = None

    for user in ev_cf:
        target_item = graph.get_target_item()
        try:
            rating = int(user.get_rating(target_item))
        except ValueError: 
            print('This user does not belong to U+!')

        probability_values = np.zeros(len(user.support)).tolist()
        probability_values[rating - 1] = 1.0
        user.add_probability(consequent, probability_values)

    # Propagate to Acf node using Theorem 1.// (see Fig. 3c)
    # Combine content-based and collaborative likelihoods at hybrid node Ah
    # Select the predicted rating.


def theorem_1(graph, matrix_S, x, s, y, k, evidence):
    """
    Function implementing Theorem 1
    
    Parameters
    ----------
    graph : Graph
    matrix_S: pd.DataFrame
    x: Node
    y: Node
    s: int
            x's state
    k: int
            y's state
    evidence: list

    Returns
    -------
    int: 
        Probability of x being in state s given the evidence P(x_s|ev)    
    """

    parents = graph.get_parents(x)
    prob = 0.0
    for j in parents:
        for k in j.support: 
            prob += w(y, k, x, s, graph, matrix_S) * y.get_prob(k, evidence)
    return prob

def theorem_2(graph, matrix_S, f, k, i, j, evidence):
    pass

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
            return ( (1.0 / m_operator(x, graph)) * math.log((m / n_k) + 1))
        return 0.0        

    elif (isinstance(y, Item) and isinstance(x, User)):
        au_row = matrix_S.loc[matrix_S['user_id'] == x.index].squeeze()
        I_u = len(au_row.values.nonzero())
        rating = x.get_rating(y)

        if (k == 1 and s == rating):
            return (1.0 / I_u)

        elif (k == 1 and s != rating):
            return 0.0

        elif (k == 0 and s == 0):
            return (1.0 / I_u)

        elif (k == 0 and s != rating):
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
    numerator = (n_u_a + 1) /  max(a.support) # I THINK that '#r' in eq (7)
    denominator = (n_u_t + 1)
    return numerator / denominator