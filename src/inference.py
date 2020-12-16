#!/usr/bin/python3
from probabilityDistribution import ProbabilityDistribution
import inferenceUtils as utils
import topologyUtils

def perform_inference(graph, matrix_D, matrix_S):
    """
    Function that performs the inference in the network once the evidence is inserted. 
    The content evidence can either be the features of the target song or the target song.
    The collaborative evidence are the user nodes of users in U+
    
    Parameters: 
    -----------
    graph: Graph object
    matrix_D: pd.DataFrame
    matrix_S: pd.DataFrame
    item_instantiation: bool

    Returns:
    --------
    int: Predicted rating of the target song by the active user

    """

    # *************** Content-based propagation ***************

    # 1.a.1) For each Fk  Pr(Fk = 1| ev) = 1
    ev_cb = graph.get_target_features()

    # 1.a.2) for each Fk that is a parent of Ij set Pr(Fk = 1|ev) = 1.// Features Inst.
    features = graph.get_feature_nodes()
    utils.initiate_features_probs(features, graph, ev_cb)

    # 1.a.3) Propagate to items using Theorem 1.  
    items = graph.get_item_nodes()
    utils.propagate_downwards(items, matrix_S, graph, ev_cb, layer='features-items')

    # 1.a.4) Propagate to Acb and Ui using Theorem 1.
    for i in items:
        users = graph.get_children(i)
        if users != []:
            utils.propagate_downwards(users, matrix_S, graph, ev_cb, layer='items-users')

    # *************** Collaborative propagation ***************

    # 1.c) For each Uk+ set Pr(Uk = rk,j|evcf) = 1.// Collaborative evidence
    ev_cf = topologyUtils.get_u_plus(graph.get_user_nodes(), graph.get_target_item(), matrix_S)
    utils.initiate_u_plus_probs(ev_cf, graph, matrix_S, ev_cf)

    # Propagate to Acf node using Theorem 1.// (see Fig. 3c)
    utils.propagate_downwards([graph.get_a_cf()], matrix_S, graph, ev_cf, layer='users-a_cf')

    # Combine content-based and collaborative likelihoods at hybrid node Ah
    a_cf = graph.get_a_cf()
    a_cb = graph.get_a_cb()
    

    # Select the predicted rating.


