#!/usr/bin/python3
from utils import compute_weights
from probabilityDistribution import ProbabilityDistribution

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

    # Propagate to Acb and Ui using Theorem 1.// (see Fig. 3b).

    # *************** Collaborative propagation ***************

    # For each Uk+ set Pr(Uk = rk,j|evcf) = 1.// Collaborative evidence
    # Propagate to Acf node using Theorem 1.// (see Fig. 3c)
    # Combine content-based and collaborative likelihoods at hybrid node Ah
    # Select the predicted rating.
