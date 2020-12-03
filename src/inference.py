#!/usr/bin/python3
from utils import compute_weights

def perform_inference(graph, matrix_D, matrix_S):

    #  Content-based propagation:

    # If (evcb == Ij)// Item instantiation (see Fig. 3a)
    #   set Pr(ij,1jev) = 1
    #   Compute Pr(Fk|ev) using Theorem 2//propagating towards features
    # else
    #    for each Fk that is a parent of Ij set Pr(Fk = 1|ev) = 1.// Features Inst.
    # Propagate to items using Theorem 1.
    # Propagate to Acb and Ui using Theorem 1.// (see Fig. 3b).

    #  Collaborative propagation:

    # For each Uk+ set Pr(Uk = rk,j|evcf) = 1.// Collaborative evidence
    # Propagate to Acf node using Theorem 1.// (see Fig. 3c)
    # Combine content-based and collaborative likelihoods at hybrid node Ah
    # Select the predicted rating.
