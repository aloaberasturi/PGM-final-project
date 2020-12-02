#!/usr/bin/python3

from topology import create_topology
from inference import perform_inference
import utils

if __name__ == '__main__':

    # A) read songs with features, users and ratings from db

    data = utils.get_data()

    # B) create matrix s and matrix D

    matrix_D = utils.compute_description_matrix(data)
    matrix_S = utils.compute_score_matrix(data)

    # C) compute weights

    [w_fi, w_iu, w_ua] = utils.compute_weights(matrix_D, matrix_S)
    

    # D) choose active user and target song
    
    active_user, target_song = utils.input_query(user, song, matrix_S)

    # E) create topology

    graph = create_topology(matrix_S, matrix_D, active_user, target_song)

    # F) compute ratings
    
    rating = perform_inference(graph)

    # G) classify rating

    user_opinion = utils.check_rating(rating)

    print("User %s would rate song %s with a %i" % (active_user, target_song, rating))
    print("The user might think that %s" % user_opinion)


