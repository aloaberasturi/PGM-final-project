#!/usr/bin/python3
import pandas as pd
import random
from pathlib import Path
import numpy as np
from edge import Edge
from node import Feature
from node import Item

def get_data(data_folder=Path('../' + "data")):
    """
    Receives path to data. Data is stored in three files, namely:
    1) music_data.csv (contains songs ids)
    2) user_data.csv (contains users ids)
    3) ratings.csv (each line is the rating of a user (by id) to a song (by id))

    Parameters:
    -----------
    data_folder: pathlib.Path

    Returns
    -------
    list 
        List of three pd.Series (on per file)
    """

    music_data = data_folder / "music_data.csv"
    music_data = pd.read_csv(music_data)
    music_data.dropna(inplace = True) 
    music_data['name'] = music_data['name'].str.lower()
    user_data = data_folder / "user_data.csv"
    user_data = pd.read_csv(user_data)
    user_data['name'] = user_data['name'].str.lower()
    user_data.dropna(inplace = True) 
    rating_data = data_folder / "ratings.csv"    
    rating_data = pd.read_csv(rating_data)
    rating_data.dropna(inplace = True) 

    return [music_data, user_data, rating_data]

def get_dicts(music_data, user_data, rating_data):
    """
    Parameters
    -----------    
    music_data : pd.dataframe

    user_data: pd.dataframe

    rating_data:pd.dataframe
    
    Returns
    -------
    list
        A list of three dictionaries to translate songs, users and features'
        names to indexes and viceversa
    """

    songs_dict = pd.Series(music_data.song_id.values, index = music_data.name).to_dict() #maybe here?
    users_dict = pd.Series(user_data.user_id.values, index = user_data.name).to_dict()
    features = list(music_data.keys()[2:])
    keys = ["f_%i" % i for i in range(len(features))]
    features_dict = {k:v for (k, v) in zip(keys, features)}

    return [songs_dict, users_dict, features_dict]

def compute_scores(user_data, music_data, rating_data):
    """
    Parameters:
    -----------
    user_data: pd.dataframe
    music_data: pd.dataframe
    rating_data: pd.dataframe

    Returns:
    --------
    pd.dataframe
                Matrix S
    """
    user_ids = user_data.user_id.values.tolist()
    song_ids = music_data.song_id.values.tolist()
    matrix_S = pd.DataFrame(np.zeros((len(user_ids), len(song_ids))), index=user_ids, columns=song_ids)

    for _, row in rating_data.iterrows():
        # Set each value in matrix_S to corresponding rating if exists, otherwise stays 0
        matrix_S[row['song_id']][row['user_id']] = row['rating']

    return matrix_S

def compute_similarity(matrix_S, u1,u2):
    """
    Computes similarity between two users according to Eq.(1)
    Parameters:
    -----------
    matrix_S: pd.dataframe
    u1: User 
    u2: User

    Returns:
    --------
    double
    """
    pass

def compute_weights(matrix_D, matrix_S, active_user): # in process...
    # parameters m and n_k used in w(f,i)
    m = len(matrix_D['song_id'])
    frequencies = {'n_{}'.format(i): sum(matrix_D['f_%s' % i]) for i in range(len(matrix_D.keys()[1:]))}

    # parameter iucb == |I(Ucb)| used in w(i,u)    
    row = matrix_S[matrix_S['user_id'] == active_user]
    iucb = row.drop(['user_id'], axis=1).values.sum(axis=1)

def select_user_and_song(matrix_S):
    """
    Selects a random user as active user and a song that hasn't rated as target song 
    Parameters:
    -----------
    matrix_S: pd.dataframe

    Returns:
    --------
    tuple
         An active user-target song pair
    """
    while True:
        user_index = random.randint(0, len(matrix_S.axes[0]) - 1)
        song_index = random.randint(0, len(matrix_S.axes[1]) - 1)
        rating = matrix_S.iloc[user_index, song_index]
        if (rating ==0):
            break
    song = matrix_S.columns.values[song_index]
    user = matrix_S.index.values[user_index]
    return (user, song)

def get_item_features(item, matrix_D):
    """
    Obtains list of features relevant to item i
    Parameters
    ----------
    item: Item
    matrix_D: pd.dataframe

    Returns
    -------
    list
        A list with the names of the features relevant to item

    """
    # 1) Select row
    aux = matrix_D.loc[matrix_D['song_id'] == item.index]
    # 2) Drop unnecessary columns 
    aux = aux.drop(['song_id'], axis=1)
    aux = aux.replace(0, np.nan)
    row = aux.dropna(how='all', axis=1).columns.tolist()
    return row

def get_edges(feature_nodes, item_nodes, matrix_D):  
    """
    Parameters:
    -----------
    feature_nodes: list
    item_nodes: list
    matrix_D: pd.dataframe

    Returns:
    --------
    list
        A list of the edges from the features in feature_nodes to the
        corresponding items in item_nodes, w/o repetitions
    """          
    pairs = []
    for i in item_nodes:
        feature_ids = get_item_features(i, matrix_D)
        pairs.extend([(f.index, i.index) for f in feature_ids])

    return [Edge(Feature(f), Item(i)) for (f,i) in list(set(pairs))]
    

def get_features(item_nodes, matrix_D):
    """
    Parameters:
    -----------
    item_nodes: list 
    matrix_D: pd.dataframe

    Returns:
    --------
    list
        A list of all the features that are relevant to the 
        given list of item_nodes, w/o repetitions
    """
    feature_ids = []
    for i in item_nodes:
        feature_ids.extend(get_item_features(i, matrix_D))
    return [Feature(f) for f in list(set(feature_ids))]

def get_users(matrix_S, active_user, k=5):
    """
    Parameters
    ----------
    matrix_S: pd.dataframe
    active_user: User
    k: int 
        The number of nearest neighbours to return

    Returns
    -------
    list
        The list of the k-nearest neighbours to active user
    """
    print('hola')
    return 9
    

def check_rating(rating):
    """
    A function to generate user-friendly ratings to songs
    
    Parameters
    ----------
    rating: int

    Returns
    -------
    str
    """
    if (rating < 3):
        opinion = 'the song is awful! :('
    elif (3 <= rating < 5): 
        opinion = 'the song is pretty bad :/'
    elif (5 <= rating < 7): 
        opinion = 'the song is fine :)'
    elif (7 <= rating < 9): 
        opinion = "the song is very interesting ^^"
    elif (9 <= rating < 10):
        opinion = 'the song is brilliant :D '
    elif (rating == 10):
        opinion = 'the song is memorable <3'
    
    return opinion