#!/usr/bin/python3
import pandas as pd
import random
from pathlib import Path
import numpy as np
from edge import Edge
from node import Feature, Item, User
from heapq import nlargest
from scipy import stats
from itertools import chain

#  ********************** Preprocessing functions ********************** 

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
        List of three pd.DataFrame (one per file)
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
    music_data : pd.DataFrame

    user_data: pd.DataFrame

    rating_data:pd.DataFrame
    
    Returns
    -------
    list
        A list of three dictionaries to translate songs, users and features'
        names to indexes and viceversa
    """

    songs_dict = pd.Series(music_data.song_id.values, index = music_data.name).to_dict() 
    users_dict = pd.Series(user_data.user_id.values, index = user_data.name).to_dict()
    features = list(music_data.keys()[2:])
    keys = ["f_%i" % i for i in range(len(features))]
    features_dict = {k:v for (k, v) in zip(keys, features)}

    return [songs_dict, users_dict, features_dict]

def compute_scores(user_data, music_data, rating_data):
    """
    Parameters:
    -----------
    user_data: pd.DataFrame
    music_data: pd.DataFrame
    rating_data: pd.DataFrame

    Returns:
    --------
    pd.DataFrame
                Matrix S
    """
    user_ids = user_data.user_id.values.tolist()
    columns = ['user_id'] + music_data.song_id.values.tolist()
    matrix_S = pd.DataFrame(np.zeros((len(user_ids), len(columns))), columns=columns)
    matrix_S['user_id'] = user_ids
    for _, row in rating_data.iterrows():
        # Set each value in matrix_S to corresponding rating if exists, otherwise stays 0
        i = matrix_S[matrix_S['user_id'] == row['user_id']].index[0]
        matrix_S.at[i, row['song_id']] = row['rating']

    return matrix_S

# ********************** Topology functions ********************** 

def compute_similarity(au_row, u_row):
    """
    Computes similarity between two users according to Eq.(1)
    Parameters:
    -----------
    au_row: pd.DataFrame
                        Scores by active user 
    u_row: pd.DataFrame
                        Scores by another user

    Returns:
    --------
    double
    """
    is_common_score = [True if (i!=0 and j!=0) else False for i, j in zip(au_row.values.tolist(), u_row.values.tolist())]
    aux_active_user = au_row[is_common_score].values
    aux_user = u_row[is_common_score].values
   
    try:
        pc = np.corrcoef(aux_active_user, aux_user)[0][1]
    except ValueError:
        return 0    

    if np.isnan(pc):
        return 0 # The NaN, in this case, is interpreted as no correlation between the two variables. 
                 # The correlation describes how much one variable changes as the other variable changes. 
                 # That requires both variables to change.  

    i_a = np.count_nonzero(au_row, axis=0)
    i_a_u = sum(is_common_score)    
    sim = abs(pc) * (i_a_u / i_a)
    return sim


def select_user_and_song(matrix_S):
    """
    Selects a random user as active user and a song that hasn't rated as target song 
    Parameters:
    -----------
    matrix_S: pd.DataFrame

    Returns:
    --------
    tuple
         An active user-target song pair
    """
    while True:
        user_index = random.randint(0, len(matrix_S.axes[0]) - 1)
        song_index = random.randint(0, len(matrix_S.axes[1]) - 2)
        rating = matrix_S.iloc[user_index, song_index]
        if (rating ==0):
            break
    song = matrix_S.columns.values[1:][song_index]
    user = matrix_S['user_id'][user_index]
    return (user, song)

def get_column_tags(node, matrix):

    """
    Obtains list of features relevant to item node according to matrix
    (or list of songs relevant to user node according to matrix)

    Parameters
    ----------
    node: Node (Item or User)
    matrix: pd.DataFrame

    Returns
    -------
    list
        A list with the names of the features relevant to item
        or the songs rated by the user

    """
    # 1) Find row and drop unnecessary columns
    if isinstance(node, Item):
        row = matrix.loc[matrix['song_id'] == node.index].drop(['song_id'], axis=1)
    elif isinstance(node, User):
        row = matrix.loc[matrix['user_id'] == node.index].drop(['user_id'], axis=1)
    # 2) Return nonzero column names
    return row.columns[row.values.nonzero()[1]].tolist()

def get_edges(nodes, matrix, type):  

    """
    Parameters:
    -----------
    nodes: list
    matrix: pd.DataFrame
    type: str

    Returns:
    --------
    list
        A list of the edges to the nodes according to matrix w/o repetitions
    """       
    # for node in nodes:
    #     row = matrix_S.loc[matrix_S['user_id'] == active_user].squeeze()
    edges = []
    for n in nodes:
        ids = get_column_tags(n, matrix)
    # ids = list(set(chain.from_iterable(get_column_tags(n, matrix) for n in nodes)))
        if type == 'f-i':
            edges.extend([Edge(Feature(f), n) for f in ids])
        elif type == 'i-u':
            edges.extend([Edge(Item(f), n) for f in ids])
    return edges
    

def get_features(item_nodes, matrix_D):

    """
    Parameters:
    -----------
    item_nodes: list 
    matrix_D: pd.DataFrame

    Returns:
    --------
    list
        A list of all the features that are relevant to the 
        given list of item_nodes, w/o repetitions
    """
    if isinstance(item_nodes, list):
        feature_ids = list(set(chain.from_iterable(get_column_tags(i, matrix_D) for i in item_nodes)))
    else:
        i = item_nodes # just one item 
        feature_ids = get_column_tags(i, matrix_D)

    return [Feature(f) for f in feature_ids]

def get_users(active_user, matrix_S, k=5):
    """
    Parameters
    ----------
    active_user: User
    matrix_S: pd.DataFrame
    k: int 
        The number of nearest neighbours to return

    Returns
    -------
    list
        The list of the k-nearest neighbours to active user
    """
    similarities = {}
    au_row = matrix_S.loc[matrix_S['user_id'] == active_user].squeeze() # au == "active user"

    for row_index in matrix_S.index:
        row = matrix_S.loc[row_index]
        if not row.equals(au_row):
            sim = compute_similarity(au_row.drop('user_id'), row.drop('user_id'))
            if sim != 0:
                u = int(row['user_id'])
                similarities[u] = sim
    return [User(u) for u in sorted(similarities, key=similarities.get, reverse=True)[:k]]

def get_u_minus(user_nodes, target_song, matrix_S):
    """
    Returns list of users from user_nodes that haven't rated target song

    Parameters
    ----------
    user_nodes: list
    target_song: Item
    matrix_S: pd.DataFrame

    Returns
    -------
    list
        A list with the users in U-
    """
    # 1) Select only rows corresponding to user_nodes
    reduced_S = matrix_S.loc[matrix_S['user_id'].isin([u.index for u in user_nodes])]

    # 2) Return only users in U-
    u_indexes = reduced_S[reduced_S[target_song.index] == 0.0]['user_id'].tolist()
    return [User(u_) for u_ in u_indexes]

# =======
def get_user_items(user, matrix_S):   # function for getting the items of each user
    row = matrix_S.loc[matrix_S['user_id'] == user.index]
    return row.columns[row.values.nonzero()[1]].tolist()


def get_u_min(user_nodes, target_song, item_nodes, matrix_S):
    u_min= []
    u_min_edges = []
    for u in user_nodes:
        user_item_ids = get_user_items(u, matrix_S)

        # check if user belongs in U-:
        if target_song not in user_item_ids:
            u_min.append(u)
    return u_min
# >>>>>>> federico

# ********************** Inference Functions **********************     

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
