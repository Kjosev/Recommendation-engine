"""some functions for help."""

from itertools import groupby

import numpy as np
import scipy.sparse as sp
from time import localtime,strftime
import pickle
import os


def read_txt(path):
    """read text file from path."""
    with open(path, "r") as f:
        return f.read().splitlines()


def load_data(path_dataset):
    """Load data in text format, one rating per line, as in the kaggle competition."""
    data = read_txt(path_dataset)[1:]
    return preprocess_data(data)


def preprocess_data(data):
    """preprocessing the text data, conversion to numerical array format."""
    def deal_line(line):
        pos, rating = line.split(',')
        row, col = pos.split("_")
        row = row.replace("r", "")
        col = col.replace("c", "")
        return int(row), int(col), float(rating)

    def statistics(data):
        row = set([line[0] for line in data])
        col = set([line[1] for line in data])
        return min(row), max(row), min(col), max(col)

    # parse each line
    data = [deal_line(line) for line in data]

    # do statistics on the dataset.
    min_row, max_row, min_col, max_col = statistics(data)

    # build rating matrix.
    ratings = sp.lil_matrix((max_row, max_col))
    for row, col, rating in data:
        ratings[row - 1, col - 1] = rating
    return ratings


def group_by(data, index):
    """group list of list by a specific index."""
    sorted_data = sorted(data, key=lambda x: x[index])
    groupby_data = groupby(sorted_data, lambda x: x[index])
    return groupby_data


def build_index_groups(train):
    """build groups for nnz rows and cols."""
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))

    grouped_nz_train_byrow = group_by(nz_train, index=0)
    nz_row_colindices = [(g, np.array([v[1] for v in value]))
                         for g, value in grouped_nz_train_byrow]

    grouped_nz_train_bycol = group_by(nz_train, index=1)
    nz_col_rowindices = [(g, np.array([v[0] for v in value]))
                         for g, value in grouped_nz_train_bycol]
    return nz_train, nz_row_colindices, nz_col_rowindices

""" PARAMETERS
    user :user index
    movie: item index
    rating: value of the rating
"""
def create_row(user,movie,rating):
    return 'r{0}_c{1},{2}'.format(user,movie,rating) + "\n"


""" PARAMETERS
    data :(matrix DN)
    positions: indices of predicitions to generate
    submission_name: name of the file
"""
def create_submission(data, positions, submission_name):
    print("Creating submission " + submission_name)
    filepath = "data/submissions/"
    filename = filepath + submission_name + ".csv"

    f = open(filename,"w")
    f.write("Id,Prediction\n")
    nz_row,nz_col=np.nonzero(positions)
    for i in range(len(nz_row)):
        movie=nz_row[i]
        user=nz_col[i]
        rating = data[movie,user]
        f.write(create_row(movie+1,user+1,rating))
    f.close()

""" PARAMETERS
    positions: indices of predicitions to generate
    method : a string to identify the method
"""
def create_submission_from_features(positions,method):

    file=open("data/item_features_" + method + ".obj",'rb')
    item_features = pickle.load(file)
    file.close()

    file=open("data/user_features_" + method + ".obj",'rb')
    user_features = pickle.load(file)
    file.close()

    predictions=np.dot(item_features.T,user_features)
    create_submission(predictions,positions,"submission_"+method)


""" Initialization for SGD and ALS
    PARAMETERS
    train: data
    num_features : dimension K
    RETURN VALUES
    user_features: initialzed matrix (KN)
    item_features: initialized matrix(KD)
"""
def init_MF(train, num_features):
    """init the parameter for matrix factorization."""

    num_items, num_users = train.shape
    user_features = np.random.rand(num_features,num_users)/num_users
    user_features[0,:]=np.ones((num_users,))
    item_features = np.random.rand(num_features,num_items)/num_items
    item_features[0,:]=sp.csr_matrix.mean(train,axis=1).reshape(num_items,)

    return user_features, item_features

""" Compute RMSE
    PARAMETERS
    data
    user_features
    item_features
    nz: number of non zero elements in data
    RETURN VALUES
    rmse
"""
def compute_error(data, user_features, item_features, nz):
    """compute the loss (MSE) of the prediction of nonzero elements."""

    nz_row, nz_col = data.nonzero();
    approx = np.dot(item_features.T, user_features)
    mse = np.sum(np.square(data[nz_row, nz_col]-approx[nz_row, nz_col]))

    return(np.sqrt(mse/data.nnz))

""" Creates a file containing a matrix in the 'data/train_test' folder
    PARAMETERS
    data
    positions: the elements of the matrix that are to be saved
    file_name
"""
def create_file_from_data(data, positions, file_name):
    filepath = "data/train_test/"
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    filename = filepath + file_name + ".csv"

    f = open(filename,"w")
    f.write("Id,Prediction\n")
    nz_row,nz_col=np.nonzero(positions)

    for i in range(len(nz_row)):
        user=nz_col[i]
        movie=nz_row[i]
        rating = data[movie,user]
        f.write(create_row(movie+1,user+1,rating))

    f.close()

""" Builds the indeces to split a matrix
    PARAMETERS
    nnzero_data
    k_fold: number of folds
    seed
    RETURN VALUES
    k_indices: list of grouped indices
"""
def build_k_indices(nnzero_data, k_fold, seed):
    """build k indices for k-fold."""

    interval = int(nnzero_data / k_fold)

    np.random.seed(seed)
    indices = np.random.permutation(nnzero_data)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]

    return k_indices

def get_train_test(ratings,k_indices,k):
    """split the ratings to training data and test data."""
    [n_items, n_users] = np.shape(ratings)
    train = sp.lil_matrix((n_items, n_users))
    test = sp.lil_matrix((n_items, n_users))
    [row_ratings, col_ratings] = np.nonzero(ratings)
    n_ratings = len(row_ratings);

    n_test_ratings = len(k_indices[k])

    shuffled_row_ratings = row_ratings[k_indices[k]]
    shuffled_col_ratings = col_ratings[k_indices[k]]

    for i in range(n_test_ratings):
        test[shuffled_row_ratings[i],shuffled_col_ratings[i]]=ratings[shuffled_row_ratings[i], shuffled_col_ratings[i]]

    for l in range(len(k_indices)):
        if (l!=k):
            shuffled_row_ratings = row_ratings[k_indices[l]]
            shuffled_col_ratings = col_ratings[k_indices[l]]
            for i in range(n_test_ratings):
                train[shuffled_row_ratings[i],shuffled_col_ratings[i]]=ratings[shuffled_row_ratings[i], shuffled_col_ratings[i]]

    train = sp.csr_matrix(train)
    test = sp.csr_matrix(test)


    return  train, test
