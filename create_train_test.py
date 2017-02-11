import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp
from helpers_MF import *

def create_train_test(ratings):
    seed = 1
    k_fold = 10
    nnzero_data=ratings.nnz
    # split data in k fold
    k_indices = build_k_indices(nnzero_data, k_fold, seed)

    for k in range(5):
        train,test=get_train_test(ratings,k_indices,k)
        print("Creating file: train_" + str(k))
        create_file_from_data(ratings, train,"train_" + str(k))
        print("Creating file: test_" + str(k))
        create_file_from_data(ratings, test,"test_" + str(k))


path_dataset = "data/data_train.csv"
ratings = load_data(path_dataset)
create_train_test(ratings)
