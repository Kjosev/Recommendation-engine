import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp
from helpers_MF import *
import pickle

#matrix factorization using SGD
""" PARAMETERS
    train:data (matrix DN)
    num_features: number of features (K)
    lambda_user: regularization parameter for user
    lambda_item: regularization parameter for item
    num_epochs : number of iteration through all the data
    gamma : step size
    
    RETURN VALUES
    user_features : matrix (KN)
    item_features : matrix (KD)
"""
def matrix_factorization_SGD(train, num_features, lambda_user, lambda_item, num_epochs, gamma):

    errors = [101, 100]
    # init matrix
    user_features, item_features = init_MF(train, num_features)

    # find the non-zero ratings indices
    nz_row, nz_col = train.nonzero()
    nz_train= list(zip(nz_row, nz_col))

    print("learn the matrix factorization using SGD...")

    it = 0
    while(it < num_epochs and errors[-1] < errors[-2] and errors[-1] == errors[-1]):

        it+=1

        # shuffle the training rating indices
        np.random.shuffle(nz_train)

        # decrease step size
        gamma /= 1.2

        item_features_temp = item_features
        user_features_temp = user_features

        for d, n in nz_train:

            stoch_grad=-train[d,n]+np.dot(item_features_temp[:,d].T,user_features_temp[:,n])

            stoch_grad_item = (stoch_grad)*user_features_temp[:,n]+lambda_item*item_features_temp[:,d]

            stoch_grad_user = (stoch_grad)*item_features_temp[:,d]+lambda_user*user_features_temp[:,n]

            item_features_temp[:,d] = item_features_temp[:,d]-gamma*stoch_grad_item.reshape(num_features,)
            user_features_temp[:,n] = user_features[:,n]-gamma*stoch_grad_user.reshape(num_features,)

        rmse = compute_error(train, user_features, item_features, nz_train)
        print("iter: {}, RMSE on training set: {}.".format(it, rmse))

        errors.append(rmse)
        if(errors[-1] < errors[-2] and errors[-1] == errors[-1]):
            item_features = item_features_temp
            user_features = user_features_temp

    return user_features, item_features

def train_SGD():
    """Train item and user features for ALS"""

    #load the data
    path_dataset = "data/data_train.csv"
    train = load_data(path_dataset)

    """matrix factorization by SGD."""

    #define parameters (optimal parameters from cross-validation)
    gamma = 0.12
    num_features = 25 
    lambda_user = 0.02 
    lambda_item = 0.24
    num_epochs = 100

    #run the factorization
    user_features, item_features = matrix_factorization_SGD(train, num_features, lambda_user, lambda_item, num_epochs, gamma)

    #save item_features_SGD
    file = open("data/item_features_SGD.obj","wb")
    pickle.dump(item_features,file)
    file.close()

    #save user_features_SGD
    file = open("data/user_features_SGD.obj","wb")
    pickle.dump(user_features,file)
    file.close()

if __name__ == "__main__":
    train_SGD()
