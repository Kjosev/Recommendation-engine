import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp
import matplotlib.pyplot as plt
from helpers_MF import *
import pickle

""" PARAMETERS
    train:data (matrix DN)
    item_features: matrix (KD)
    lambda_user: regularization parameter for user
    nz_user_itemindices :  indices of non zero elements
    
    RETURN VALUES
    user_features : matrix (KN)
    
"""
def update_user_feature( train, item_features, lambda_user, nz_user_itemindices):
    """update user feature matrix."""
    K=np.shape(item_features)[0]
    [D,N]=np.shape(train)
    user_features=np.empty((K,N))
    
    #solve Ax=b
    for user, nz_item_indices  in nz_user_itemindices:
        good_item_features=item_features[:,nz_item_indices]
        
        A=np.dot(good_item_features,good_item_features.T)+lambda_user*np.identity(K)
        
        b=good_item_features*(train[nz_item_indices,user])
        
        user_features[:,user]=np.linalg.solve(A,b).reshape(K,)

    return user_features

""" PARAMETERS
    train:data (matrix DN)
    user_features: matrix (KN)
    lambda_item: regularization parameter for iteme
    nz_item_itemindices :  indices of non zero elements
    
    RETURN VALUES
    item_features : matrix (KD)
    
"""
def update_item_feature( train, user_features, lambda_item, nz_item_userindices):
    """update user feature matrix."""
    K=np.shape(user_features)[0]
    [D,N]=np.shape(train)
    item_features=np.empty((K,D))
    
    #solve Ax=b
    for item, nz_user_indices  in nz_item_userindices:
        good_user_features=user_features[:,nz_user_indices]
        
        A=np.dot(good_user_features,good_user_features.T)+lambda_item*np.identity(K)
        
        b=good_user_features*train[item,nz_user_indices].T
        
        item_features[:,item]=np.linalg.solve(A,b).reshape(K,)

    return item_features
    
""" PARAMETERS
    train:data (matrix DN)
    lambda_user: regularization parameter for user
    lambda_item: regularization parameter for item
    num_features: number of features (K)
    max_iter : max number of iterations
    
    RETURN VALUES
    user_features : matrix (KN)
    item_features : matrix (KD)
"""
def ALS(train,lambda_user,lambda_item,num_features,max_iter):
    """Alternating Least Squares (ALS) algorithm."""
    
    # define parameters
    stop_criterion = 1e-6
    change = 1
    error_list = [0]
    
    n_iter=0
    # set seed
    np.random.seed(988)

    # init ALS
    user_features, item_features = init_MF(train, num_features)
    
    nz_row, nz_col = train.nonzero()
    nz_test = list(zip(nz_row, nz_col))
 
    nz_train, nz_row_colindices, nz_col_rowindices=build_index_groups(train)
    print("learn the matrix factorization using ALS...")

    while(n_iter<max_iter and change>stop_criterion):
        
        user_features=update_user_feature( train, item_features, lambda_user, nz_col_rowindices)
        item_features=update_item_feature( train, user_features, lambda_item, nz_row_colindices)
    
        rmse = compute_error(train, user_features, item_features, nz_train)
        print("iter: {}, RMSE on training set: {}.".format(n_iter, rmse))
        
        n_iter=n_iter+1
      
        if(n_iter>=2): change=abs(rmse-error_list[-1])
        error_list.append(rmse)
   
    return user_features,item_features

def train_ALS():
    """Train item and user features for ALS"""

    #load data
    path_dataset = "data/data_train.csv"
    ratings = load_data(path_dataset)

    #define parameters (optimal parameters from cross-validation)
    num_features=10
    lambda_user=31.8553
    lambda_item=20.05672522
    max_iter=30

    #run ALS
    user_features,item_features=ALS(ratings,lambda_user,lambda_item,num_features,max_iter)


    #save item_features_SGD
    file = open("data/item_features_ALS.obj","wb")
    pickle.dump(item_features,file)
    file.close()

    #save user_features_SGD
    file = open("data/user_features_ALS.obj","wb")
    pickle.dump(user_features,file)
    file.close()

if __name__ == "__main__":
    train_ALS()