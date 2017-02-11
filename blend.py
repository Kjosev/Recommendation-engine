from helpers import *
from math import sqrt
import pickle

def least_squares(y, tx):
    """calculate the least squares solution."""
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))    
    rmse = sqrt(np.average((y - np.dot(tx,w))*(y - np.dot(tx,w))))
    
    return rmse, w


def blend(TRAIN_PATH="data/joint_dataset.csv"):
	"""calculate coeffiecients for blending the different methods"""

	print("Creating joint dataset")
	create_dataset("data/methods/","data/data_train.csv",TRAIN_PATH)
	
	y, tX, methods = load_csv_data(TRAIN_PATH)

	print("Finding coefficients")
	trmse,weights = least_squares(y,tX)

	coefs = {}
	
	for i in range(len(methods)):
		coefs[methods[i]] = weights[i]

	print("Saving coefficinets")
	file = open("data/coefs.obj","wb")
	pickle.dump(coefs,file)
	file.close()

	return coefs

if __name__ == "__main__":
	blend()