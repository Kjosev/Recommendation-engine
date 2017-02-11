#This file generates the submission ALS.csv from the files item_features_ALS and user_features_ALS

from helpers_MF import *
from train_ALS import train_ALS
import os

def run_ALS():
	"""Create subimssion with predicitons of the ALS model"""

	print("Running ALS...")

	#load the positions of the predictions to generate
	path_dataset = "data/data_test.csv"
	positions= load_data(path_dataset)

	#if features do not exist, traint the model
	if not os.path.isfile("data/item_features_ALS.obj") or not os.path.isfile("data/user_features_ALS.obj"):
		train_ALS()

	#load the item features
	file=open("data/item_features_ALS.obj",'rb')
	item_features = pickle.load(file)
	file.close()

	#load the user features
	file=open("data/user_features_ALS.obj",'rb')
	user_features = pickle.load(file)
	file.close()

	#get the predictions based on the features
	predictions=np.dot(item_features.T,user_features)

	#create submission
	create_submission(predictions,positions,"ALS")

if __name__ == "__main__":
	run_ALS()