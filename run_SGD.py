#This file generates the submission SGD.csv from the files item_features_SGD and user_features_SGD

from helpers_MF import *
from train_SGD import train_SGD
import os

def run_SGD():
	"""Create subimssion with predicitons of the SGD model"""

	print("Running SGD...")
	
	#load the positions of the predictions to generate
	path_dataset = "data/data_test.csv"
	positions= load_data(path_dataset)

	#if features do not exist, train the model
	if not os.path.isfile("data/item_features_SGD.obj") or not os.path.isfile("data/user_features_SGD.obj"):
		train_SGD()

	#load the item features
	file=open("data/item_features_SGD.obj",'rb')
	item_features = pickle.load(file)
	file.close()

	#load the user features
	file=open("data/user_features_SGD.obj",'rb')
	user_features = pickle.load(file)
	file.close()

	#get the predictions based on the features
	predictions=np.dot(item_features.T,user_features)

	#create submission
	create_submission(predictions,positions,"SGD")

if __name__ == "__main__":
	run_SGD()