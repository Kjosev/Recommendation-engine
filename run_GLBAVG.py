from helpers import *
from collaborative import *
import pickle

def run_GLBAVG():
	"""Create subimssion with predicitons of the global average"""

	print("Running GLBAVG...")
	
	TRAIN_PATH = "data/data_train.csv"
	TEST_PATH = "data/data_test.csv"

	#load the data
	ratings = format_data(load_data(TRAIN_PATH))
	predictions = format_data(load_data(TEST_PATH))

	#calculate averages
	avgUser, avgMovie, avgGlobal = calculate_averages(ratings)

	#set the prediction to the global average rating
	for userid in predictions:
		for movieid in predictions[userid]:
			score = avgGlobal
			predictions[userid][movieid] = score

	#create submission
	create_submission(predictions,"data/submissions/GLBAVG.csv")

if __name__ == "__main__":
	run_GLBAVG()