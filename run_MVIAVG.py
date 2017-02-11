from helpers import *
from collaborative import *
import pickle

def run_MVIAVG():
	"""Create subimssion with predicitons of the movie's average"""

	print("Running MVIAVG...")
	
	TRAIN_PATH = "data/data_train.csv"
	TEST_PATH = "data/data_test.csv"

	#load the data
	ratings = format_data(load_data(TRAIN_PATH))
	predictions = format_data(load_data(TEST_PATH))

	#calculate the averages
	avgUser, avgMovie, avgGlobal = calculate_averages(ratings)

	#set the predictions to the average of the movie ratings
	for userid in predictions:
		for movieid in predictions[userid]:
			score = avgMovie[movieid]
			predictions[userid][movieid] = score

	#create submission
	create_submission(predictions,"data/submissions/MVIAVG.csv")

if __name__ == "__main__":
	run_MVIAVG()