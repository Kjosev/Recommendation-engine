from helpers import *
from collaborative import *
import pickle

def run_CFI():
	"""Create subimssion with predicitons of User-Based collaborative filtering"""

	print("Running CFI...")

	TRAIN_PATH = "data/data_train.csv"
	TEST_PATH = "data/data_test.csv"

	#load the data
	ratings = format_data(load_data(TRAIN_PATH))
	predictions = format_data(load_data(TEST_PATH))

	#calculate the averages
	avgUser, avgMovie, avgGlobal = calculate_averages(ratings)

	#invert the dictionary
	movieDic = invert_data(ratings)

	#computer movie similarities if not yet precomputed
	movieSimName = "data/movieSim.obj"

	if not os.path.isfile(movieSimName):
		movieSim = computeSim(movieDic)

		filehandler = open(movieSimName,"wb")
		pickle.dump(movieSim,filehandler)
		filehandler.close()

	#load the movie similarities
	file = open(movieSimName,'rb')
	movieSim = pickle.load(file)
	file.close()

	#set the predictions to the Item-Based CF score
	for userid in predictions:
		for movieid in predictions[userid]:
			scoreMovie = predictRating(movieDic,ratings,movieid,userid,movieSim,avgMovie)
			predictions[userid][movieid] = scoreMovie

	#create submission
	create_submission(predictions,"data/submissions/CFI.csv")

if __name__ == "__main__":
	run_CFI()