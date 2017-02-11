from helpers import *
from collaborative import *
import pickle

def run_CFU():
	"""Create subimssion with predicitons of User-Based collaborative filtering"""

	print("Running CFU...")
	
	TRAIN_PATH = "data/data_train.csv"
	TEST_PATH = "data/data_test.csv"

	#load the data
	ratings = format_data(load_data(TRAIN_PATH))
	predictions = format_data(load_data(TEST_PATH))

	#calculate the averages
	avgUser, avgMovie, avgGlobal = calculate_averages(ratings)

	#invert the dictionary
	movieDic = invert_data(ratings)

	#compute user similarities if not yet precomputed
	userSimName = "data/userSim.obj"
	
	if not os.path.isfile(userSimName):
		userSim = computeSim(ratings)

		filehandler = open(userSimName,"wb")
		pickle.dump(userSim,filehandler)
		filehandler.close()

	#load user similarities
	file = open(userSimName,'rb')
	userSim = pickle.load(file)
	file.close()

	#set the predicitons to User-Based CF score
	for userid in predictions:
		for movieid in predictions[userid]:
			score = predictRating(ratings,movieDic,userid,movieid,userSim,avgUser)
			predictions[userid][movieid] = score

	#create submission
	create_submission(predictions,"data/submissions/CFU.csv")

if __name__ == "__main__":
	run_CFU()