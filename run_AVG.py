from helpers import *

def run_AVG():
	"""Create subimssion with predicitons of smart average"""

	print("Running AVG...")

	TRAIN_PATH = "data/data_train.csv"
	TEST_PATH = "data/data_test.csv"

	#load the data
	ratings = format_data(load_data(TRAIN_PATH))
	predictions = format_data(load_data(TEST_PATH))

	#calculate the averages
	avgUser, avgMovie, avgGlobal = calculate_averages(ratings)

	#invert the dictionary
	movieDic = invert_data(ratings)

	#variance of the movie average ratings
	Va = np.var(list(avgMovie.values()))
	#average variance of the individual movie ratings
	Vb = np.average([np.var(list(movieDic[movie].values())) for movie in movieDic]) 

	#blending factor
	K = Vb / Va

	betterMeanMovie = {}

	#calculate the better mean for each movie by blending the individual and global means
	for movie in movieDic:
		observedSum = np.sum(list(movieDic[movie].values()))
		betterMeanMovie[movie] = (avgGlobal*K + observedSum) / (K + len(movieDic[movie]))

	#calculate average offset for each user and the global offset 
	avgOffset = {}
	for user in ratings:
		avgOffset[user] = np.average([(ratings[user][movie] - avgMovie[movie]) for movie in ratings[user]])
	avgGlobalOffset = np.average(list(avgOffset.values()))

	#variance of the user offsets
	offVa = np.var(list(avgOffset.values()))
	#average variance of the individual user offsets
	offVb = np.average([np.var([(ratings[user][movie] - avgMovie[movie]) for movie in ratings[user]]) for user in ratings] )

	#blending factor
	offK = offVb / offVa

	betterOffsetUser = {}

	#calculate the better offset for each user by blending the individual and global offsets
	for user in ratings:
		observedSum = np.sum([(ratings[user][movie] - avgMovie[movie]) for movie in ratings[user]])
		betterOffsetUser[user] = (avgGlobalOffset*offK + observedSum) / (offK + len(ratings[user]))

	#set the predictions to the mean + offset
	for userid in predictions:
		for movieid in predictions[userid]:
			score = betterMeanMovie[movieid] + betterOffsetUser[userid]
			#trim score between 1-5
			score = min(score, 5)
			score = max(score, 1)
			predictions[userid][movieid] = score

	#create submission
	create_submission(predictions,"data/submissions/AVG.csv")

if __name__ == "__main__":
	run_AVG()