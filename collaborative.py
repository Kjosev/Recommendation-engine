from math import sqrt
import numpy as np
# Returns the Pearson correlation coefficient for p1 and p2
def sim_pearson(prefs,p1,p2):
  # Get the list of mutually rated items
	si={}
	for item in prefs[p1]: 
		if item in prefs[p2]: si[item]=1

	# if there are no ratings in common, return 0
	if len(si)==0: return 0

	# Sum calculations
	n=len(si)

	# Sums of all the preferences
	sum1=sum([prefs[p1][it] for it in si])
	sum2=sum([prefs[p2][it] for it in si])

	# Sums of the squares
	sum1Sq=sum([pow(prefs[p1][it],2) for it in si])
	sum2Sq=sum([pow(prefs[p2][it],2) for it in si])	

	# Sum of the products
	pSum=sum([prefs[p1][it]*prefs[p2][it] for it in si])

	# Calculate r (Pearson score)
	num=pSum-(sum1*sum2/n)
	den=sqrt((sum1Sq-pow(sum1,2)/n)*(sum2Sq-pow(sum2,2)/n))
	if den==0: return 0

	r=num/den

	return r

def computeSim(ratings):
	print("Precomputing simularities")
	count = 1
	sim = {}
	for p1 in ratings:
		print(str(count) + "/" + str(len(ratings)))
		count += 1
		sim[p1] = {}
		for p2 in ratings:
			if p1 == p2: continue

			if p2 in sim: sim[p1][p2] = sim[p2][p1]
			else: 
				sim[p1][p2] = sim_pearson(ratings,p1,p2)
	return sim


def predictRating(ratings,movieDic,user,movie,userSim,avgUser):
	"""caclulate score based on similar users/items"""

	total = 0
	simSum = 0

	AvgU = avgUser[user]

	#look at only users that rated this movie
	for other in movieDic[movie]:
		if other==user: continue
		sim = userSim[user][other]

		#ignore scores of zero or lower
		if sim <= 0: continue
		AvgO = avgUser[other]

		#weighted average of scores
		total  += (ratings[other][movie] - AvgO) * sim
		simSum += sim

	if simSum > 0:
		prediction = AvgU + total/simSum
		prediction = min(prediction,5)
		prediction = max(prediction,1)
	else:
		prediction = -1

	return prediction