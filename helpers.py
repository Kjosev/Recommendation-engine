import csv
import numpy as np
import os

def read_txt(path):
	"""read text file from path."""
	with open(path, "r") as f:
		return f.read().splitlines()

def load_data(path_dataset):
	"""Load data in text format, one rating per line, as in the kaggle competition."""
	data = read_txt(path_dataset)[1:]
	return preprocess_data(data)

def preprocess_data(data):
	"""preprocessing the text data, conversion to numerical array format."""
	def deal_line(line):
		pos, rating = line.split(',')
		row, col = pos.split("_")
		row = row.replace("r", "")
		col = col.replace("c", "")
		return int(row), int(col), float(rating)

	data = [deal_line(line) for line in data]
	return data

def format_data(data):
	"""format the data as a dictionary"""
	ratings = {}
	for userId, movieId, rating in data:
		if not userId in ratings: 
			ratings[userId] = {}
		ratings[userId][movieId] = rating

	return ratings

def load_csv_data(data_path):
    """Loads data and returns y (ratings), tX (predictions) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=float, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    input_data = x[:, 2:]
    
    with open(data_path, "r") as f:
    	headers = [x.strip() for x in f.readline().split(",")[2:]]


    return y, input_data,headers

def create_dataset(path_methods,path_true_val,filename):
	"""take multiple submissions and merge them into one joint dataset"""

	methods = {}
	methods_arr = []

	for f in os.listdir(path_methods):
		methods[f] = format_data(load_data(path_methods + str(f)))
		methods_arr.append(f)
		
	true_values = format_data(load_data(path_true_val))

	result_f = open(filename,"w")

	result_f.write("Id,Prediction")

	for method in methods_arr:
		result_f.write("," + str(method))

	result_f.write("\n")

	first = next (iter (methods.values()))

	for r in first:
		for c in first[r]:
			result_f.write("r" + str(r) + "_c" + str(c) + ",")
			result_f.write(str(true_values[r][c]))
			
			for method in methods_arr:
				pred = methods[method][r][c]
				row = "," + str(pred)
				result_f.write(row)
			result_f.write("\n")

	result_f.close()

def calculate_averages(ratings):
	"""Calculate user,movie and global average"""

	userDic = ratings
	movieDic = invert_data(ratings)

	avgUser = {}
	avgMovie = {}
	avgGlobal = 0

	for userid in ratings:	
		avgUser[userid] = sum(list(userDic[userid].values())) / float(len(userDic[userid]))
		avgGlobal += avgUser[userid]

		for movieid in ratings[userid]:
			if not movieid in avgMovie:
				avgMovie[movieid] = sum(list(movieDic[movieid].values())) / float(len(movieDic[movieid]))

	avgGlobal /= float(len(userDic))

	return avgUser, avgMovie, avgGlobal

def create_submission(data, filename="submission.csv"):
	print("Creating submission " + str(filename))
	f = open(filename,"w")
	f.write("Id,Prediction\n")

	for user in data:
		for movie in data[user]:
			rating = data[user][movie]
			f.write('r{0}_c{1},{2}'.format(user,movie,rating) + "\n")
	f.close()

def invert_data(data):
	"""Invert a dictionary"""
	result={}
	for user in data:
		for movie in data[user]:
			result.setdefault(movie,{})
	  
		  	# Flip item and person
			result[movie][user]=data[user][movie]
	return result