from helpers import *
import pickle 
import blend

from run_AVG import run_AVG
from run_SGD import run_SGD
from run_ALS import run_ALS
from run_CFI import run_CFI
from run_CFU import run_CFU
from run_GLBAVG import run_GLBAVG
from run_MVIAVG import run_MVIAVG
from run_USRAVG import run_USRAVG

def train(method):
	if(method == "AVG"):
		run_AVG()
	elif(method == "SGD"):
		run_SGD()
	elif(method == "ALS"):
		run_ALS()
	elif(method == "CFI"):
		run_CFI()
	elif(method == "CFU"):
		run_CFU()
	elif(method == "GLBAVG"):
		run_GLBAVG()
	elif(method == "MVIAVG"):
		run_MVIAVG()
	elif(method == "USRAVG"):
		run_USRAVG()


SUBMISSIONS_PATH = "data/submissions/"
TEST_PATH = "data/data_test.csv"

methods = ["AVG","SGD","ALS", "CFI","CFU","GLBAVG","MVIAVG","USRAVG"]

for method in methods:
	if not os.path.isfile(SUBMISSIONS_PATH + method + ".csv"):
		train(method)

predictions = {}

"""load submissions created with each method"""
for method in methods:
	predictions[method] = format_data(load_data(SUBMISSIONS_PATH + method + ".csv"))

test_data = format_data(load_data(TEST_PATH))

if not os.path.isfile("data/coefs.obj"):
	blend.blend()

"""load precomputed coefficient"""
file = open("data/coefs.obj",'rb')
coefs = pickle.load(file)
file.close()

print("Blending results with coefs:")
print(coefs)
"""blend submissions based on the coefficients"""
result = {}
for userid in test_data:
	result[userid] = {}
	for movieid in test_data[userid]:
		score = 0
		for method in methods:
			score += coefs[method] * predictions[method][userid][movieid]
		
		score = max(1,score)
		score = min(5, score)
		result[userid][movieid] = score

create_submission(result)