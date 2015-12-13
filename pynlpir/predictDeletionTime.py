import csv
import sys
import re
import math
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from sets import Set
from sklearn.feature_selection import RFECV
from sklearn import metrics
import random
import datetime
from sklearn import linear_model
from sklearn.svm import SVR

provinces = [11, 82, 54]
months = [1,2,3,4,5,6,7,8,9,10,11,12]

mi_infile = "mi_results.csv"
mi_read = open(mi_infile, 'rb')
reader = csv.reader(mi_read, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
rowCount = 0
feature_words = []

print "loading sentiment sets"
pos = open('pos.txt', 'rb')
neg = open('neg.txt', 'rb')
pos_set = Set([])
neg_set = Set([])

for line in pos:
	pos_set.add(line.rstrip())

for line in neg:
	neg_set.add(line.rstrip())

print "appending feature words"

for row in reader:
	if row[0]=="||":
		print rowCount
	if rowCount < 0:
		feature_words.append(row[0])
	else:
		break
	rowCount += 1

train_matrix = []
test_matrix = []

print "adding censored data"
infile = "./all_censored.csv"
f = open(infile, 'rb')
reader = csv.reader(f, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
censored_train_cutoff = 73567 - (73567/float(7))# /float(10)
userDict = defaultdict(int)
rowCount = 0
addedCount = 0
greaterThanOne = 0
numTrain = 0
numTest = 0

for row in reader:
	if rowCount < censored_train_cutoff:
		userDict[row[4]] += 1
	else:
		break
	rowCount += 1

f.seek(0)

rowCount = 0 
for row in reader:
	if row[0]!="mid":
		feature_vector = []

		positives = 0
		negatives = 0

		these_words = Set([])
		for w in row[2].split(" "):
			these_words.add(w)

			if w in pos_set:
				positives += 1
			
			if w in neg_set:
				negatives += 1

		# feature words (top x based on mi)
		for word in feature_words:
			if word in these_words:
				feature_vector.append(1)
			else:
				feature_vector.append(0)

		#positive/negative sentiment

		# if positives > negatives:
		# 	feature_vector.append(0)
		# 	posCount += 1
		# else:
		# 	feature_vector.append(1)
		# 	negCount += 1

		#pos/neg scores
		# total = positives + negatives
		# if total > 0:
		# 	feature_vector.append(positives/float(total))
		# 	feature_vector.append(negatives/float(total))
		# else:
		# 	feature_vector.append(0)
		# 	feature_vector.append(0)

		# these_words = defaultdict(int)
		# for w in row[2].split(" "):
		# 	these_words[w] += 1

		# for word in feature_words:
		# 	feature_vector.append(these_words[w])

		# if rowCount < 2:
		# 	for key, value in these_words.iteritems():
		# 		print str(key) + ": " + str(value)

		# tweet has an image or not
		if row[5]=="1":
			feature_vector.append(1)
		else:
			feature_vector.append(0)

		#gender of the user
		if row[9]=="m":
			feature_vector.append(1)
		else:
			feature_vector.append(0)

		#province of the user

		for province in provinces:
			if int(row[8])==province:
				feature_vector.append(1)
			else:
				feature_vector.append(0)

		#whether the message is a retweet

		if row[3]=="":
			feature_vector.append(0)
		else:
			feature_vector.append(1)

		# total number of tweets of the user that were censored
		feature_vector.append(userDict[row[4]])


		# feature_vector.append(len(these_words))
		# month of posting

		# date = row[6].split('-')
		# # print date
		# for month in months:
		# 	if int(date[1])==month:
		# 		feature_vector.append(1)
		# 	else:
		# 		feature_vector.append(0)

		# account verified or not

		# if row[10]=="True":
		# 	feature_vector.append(1)
		# else:
		# 	feature_vector.append(0)

		#carries the time target - to be deleted afterwards

		# feature_vector.append(1)

		# row[7]-row[6]
		#2012-12-29 13:18:28

		dateFormat = "%Y-%m-%d %H:%M:%S.%f"
		dateFormat2 = "%Y-%m-%d %H:%M:%S"

		try:	
			deletedTime = datetime.datetime.strptime(row[7], dateFormat)
		except:
			try:
				deletedTime = datetime.datetime.strptime(row[7], dateFormat2)
			except:
				# print "error1"
				continue

		try:	
			postedTime = datetime.datetime.strptime(row[6], dateFormat)
		except:
			try:
				postedTime = datetime.datetime.strptime(row[6], dateFormat2)
			except:
				# print "error2"
				continue

		delta = deletedTime - postedTime
		days = delta.total_seconds()/float(60*60*24)
		feature_vector.append(days)
		if days>1:
			# print "delta: " + str(days)
			greaterThanOne += 1

		if addedCount < censored_train_cutoff:
			train_matrix.append(feature_vector)
			numTrain += 1
		else:
			test_matrix.append(feature_vector)
			numTest += 1

		addedCount += 1

	rowCount += 1
	# print str(rowCount) + "\r"
	# sys.stdout.flush()


print "numTrain" + str(numTrain)
print "numTest" + str(numTest)
print "cutoff" + str(censored_train_cutoff)
print "addedCount" + str(addedCount)
print "greaterThanOne" + str(greaterThanOne)

print "building train/test X,Y"
train_X = []
test_X = []
train_Y = []
test_Y = []

random.shuffle(train_matrix)

for element in train_matrix:
	train_Y.append(element[len(element)-1])
	train_X.append(element[0:len(element)-1])

for element in test_matrix:
	test_Y.append(element[len(element)-1])
	test_X.append(element[0:len(element)-1])

#testing the matrices
print len(train_X[0])
print len(train_X[15])

print len(test_X[0])
print len(test_X[17])

print len(test_Y)
print len(train_Y)

print "Training Linear Regression with Regularization"
clf = linear_model.Ridge (alpha =10)
clf.fit(train_X, train_Y)
preds = clf.predict(test_X)

diffs = []

for i in range(len(preds)):
	if (i<50):
		print str(preds[i]) + " : " + str(test_Y[i])
	diffs.append(abs(preds[i] - test_Y[i]))

print "Average Absolute Error: " + str(np.mean(diffs))
print "Standard Deviation: " + str(np.std(diffs))

print "Training SVR"
clf = SVR()
clf.fit(train_X, train_Y)
preds = clf.predict(test_X)

diffs = []

for i in range(len(preds)):
	if (i<50):
		print str(preds[i]) + " : " + str(test_Y[i])
	diffs.append(abs(preds[i] - test_Y[i]))

print "Average Absolute Error: " + str(np.mean(diffs))
print "Standard Deviation: " + str(np.std(diffs))