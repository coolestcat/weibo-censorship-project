import csv
import sys
import re
import math
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from sets import Set
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# provinces = [11, 12, 13, 14, 15, 21, 22, 23, 31, 32, 33, 34, 35, 36, 37, 41, 42, 43, 44, 45, 46, 50, 51, 52, 53, 54, 61, 62, 63, 64, 65, 71, 81, 82, 100, 400]
provinces = [11, 82, 54]
months = [1,2,3,4,5,6,7,8,9,10,11,12]
#mi_infile = "censored_word_frequencies.csv"
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
	if rowCount < 2000:
		feature_words.append(row[0])
	else:
		break
	rowCount += 1

print len(feature_words)
train_matrix = []
test_matrix = []

print "adding censored data"
infile = "./all_censored.csv"
f = open(infile, 'rb')
reader = csv.reader(f, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
censored_train_cutoff = 85802 - (85802/float(5))

rowCount = 0 
mCount = 0
tCount = 0
iCount = 0
addedCount = 0
posCount = 0
negCount = 0
retweetCount = 0
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
			iCount += 1
		else:
			feature_vector.append(0)

		#gender of the user
		if row[9]=="m":
			feature_vector.append(1)
			mCount += 1
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
			retweetCount += 1

		# total number of tweets of the user


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
		# 	tCount += 1
		# else:
		# 	feature_vector.append(0)

		#carries the class label - deleted afterwards
		feature_vector.append(1)
		if rowCount < censored_train_cutoff:
			train_matrix.append(feature_vector)
		else:
			test_matrix.append(feature_vector)

		addedCount += 1

	rowCount += 1
	# print str(rowCount) + "\r"
	# sys.stdout.flush()

print "mCount" + str(mCount)
print "tCount" + str(tCount)
print "iCount" + str(iCount)
print "addedCount" + str(addedCount)
print "retweetCount" + str(retweetCount)
print "posCount" + str(posCount)
print "negCount" + str(negCount)

print "adding uncensored data"
uncensored = "./full_uncensored_sample.csv"
uc = open(uncensored, 'rb')
reader = csv.reader(uc, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
uncensored_train_cutoff = 93631 - (93631/float(5))
rowCount = 0 
added = 0
retweetCount = 0
posCount = 0
negCount = 0

for row in reader:
	# if rowCount%52 == 0:
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

		for word in feature_words:
			if word in these_words:
				feature_vector.append(1)
			else:
				feature_vector.append(0)

		# if positives > negatives:
		# 	feature_vector.append(0)
		# 	posCount += 1
		# else:
		# 	feature_vector.append(1)
		# 	negCount += 1
		
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

		if row[5]=="1":
			feature_vector.append(1)
		else:
			feature_vector.append(0)

		if row[9]=="m":
			feature_vector.append(1)
		else:
			feature_vector.append(0)

		for province in provinces:
			if int(row[8])==province:
				feature_vector.append(1)
			else:
				feature_vector.append(0)

		if row[3]=="":
			feature_vector.append(0)
		else:
			feature_vector.append(1)
			retweetCount += 1

		# date = row[6].split('-')
		# for month in months:
		# 	if int(date[1])==month:
		# 		feature_vector.append(1)
		# 	else:
		# 		feature_vector.append(0)

		# if row[10]=="True":
		# 	feature_vector.append(1)
		# else:
		# 	feature_vector.append(0)

		feature_vector.append(0)
		if rowCount < uncensored_train_cutoff:
			train_matrix.append(feature_vector)
		else:
			test_matrix.append(feature_vector)

		added += 1

	rowCount += 1
	# print str(rowCount) + "\r"
	# sys.stdout.flush()

print "Added: " + str(added)
print "retweetCount" + str(retweetCount)
print "posCount" + str(posCount)
print "negCount" + str(negCount)

print "building train/test X,Y"
train_X = []
test_X = []
train_Y = []
test_Y = []

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


# print "Training SVC"
# clf = SVC(class_weight={0:0.45,1:0.55}, tol=0.001)
# clf.fit(train_X, train_Y)
# preds = clf.predict(test_X)

# if not (len(preds)==len(test_Y)):
# 	print "lengths don't match"

# correct = 0
# incorrect = 0
# correct_censored = 0
# incorrect_censored = 0
# correct_uncensored = 0
# incorrect_uncensored = 0

# for i in range(len(preds)):
# 	if preds[i] == test_Y[i]:
# 		correct += 1
# 	else:
# 		incorrect += 1

# 	if test_Y[i] == 1 and preds[i] == 1:
# 		correct_censored += 1
# 	if test_Y[i] == 1 and preds[i] == 0:
# 		incorrect_censored += 1
# 	if test_Y[i] == 0 and preds[i] == 0:
# 		correct_uncensored += 1
# 	if test_Y[i] == 0 and preds[i] == 1:
# 		incorrect_uncensored += 1

# print "Total Accuracy: " + str(correct/float(correct + incorrect))
# print "Censored Accuracy: " + str(correct_censored/float(correct_censored + incorrect_censored))
# print "Uncensored Accuracy: " + str(correct_uncensored/float(correct_uncensored + incorrect_uncensored))


print "Training Logistic Regression L1"
clf = LogisticRegression(C=0.1, penalty='l2', class_weight={0:0.45,1:0.55}, tol=0.001)
clf.fit(train_X, train_Y)
preds = clf.predict(test_X)

if not (len(preds)==len(test_Y)):
	print "lengths don't match"

correct = 0
incorrect = 0
correct_censored = 0
incorrect_censored = 0
correct_uncensored = 0
incorrect_uncensored = 0

for i in range(len(preds)):
	if preds[i] == test_Y[i]:
		correct += 1
	else:
		incorrect += 1

	if test_Y[i] == 1 and preds[i] == 1:
		correct_censored += 1
	if test_Y[i] == 1 and preds[i] == 0:
		incorrect_censored += 1
	if test_Y[i] == 0 and preds[i] == 0:
		correct_uncensored += 1
	if test_Y[i] == 0 and preds[i] == 1:
		incorrect_uncensored += 1

print "Total Accuracy: " + str(correct/float(correct + incorrect))
print "Censored Accuracy: " + str(correct_censored/float(correct_censored + incorrect_censored))
print "Uncensored Accuracy: " + str(correct_uncensored/float(correct_uncensored + incorrect_uncensored))

print "Training Logistic Regression L2"
clf = LogisticRegression(C=0.1, penalty='l2', class_weight={0:0.45,1:0.55}, tol=0.001)
clf.fit(train_X, train_Y)
preds = clf.predict(test_X)

if not (len(preds)==len(test_Y)):
	print "lengths don't match"

correct = 0
incorrect = 0
correct_censored = 0
incorrect_censored = 0
correct_uncensored = 0
incorrect_uncensored = 0

for i in range(len(preds)):
	if preds[i] == test_Y[i]:
		correct += 1
	else:
		incorrect += 1

	if test_Y[i] == 1 and preds[i] == 1:
		correct_censored += 1
	if test_Y[i] == 1 and preds[i] == 0:
		incorrect_censored += 1
	if test_Y[i] == 0 and preds[i] == 0:
		correct_uncensored += 1
	if test_Y[i] == 0 and preds[i] == 1:
		incorrect_uncensored += 1

print "Total Accuracy: " + str(correct/float(correct + incorrect))
print "Censored Accuracy: " + str(correct_censored/float(correct_censored + incorrect_censored))
print "Uncensored Accuracy: " + str(correct_uncensored/float(correct_uncensored + incorrect_uncensored))



# print "Training Bernoulli Naive Bayes"

# clf = BernoulliNB(alpha=1.0)
# clf.fit(train_X, train_Y)
# preds = clf.predict(test_X)

# correct = 0
# incorrect = 0
# correct_censored = 0
# incorrect_censored = 0
# correct_uncensored = 0
# incorrect_uncensored = 0

# for i in range(len(preds)):
# 	if preds[i] == test_Y[i]:
# 		correct += 1
# 	else:
# 		incorrect += 1

# 	if test_Y[i] == 1 and preds[i] == 1:
# 		correct_censored += 1
# 	if test_Y[i] == 1 and preds[i] == 0:
# 		incorrect_censored += 1
# 	if test_Y[i] == 0 and preds[i] == 0:
# 		correct_uncensored += 1
# 	if test_Y[i] == 0 and preds[i] == 1:
# 		incorrect_uncensored += 1

# print "Total Accuracy: " + str(correct/float(correct + incorrect))
# print "Censored Accuracy: " + str(correct_censored/float(correct_censored + incorrect_censored))
# print "Uncensored Accuracy: " + str(correct_uncensored/float(correct_uncensored + incorrect_uncensored))

# print "Training KNN"

# clf = KNeighborsClassifier(n_neighbors=5)
# clf.fit(train_X, train_Y)
# preds = clf.predict(test_X)

# correct = 0
# incorrect = 0
# correct_censored = 0
# incorrect_censored = 0
# correct_uncensored = 0
# incorrect_uncensored = 0

# for i in range(len(preds)):
# 	if preds[i] == test_Y[i]:
# 		correct += 1
# 	else:
# 		incorrect += 1

# 	if test_Y[i] == 1 and preds[i] == 1:
# 		correct_censored += 1
# 	if test_Y[i] == 1 and preds[i] == 0:
# 		incorrect_censored += 1
# 	if test_Y[i] == 0 and preds[i] == 0:
# 		correct_uncensored += 1
# 	if test_Y[i] == 0 and preds[i] == 1:
# 		incorrect_uncensored += 1

# print "Total Accuracy: " + str(correct/float(correct + incorrect))
# print "Censored Accuracy: " + str(correct_censored/float(correct_censored + incorrect_censored))
# print "Uncensored Accuracy: " + str(correct_uncensored/float(correct_uncensored + incorrect_uncensored))

# print "Training KNN"

# clf = KNeighborsClassifier(n_neighbors=7)
# clf.fit(train_X, train_Y)
# preds = clf.predict(test_X)

# correct = 0
# incorrect = 0
# correct_censored = 0
# incorrect_censored = 0
# correct_uncensored = 0
# incorrect_uncensored = 0

# for i in range(len(preds)):
# 	if preds[i] == test_Y[i]:
# 		correct += 1
# 	else:
# 		incorrect += 1

# 	if test_Y[i] == 1 and preds[i] == 1:
# 		correct_censored += 1
# 	if test_Y[i] == 1 and preds[i] == 0:
# 		incorrect_censored += 1
# 	if test_Y[i] == 0 and preds[i] == 0:
# 		correct_uncensored += 1
# 	if test_Y[i] == 0 and preds[i] == 1:
# 		incorrect_uncensored += 1

# print "Total Accuracy: " + str(correct/float(correct + incorrect))
# print "Censored Accuracy: " + str(correct_censored/float(correct_censored + incorrect_censored))
# print "Uncensored Accuracy: " + str(correct_uncensored/float(correct_uncensored + incorrect_uncensored))

# clf = KNeighborsClassifier(n_neighbors=10)
# clf.fit(train_X, train_Y)
# preds = clf.predict(test_X)

# correct = 0
# incorrect = 0
# correct_censored = 0
# incorrect_censored = 0
# correct_uncensored = 0
# incorrect_uncensored = 0

# for i in range(len(preds)):
# 	if preds[i] == test_Y[i]:
# 		correct += 1
# 	else:
# 		incorrect += 1

# 	if test_Y[i] == 1 and preds[i] == 1:
# 		correct_censored += 1
# 	if test_Y[i] == 1 and preds[i] == 0:
# 		incorrect_censored += 1
# 	if test_Y[i] == 0 and preds[i] == 0:
# 		correct_uncensored += 1
# 	if test_Y[i] == 0 and preds[i] == 1:
# 		incorrect_uncensored += 1

# print "Total Accuracy: " + str(correct/float(correct + incorrect))
# print "Censored Accuracy: " + str(correct_censored/float(correct_censored + incorrect_censored))
# print "Uncensored Accuracy: " + str(correct_uncensored/float(correct_uncensored + incorrect_uncensored))

# clf = KNeighborsClassifier(n_neighbors=20)
# clf.fit(train_X, train_Y)
# preds = clf.predict(test_X)

# correct = 0
# incorrect = 0
# correct_censored = 0
# incorrect_censored = 0
# correct_uncensored = 0
# incorrect_uncensored = 0

# for i in range(len(preds)):
# 	if preds[i] == test_Y[i]:
# 		correct += 1
# 	else:
# 		incorrect += 1

# 	if test_Y[i] == 1 and preds[i] == 1:
# 		correct_censored += 1
# 	if test_Y[i] == 1 and preds[i] == 0:
# 		incorrect_censored += 1
# 	if test_Y[i] == 0 and preds[i] == 0:
# 		correct_uncensored += 1
# 	if test_Y[i] == 0 and preds[i] == 1:
# 		incorrect_uncensored += 1

# print "Total Accuracy: " + str(correct/float(correct + incorrect))
# print "Censored Accuracy: " + str(correct_censored/float(correct_censored + incorrect_censored))
# print "Uncensored Accuracy: " + str(correct_uncensored/float(correct_uncensored + incorrect_uncensored))