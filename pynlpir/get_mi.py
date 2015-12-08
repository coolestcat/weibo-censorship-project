import csv
import sys
from collections import defaultdict
from sets import Set
import re
import math

#ignore that the count actually reflects all times the word appears 
#(should be number of messages where the word appears - assume usually a word appears in a message only once)

def sampleParsed():
	print "Sampling Parsed:"

	files = ["./../parsed/week" + str(i) + "parsed.csv" for i in range(1,53)]
	outfile = "uncensored_sample.csv"
	of = open(outfile, 'wb')
	writer = csv.writer(of, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
	fileCount = 0

	for fi in files:
		read = open(fi, 'rb')
		reader = csv.reader(read, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
		rowCount = 0
		for row in reader:
			if len(row) > 3:
				print "err"
				return
			if rowCount % 50 == 0: # sample every 50th message
				if row[1] == "0":
					writer.writerow([row[0], row[1], row[2]])

			rowCount += 1

		fileCount += 1
		print str(fileCount) + "\r"
		sys.stdout.flush()

def mutual_information(word, tpl, total_censored, total_uncensored):
	ret_sum = 0
	if tpl[0] == 0: #present in uncensored but not censored - probably useless
		return -1.0
	if tpl[1] == 0: # present in censored but not uncensored - probably useful
		return -2.0

	matrix = [[total_uncensored - tpl[1], total_censored - tpl[0]],[tpl[1], tpl[0]]]
	N = total_censored + total_uncensored
	for row in range(2):
		for column in range(2):
			x = ((N)*matrix[row][column])/float((matrix[row][0] + matrix[row][1])*(matrix[0][column] + matrix[1][column]))
			s = (matrix[row][column]/float(N)) * math.log(x, 2)
			ret_sum += s

	return ret_sum


def getMI():
	total_censored = 85855
	total_uncensored = 3987354
	# stop words
	stopwords = Set([])
	newf = open("stopwords.txt", 'rb')
	for line in newf:
		stopwords.add(line.rstrip())

	print "reading censored"

	infile = "./all_censored.csv"
	f = open(infile, 'rb')
	reader = csv.reader(f, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
	censored_dict = defaultdict(int)

	for row in reader:
		for w in row[2].split(" "):
			censored_dict[w] += 1

	print "reading uncensored"
	uncensored = "./uncensored_sample.csv"
	uc = open(uncensored, 'rb')
	reader = csv.reader(uc, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
	uncensored_dict = defaultdict(int)

	for row in reader:
		if len(row)>3:
			print "greater than 3"
		if row[1] == "0":
			for w in row[2].split(" "):
				uncensored_dict[w] += 1

	print "building matrices"
	tuple_dict = dict()

	#all of the words in censored_dict and uncensored_dict get put into tuple_dict (sometimes overwritten)
	for w in sorted(uncensored_dict, key=uncensored_dict.get, reverse=True):
		if not ((w in stopwords) or re.match("@.*", w) or re.match("http.*", w)):
			tuple_dict[w] = [censored_dict[w], uncensored_dict[w]]

	for w in sorted(censored_dict, key=censored_dict.get, reverse=True):
		if not ((w in stopwords) or re.match("@.*", w) or re.match("http.*", w)):
			tuple_dict[w] = [censored_dict[w], uncensored_dict[w]]

	print "writing mi's into dict"
	mi_dict = defaultdict(int)
	for key, value in tuple_dict.iteritems():
		mi_dict[key] = mutual_information(key, value, total_censored, total_uncensored)

	result_file = "mi_results.csv"
	out_result_file = open(result_file, 'wb')
	writer = csv.writer(out_result_file, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)

	print "writing mi dict into file"
	for w in sorted(mi_dict, key=mi_dict.get, reverse=True):
		writer.writerow([w, mi_dict[w]])

# sampleParsed()
getMI()