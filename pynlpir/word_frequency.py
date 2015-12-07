import csv
import sys
from collections import defaultdict
from sets import Set
import re

csv.field_size_limit(sys.maxsize)

# infile = "./../parsed/week7parsed.csv"
infile = "./all_censored.csv"
f = open(infile, 'rb')
reader = csv.reader(f, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
d = defaultdict(int)


stopwords = Set([])
newf = open("stopwords.txt", 'rb')
for line in newf:
	stopwords.add(line.rstrip())

print "reading censored"

for row in reader:
	for w in row[2].split(" "):
		d[w] += 1

infile = "./../parsed/week1parsed.csv"
outfile = "./diff_word_frequencies.csv"
outfile2 = "./uncensored_word_frequencies.csv"
outfile3 = "./censored_word_frequencies.csv"
f = open(infile, 'rb')
of = open(outfile, 'wb')
of2 = open(outfile2, 'wb')
of3 = open(outfile3, 'wb')
reader = csv.reader(f, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
writer = csv.writer(of, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
writer2 = csv.writer(of2, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
writer3 = csv.writer(of3, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
d2 = defaultdict(int)

print "reading random parsed"
for row in reader:
	if row[1] == "0":
		for w in row[2].split(" "):
			d2[w] += 1

print "adding to all words set"
all_words_set = Set([])
for w in sorted(d2, key=d2.get, reverse=True):
	if not ((w in stopwords) or re.match("@.*", w) or re.match("http.*", w)):
		all_words_set.add(w)
		writer2.writerow([w, d2[w]])

print "writing frequencies"
for w in sorted(d, key=d.get, reverse=True):
	if (w in stopwords) or re.match("@.*", w) or re.match("http.*", w):
		print w
	else:
		writer3.writerow([w, d[w]])
		if not (w in all_words_set):
			writer.writerow([w, d[w]])

