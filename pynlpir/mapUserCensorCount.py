import csv
import sys
import re
import math
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from sets import Set

infile = "./all_censored.csv"
f = open(infile, 'rb')
reader = csv.reader(f, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
userDict = defaultdict(int)

for row in reader:
	userDict[row[4]] += 1

for key, value in userDict.iteritems():
	print str(key) + ":" + str(value)