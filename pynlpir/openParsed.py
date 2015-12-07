import csv
import sys
import pynlpir

csv.field_size_limit(sys.maxsize)

# infile = "./../parsed/week7parsed.csv"
infile = "./all_censored.csv"
f = open(infile, 'rb')
reader = csv.reader(f, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
count = 0
total = sum(1 for row in reader)
print total
f.seek(0)
ccount = 0

for row in reader:
	if len(row)>3:
		print row[3]
	if row[1]=="1":
		# print row[2]
		ccount += 1

print ccount

