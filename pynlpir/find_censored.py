import csv
import sys
import pynlpir

csv.field_size_limit(sys.maxsize)

pynlpir.open()

n = pynlpir.nlpir.ImportUserDict("./../Scel2Txt/places_raw.txt")
print "num imported: " + str(n)

n = pynlpir.nlpir.ImportUserDict("./../Scel2Txt/pop_raw.txt")
print "num imported: " + str(n)

n = pynlpir.nlpir.ImportUserDict("./../Scel2Txt/poprecommend_raw.txt")
print "num imported: " + str(n)

n = pynlpir.nlpir.ImportUserDict("./../Scel2Txt/politics_raw.txt")
print "num imported: " + str(n)

n = pynlpir.nlpir.ImportUserDict("./../Scel2Txt/weibo-data/data/0.txt")
print "num imported: " + str(n)

n = pynlpir.nlpir.ImportUserDict("./../Scel2Txt/weibo-data/data/1.txt")
print "num imported: " + str(n)

n = pynlpir.nlpir.ImportUserDict("./../Scel2Txt/weibo-data/data/2.txt")
print "num imported: " + str(n)

n = pynlpir.nlpir.ImportUserDict("./../Scel2Txt/weibo-data/data/3.txt")
print "num imported: " + str(n)

n = pynlpir.nlpir.ImportUserDict("./../Scel2Txt/weibo-data/data/4.txt")
print "num imported: " + str(n)

files = ["week" + str(i) + ".csv" for i in range(1,53)]

total_count = 0
of = open("all_censored.csv", 'wb')
writer = csv.writer(of, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
errors = 0
unbounderrors = 0

for f in files:
	infile = "./../" + f
	with open(infile, 'rb') as csvfile:
		count = 0
		reader = csv.reader(csvfile, delimiter=",")
		for row in reader:
			if row[10]!="":
				mid = row[0]
				message = row[6]
				censored = 1
				try:
					segmented = pynlpir.segment(message)
				except UnicodeDecodeError:
					errors += 1
					continue
				except UnboundLocalError:
					unbounderrors += 1
					print "what??"
					continue
				except:
					print "core dump...?"
					continue

				mString = ""
				for segment in segmented:
					mString += segment[0]
					mString += " "

				writer.writerow([mid, censored, mString.encode("utf-8")])

				# progress

				print str(count) + "\r",
				sys.stdout.flush()

				count += 1

		print f + ": " + str(count)
		total_count += count

print "total: " + str(total_count)



# with open('week4.csv', 'rb') as csvfile:
# 	dcount = 0
# 	count = 0
# 	ccount = 0
# 	# reader = csv.reader(csvfile, delimiter=",", quotechar='|')
# 	reader = csv.reader(csvfile, delimiter=",")
# 	for row in reader:
# 		count += 1
		
# 		if row[9] != "":
# 			# print row[9]
# 			# print row[10]
# 			# print "---"
# 			dcount += 1

# 		if row[10] != "":
# 			ccount += 1
# 			print count

# 	print "Count: " + str(count)
# 	print "Deleted Count: " + str(dcount)
# 	print "Censored Count: " + str(ccount)
