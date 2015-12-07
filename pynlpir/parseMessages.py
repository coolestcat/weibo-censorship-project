import csv
import sys
import pynlpir

def main(argv):
	csv.field_size_limit(sys.maxsize)

	pynlpir.open()
	#load sogou dictionaries

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

	#write out parsed words
	# files = ["week" + str(i) + ".csv" for i in range(4,53)]

	# for fi in files:
	fi = "week" + str(argv[0]) + ".csv"
	print fi
	# for fi in files:
	infile = "./../" + fi
	a = fi.split('.')
	outfile = "./../parsed/" + a[0] + "parsed.csv"

	f = open(infile, 'rb')
	of = open(outfile, 'wb')
	reader = csv.reader(f, delimiter=",")
	writer = csv.writer(of, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
	count = 0
	total = sum(1 for row in reader)
	print total
	f.seek(0)
	errors = 0
	unbounderrors = 0

	for row in reader:
		mid = row[0]
		message = row[6]
		censored = None
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

		if row[10]!="":
			censored = 1
		else:
			censored = 0

		writer.writerow([mid, censored, mString.encode("utf-8")])

		# progress
		if count%1000 == 0:
			print str(count) + "/" + str(total) + "\r",
			sys.stdout.flush()
		count += 1


	print "count: " + str(count)
	print "errors: " + str(errors)
	print "unbounderrors: " + str(unbounderrors)

if __name__ == "__main__":
   main(sys.argv[1:])
