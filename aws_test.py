import csv

with open('csv_test.csv', 'wb') as csvfile:

	writer = csv.writer(csvfile)
	writer.writerow(['a', 'b', 'c'])

	data = [
	    ('1', '25', '1234567')
	]
	writer.writerows(data)