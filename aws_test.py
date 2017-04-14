import csv

with open('csv_test.csv', 'w') as csvfile:

	writer = csv.writer(csvfile)
	writer.writerow([1, 2, 3])

	data = [(1,3,4)]
	writer.writerows(data)
