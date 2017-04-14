import csv


csvfile = open('csv_test.csv', 'wb')
writer = csv.writer(csvfile)
writer.writerow(['a', 'b', 'c'])

data = [
    ('1', '25', '1234567')
]
writer.writerows(data)

csvfile.close()

