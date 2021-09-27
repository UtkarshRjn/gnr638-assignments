import csv

file = open('200050147_2.csv')
type(file)
csvreader = csv.reader(file)

my_y = []
for row in csvreader:
	my_y.append(row[1])

file2 = open('label.csv')
type(file2)
csvreader = csv.reader(file2)

y = []
for row in csvreader:
	y.append(row[1])

count = 0
for i in range(100):
	if(y[i] == my_y[i]):
	 	count+=1

print(count)