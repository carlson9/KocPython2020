import csv
import os

#Open a file stream and create a CSV writer object
os.getcwd()
os.chdir("KocPython2020/in-classMaterial/day5/")
f = open('test.csv', 'w')
my_writer = csv.writer(f)

for i in range(1, 100):
  my_writer.writerow([i, i-1])
  
f.flush()
f.close()

#The correct way!
with open('test1.csv', 'w') as f:
  my_writer = csv.writer(f)
  for i in range(1, 100):
    my_writer.writerow([i, i-1])
    
#How about with field names
with open('test_with_fields.csv', 'w') as f:
  my_writer = csv.DictWriter(f, fieldnames=("A", "B"))
  my_writer.writeheader()
  for i in range(1, 100):
    my_writer.writerow({"B":i, "A":i-1})

#Now lets read some things
with open('test1.csv', 'r') as f:
  print("Reading test1.csv")
  my_reader = csv.reader(f)
  for row in my_reader:
    print(row)

#Now lets read some things with field names
with open('test_with_fields.csv', 'r') as f:
  print("\nReading test_with_fields.csv")
  my_reader = csv.DictReader(f)
  for row in my_reader:
    print(row)
