#from sklearn.model_selection import train_test_split
#from sklearn.naive_bayes import GaussianNB
#from sklearn.metrics import accuracy_score
import csv 

csv_reader = csv.reader(open('Training_Dataset.csv'), delimiter= ' ', quotechar='|')
line_count = 0
for row in csv_reader:
    line_count += 1
    print (', '.join(row))