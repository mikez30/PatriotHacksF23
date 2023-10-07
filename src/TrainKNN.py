from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from KNN import KNN
from RandomForest import RandomForest
import csv


# Load the CSV data into lists
data = []
target = []
with open('heart.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile)
    
    # Iterate 2216 times (assuming 2216 rows in the CSV file)
    for row in csv_reader:
        dataRow = []
        #print(row)
        # Assuming the features are in columns 1 to n-1 and the target is in the last column
        for value in row:
            try:
                # Try to convert the value to a float
                dataRow.append(float(value))
            except ValueError:
                # If conversion to float fails, it's not numeric
                pass
        target.append(int (row[-1]))
        
        #print(dataRow)
        
        data.append(dataRow)


#print(data)
#print(target)


# Convert to numpy arrays
X = np.array(data)
y = np.array(target)

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=1234)



def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


clf = KNN(k=5)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

#print(predictions)

acc = np.sum(predictions == y_test) / len(y_test)
print("KNN Accuracy Model: " + str(acc * 100))