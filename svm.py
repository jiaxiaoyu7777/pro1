# import the necessary packages
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

start1 =time.time()
start2 = time.process_time()
df = pd.read_csv('/Users/jiaxiaoyu/Downloads/diabetes.csv')
#Let's create numpy arrays for features and target
X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values

#pre-processing
scaler = preprocessing.StandardScaler()
X1 = scaler.fit_transform(X)
#X2 = preprocessing.normalize(X, norm='l2')

#importing train_test_split
X_train, X_valtest, y_train, y_valtest = train_test_split(X1, y, test_size=0.2, random_state=0, stratify=y)
X_test, X_val, y_test, y_val = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=0, stratify=y_valtest)

# Setup arrays to store training and val accuracies
degree = range(1, 10, 1)
accuracies = []
for k in range(1, 10, 1):
    # Setup a knn classifier with k neighbors
    svm = SVC(gamma='scale',kernel='poly',degree=k)
    # Fit the model
    svm.fit(X_train, y_train)
    # Compute accuracy on the test set
    val_accuracy = svm.score(X_val, y_val)
    print("k=%d, accuracy=%.2f%%" % (k, val_accuracy * 100))
    accuracies.append(val_accuracy)

a = int(np.argmax(accuracies))
print("k=%d achieved highest accuracy of %.2f%% on validation data" % (degree[a], accuracies[a] * 100))

#Setup a knn classifier with k neighbors
new = SVC(gamma='scale',kernel='poly',degree=degree[a])
#Fit the model
new.fit(X_train, y_train)
test_accuracy = new.score(X_test, y_test)
print("the accuracy is %.2f%% on test data" % (test_accuracy * 100))
#import confusion_matrix
y_pred = new.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))

end1 = time.time()
end2 = time.process_time()
clock = end1 - start1
cpu = end2-start2
print("wall-clock time is" + str(clock) + "s")
print("run-time is" + str(cpu) + "s")

#import classification_report
#print(classification_report(y_test,y_pred))


