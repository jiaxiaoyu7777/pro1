# import the necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
import time



start1 = time.time()
start2 = time.process_time()
df = pd.read_csv('/Users/jiaxiaoyu/Downloads/all/train.csv')

#Let's create numpy arrays for features and target
X = df.drop('label', axis=1).values
y = df['label'].values

#pre-processing
X1 = preprocessing.normalize(X, norm='l2')

#importing train_test_split
X_train, X_valtest, y_train, y_valtest = train_test_split(X1, y, test_size=0.2, random_state=0, stratify=y)
X_test, X_val, y_test, y_val = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=0, stratify=y_valtest)

# Setup arrays to store training and val accuracies
neighbors = range(1, 6, 1)
accuracies = []
precision_8 = np.empty(len(neighbors))
recall_8 =np.empty(len(neighbors))
f1_8 = np.empty(len(neighbors))
for i, k in enumerate(neighbors):
    # Setup a knn classifier with k neighbors
    knn = KNeighborsClassifier(n_neighbors=k)
    # Fit the model
    knn.fit(X_train, y_train)
    # Compute accuracy on the test set
    val_accuracy = knn.score(X_val, y_val)
    print("k=%d, accuracy=%.2f%%" % (k, val_accuracy * 100))
    accuracies.append(val_accuracy)
    y_pred = knn.predict(X_test)
    precision_8[i] = metrics.precision_score(y_test, y_pred, labels=[8], average=None)
    recall_8[i] = metrics.recall_score(y_test, y_pred, labels=[8], average=None)
    f1_8[i] = metrics.f1_score(y_test, y_pred, labels=[8], average=None)
a = int(np.argmax(accuracies))
print("k=%d achieved highest accuracy of %.2f%% on validation data" % (neighbors[a], accuracies[a] * 100))

plt.title('number "8" accuracy varying number of neighbors')
plt.plot(neighbors, precision_8, label='precision of number "8"')
plt.plot(neighbors, recall_8, label='recall of number "8"')
plt.plot(neighbors, f1_8, label='f1-score of number "8"')
# plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Precision/Recall/F1-score')
plt.show()

#Setup a knn classifier with k neighbors
new = KNeighborsClassifier(n_neighbors=neighbors[a])
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


