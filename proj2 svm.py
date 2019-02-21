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

from sklearn.svm import SVC

start1 =time.time()
start2 = time.process_time()
df = pd.read_csv('/Users/jiaxiaoyu/Downloads/data.csv')

df=df.drop(['id'],axis=1)
df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})

X = df.drop('diagnosis', axis=1).values
y = df['diagnosis'].values

#pre-processing
scaler = preprocessing.StandardScaler()
X1 = scaler.fit_transform(X)
X2 = preprocessing.normalize(X1, norm='l2')

#importing train_test_split
X_train, X_valtest, y_train, y_valtest = train_test_split(X2, y, test_size=0.2, random_state=0, stratify=y)
X_test, X_val, y_test, y_val = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=0, stratify=y_valtest)

# Setup arrays to store training and val accuracies
kernel = ['linear','poly','rbf','sigmoid']
accuracies = []
C=range(1,1002,50)
for k in kernel:
    for i in C:
        # Setup a Support Vector Classifier
        svm = SVC(gamma='scale',kernel=k,C=i)
        # Fit the model
        svm.fit(X_train, y_train)
        # Compute accuracy on the test set
        val_accuracy = svm.score(X_val, y_val)
        print("kernel=%s,C=%d,accuracy=%.2f%%" % (k,i,val_accuracy * 100))
        accuracies.append(val_accuracy)

a = int(np.argmax(accuracies))

print("kernel=%s and C=%d achieved highest accuracy of %.2f%% on validation data" % (kernel[a],C[a],accuracies[a] * 100))

#Setup a support vector classifier
new = SVC(gamma='scale',kernel=kernel[a],C=C[a])
#Fit the model
new.fit(X_train, y_train)
test_accuracy = new.score(X_test, y_test)
print("the accuracy is %.2f%% on test data" % (test_accuracy * 100))

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



