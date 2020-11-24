import numpy as np
import pandas as pd 
from sklearn.neighbors import KNeighborsClassifier

train_data = pd.read_csv("traindata.txt",sep='\s',engine='python')
train_label = pd.read_csv("trainlabel.txt",sep='\s',engine='python')
test_data = pd.read_csv("testdata.txt",sep='\s',engine='python')

knn = KNeighborsClassifier()
knn.fit(train_data,train_label)

predict_label = knn.predict(test_data)
print(predict_label)

