import numpy as np
import pandas as pd
     

iris_data = pd.read_excel('C:/Users/hp/Desktop/python/iris .xls')

x = iris_data.drop(['Classification'],axis = 1)
y = iris_data['Classification']
     

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =  train_test_split(x,y,test_size=0.2, random_state=42)
     

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
model  = lr.fit(x_train,y_train)
lr_predictions = model.predict(x_test)

     

from sklearn.metrics import accuracy_score


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
     

model.fit(x_train, y_train)
     
KNeighborsClassifier()


model.score(x_test, y_test)
     


from sklearn.svm import SVC
svm_class = SVC(kernel = 'linear')
model = svm_class.fit(x_train,y_train)
svm_pred = model.predict(x_test)
     

print('Logistic regression Accuracy : ',accuracy_score(y_test,lr_predictions))
print('SVM linear Accuracy : ',accuracy_score(y_test,svm_pred))
print('KNN Accuracy : ',model.score(x_test, y_test))

import pickle
filename = 'iris.pkl'
pickle.dump(model, open(filename, 'wb'))
     

load_model = pickle.load(open(filename,'rb'))
     