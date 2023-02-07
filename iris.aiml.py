##iris flower classification##

##Iris flower classification is a very popular machine learning project.
##Considering the three classes of flowers, Versicolor, Setosa, Virginica to iris dataset.
##Taking the 4 attributes as, ‘Sepal length’, ‘Sepal width’, ‘Petal length’, ‘Petal width’.
##It is used to predict flowers based on their specific features.


##importing the builtin library
import pandas as pd
import numpy as np


##reading the dataset in the form of csv file
data=pd.read_csv(r"C:\files\AIML note\iris.csv")


##defines shape of the dataset.
data.shape


##defines size of the dataset.
data.size


##defines information about the dataset.
data.info()
data.describe()


##Taking the dependent and target variables as x and y.
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values


##Using MinMaxScaler to re-arrange the values in range of [0,1].
from sklearn.preprocessing import MinMaxScaler
s=MinMaxScaler()
x=s.fit_transform(x)


##Using the train_test_split, the algorithm divides data as training data and testing data.
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=5)


##Using the KNN algorithm.
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=1)


##Model by KNN algorithm.
model.fit(xtrain,ytrain)
ypred=model.predict(xtest)


##Using accuracy_score to know the accuracy of the prediction of given dataset.
from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,ypred)*100)


##Using confusion_matrix to know matrix between test values and predictied values.
from sklearn.metrics import confusion_matrix
print(confusion_matrix(ytest,ypred))


##The algorithm is trained and its prediction is done.
print(model.predict([[7.1,3.1,2.5,0.4]]))

