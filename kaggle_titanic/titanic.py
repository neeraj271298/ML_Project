#importing required library
import numpy as np
import pandas as pd

#read data from csv file
trainData = pd.read_csv('train.csv')
testData = pd.read_csv('test.csv')
Y_train = trainData.iloc[:,1].values
X_train = trainData.iloc[:,[2,4,5,6,7,9,11]].values
X_test = testData.iloc[:,[1,3,4,5,6,8,10]].values

# read result data
prediction = pd.read_csv('submission.csv')
pred = prediction.iloc[:,1].values


# import Imputer Class For Removing NaN Value By Mean Of Column
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

# Training Set Imputer
imputer = imputer.fit(X_train[:, 2:6])
X_train[:, 2:6] = imputer.transform(X_train[:, 2:6])
#Test Set Imputer
imputer = imputer.fit(X_train[:,2:6])
X_test[:,2:6] = imputer.transform(X_test[:,2:6])

# Convet Categorical Data 
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
Label_male = LabelEncoder()
Label_embarkeked = LabelEncoder()
one_embarkeked = OneHotEncoder(categorical_features=[6])

# Training Set Categorical Convert
X_train[:,1] = Label_male.fit_transform(X_train[:,1])
X_train[:,6]=Label_embarkeked.fit_transform(X_train[:,6])
X_train = one_embarkeked.fit_transform(X_train).toarray()
X_train = X_train[:,1:]

# Test Set Categorical Convert
X_test[:,1] = Label_male.transform(X_test[:,1])
one_embarkeke = OneHotEncoder(categorical_features=[6])
X_test[:,6]=Label_embarkeked.transform(X_test[:,6])
X_test = one_embarkeked.fit_transform(X_test).toarray()
X_test = X_test[:,1:]

# Scaling The Data 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
sc_test  =StandardScaler()
X_test = sc.fit_transform(X_test)


# Principal Component Analysis for Feature Extraction
from sklearn.decomposition import PCA
pca = PCA(n_components = None)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
varience = pca.explained_variance_ratio_
# varience Variable Shows the Extracted Features
pca1 = PCA(n_components = 5)
X_train = pca1.fit_transform(X_train)
X_test = pca1.transform(X_test)


#first Model
# Applying KNearestClassification Algorithm
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(algorithm='ball_tree', leaf_size=30, metric='minkowski',
                           metric_params=None, n_jobs=1, n_neighbors=25, p=2,
                           weights='uniform')
clf.fit(X_train,Y_train)
pred1 = clf.predict(X_test)

# confusion Matrix To Find The Accuracy 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(pred,pred1)

# K fold cross validation To Find Accuracy on Diffrent Sets Of Training Dataset
from sklearn.model_selection import cross_val_score
acc = cross_val_score(estimator = clf , X= X_train, y=Y_train ,cv =10)
mean_accuracy = acc.mean()


#Grid search For Finding The Optimal Value Of Parameters Of Model
from sklearn.model_selection import GridSearchCV
param = [{'n_neighbors' : [1,10,15,25,35,22],'algorithm':['kd_tree'],'weights':['distance']},
          {'n_neighbors' : [1,10,15,25,35,22],'algorithm':['ball_tree'],'weights':['uniform']}]
grid = GridSearchCV(estimator=clf,param_grid=param,cv=10,scoring='accuracy')
grid = grid.fit(X_train,Y_train)
bestS = grid.best_score_
bestP = grid.best_estimator_



# second model
# Applying Artificial Neural Network 

#import Library
from keras.models import Sequential
from keras.layers import Dense
# creating Layers Of ANN
clf_ann = Sequential()
clf_ann.add(Dense(output_dim=5,input_dim=5,activation='relu',init='uniform'))
clf_ann.add(Dense(output_dim = 5, init = 'uniform', activation = 'relu'))
clf_ann.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
clf_ann.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
clf_ann.fit(X_train,Y_train,batch_size=10,nb_epoch=100) # fit the model
clf_ann.evaluate(X_test,pred)                           # evaluate the model
pred_ann=clf_ann.predict(X_test)
pred_ann=pred_ann > 0.5
pred2 = [];
for i in range(418):
    if(pred_ann[i] == False):
        pred2.append(0)
    else:
        pred2.append(1)
cm_ann = confusion_matrix(pred,pred2)




# Result
# KNN model
        # Accuracy = 92.11 %
        # mean Accuracy = 81.26%
        
# ANN Model
        # Accuracy = 91.38 %
        





