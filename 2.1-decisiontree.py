# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 12:46:40 2021

@author: naitik
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

my_data = pd.read_csv("drug200.csv", delimiter=",")
my_data[0:5]

len(my_data)  # Gives rows of data
my_data.shape # size of the data (m,n)
my_data['Drug'].value_counts() # Gives drug types information

# pandas to numpy
X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
type(X)
X[0:5]

# Convert string or names into values in X data
from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1]) 

le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])

le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 

X[0:5]

# Output
y = my_data["Drug"]
y[0:5]


# Train test split

from sklearn.model_selection import train_test_split
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

# plot of Each traincategory versur output
plt.plot(X_trainset[:,0], y_trainset,  'bo')
plt.plot(X_trainset[:,1], y_trainset,  'bo')
plt.plot(X_trainset[:,2], y_trainset,  'bo')
plt.plot(X_trainset[:,3], y_trainset,  'bo')
plt.plot(X_trainset[:,4], y_trainset,  'bo')

# Other way
print('Shape of X training set {}'.format(X_trainset.shape),'&',' Size of Y training set {}'.format(y_trainset.shape))

print('Shape of X training set {}'.format(X_testset.shape),'&',' Size of Y training set {}'.format(y_testset.shape))
X_trainset[:,0]


#######
### Model

drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree # it shows the default parameters
drugTree.fit(X_trainset,y_trainset)

predTree = drugTree.predict(X_testset)

print (predTree [0:5])
print (y_testset [0:5])

# Accuracy
from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))


# visualize the tree
!conda install -c conda-forge pydotplus -y
!conda install -c conda-forge python-graphviz -y

from  io import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree
%matplotlib inline 

dot_data = StringIO()
filename = "drugtree.png"
featureNames = my_data.columns[0:5]
out=tree.export_graphviz(drugTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')