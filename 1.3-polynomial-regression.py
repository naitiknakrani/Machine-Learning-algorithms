# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 17:42:30 2021

@author: naitik
"""

import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
%matplotlib inline

df = pd.read_csv("FuelConsumptionCo2.csv")

df.head() # show data

df.describe()  #summarize data

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']] # select few samples

cdf.iloc[5:7,:] # select specific row and column
cdf.head(9) # give first 9 rows data

# Training starts from here
# Let's split our dataset into train and test sets. 
# 80% of the entire dataset will be used for training and 20% for testing. 
# We create a mask to select random rows using np.random.rand() function:

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]


# Build a Model of Linear Regression
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

poly = PolynomialFeatures(degree=2)
train_x_poly = poly.fit_transform(train_x) # convert Train_x into 1 x x^2
train_x_poly

# Now we can use same linear model for polynomial data
regr = linear_model.LinearRegression()
regr.fit (train_x_poly, train_y)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)

# plot the data and non-linear regression line
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
XX = np.arange(0.0, 10.0, 0.1) # generate sample sequence like for XX=0:0.1:10
yy = regr.intercept_[0]+ regr.coef_[0][1]*XX+ regr.coef_[0][2]*np.power(XX, 2)
plt.plot(XX, yy, '-r' )
plt.xlabel("Engine size")
plt.ylabel("Emission")

#Evaluation matrix
from sklearn.metrics import r2_score

test_x_poly=poly.fit_transform(test_x)
test_y_ = regr.predict(test_x_poly)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , test_y_) )
print('Variance score: %.2f' % regr.score(test_x_poly, test_y))


# regression with cubic polynomial

poly = PolynomialFeatures(degree=3)
train_x_poly = poly.fit_transform(train_x) # convert Train_x into 1 x x^2
train_x_poly

# Now we can use same linear model for polynomial data
regr = linear_model.LinearRegression()
regr.fit (train_x_poly, train_y)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)

# plot the data and non-linear regression line
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
XX = np.arange(0.0, 10.0, 0.1) # generate sample sequence like for XX=0:0.1:10
yy = regr.intercept_[0]+ regr.coef_[0][1]*XX+ regr.coef_[0][2]*np.power(XX, 2)+regr.coef_[0][3]*np.power(XX, 3)
plt.plot(XX, yy, '-r' )
plt.xlabel("Engine size")
plt.ylabel("Emission")

#Evaluation matrix
from sklearn.metrics import r2_score

test_x_poly=poly.fit_transform(test_x)
test_y_ = regr.predict(test_x_poly)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , test_y_) )
print('Variance score: %.2f' % regr.score(test_x_poly, test_y))