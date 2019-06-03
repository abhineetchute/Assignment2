#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 09:43:39 2019

@author: abhineet
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('winequality-red.csv', delimiter=';')

col = data.columns
mean = data.mean()
for i in col:
    if i == 'quality':
        continue
    data[i]=data[i].apply(lambda x:(x-mean[i])/(data[i].max()-data[i].min()))

data.insert(0, 'x0', 1)
print(data.head())

X_df = data.iloc[:,:-1]
y_df = data.iloc[:,12:]

m=len(y_df)
#print (m)

#plt.figure(figsize=(10,8))
#plt.plot(X_df, y_df)

alpha1 = 1
alpha2 = 0.003
alpha3 = 0.00003

noi = 500

X = np.array(X_df)
y = np.array(y_df)
theta = np.array([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]])

def cost_function(X, y, theta):
    
    ## number of training examples
    m = len(y) 
    
    ## Calculate the cost with the given parameters
    J_theta = (np.sum((X.dot(theta)-y)**2))/2/m
    
    return J_theta

J = cost_function(X, y, theta)
print(J)

def gradient_descent(X, y, theta, alpha, iterations):
    
    cost_update = [0] * iterations
    
    for i in range(iterations):
        #print(theta.shape,X.shape,"theta,X")
        hypothesis = X.dot(theta)
        #print(hypothesis.shape,"hypo")
        #print(y.shape)
        loss = hypothesis-y
        #print(X.shape, loss.shape,"X shape")
        #print(X.transpose().shape,"transpose")
        gradient = (X.transpose()).dot(loss)/m
        #print(gradient, "grad")
        theta1 = theta - alpha*gradient
        cost = cost_function(X, y, theta1)
        cost_update[i] = cost
        theta = theta1
        print(i,cost)
        plt.scatter(i,cost)
    return theta, cost_update

(t1, c1) = gradient_descent(X, y, theta , alpha1, noi)
#print("Theta Parameters")
#print(X,"X")
#print(t1,"t1")
#print(X.dot(t1), y)

#print("Cost Function for alpha = 0.0003")
#print(c1)
#
#(t2, c2) = gradient_descent(X, y, theta , alpha2, noi)
##print("Theta Parameters")
##print(t2)
#
#print("Cost Function for alpha = 0.003")
#print(c2)
#
#(t3, c3) = gradient_descent(X, y, theta , alpha3, noi)
##print("Theta Parameters")
##print(t3)
#
#print("Cost Function for alpha = 0.00003")

#r2 = r2_score(y, X.dot(theta))
#print(r2)

# mean squared error
mse = np.sum((y - X.dot(t1))**2)

# root mean squared error
# m is the number of training examples
rmse = np.sqrt(mse/m)
# sum of square of residuals
ssr = np.sum((y - X.dot(t1))**2)

#  total sum of squares
sst = np.sum((y - np.mean(y))**2)

# R2 score
r2 = 1 - (ssr/sst)
print(r2)
