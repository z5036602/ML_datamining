#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 09:46:11 2019

@author: joshualiu
"""
import numpy as np
import csv 

  
def data_extraction (file_name):
    
    with open(file_name,"r" ) as data_sheet:
        reader = csv.reader(data_sheet, delimiter=",")
        data = list(reader)

    total_matrix = np.array(data[1:],dtype = float);
    return total_matrix
    #result = numpy.array(x).astype("float")'

def normalization (data):
    datamax = data.max()
 
    datamin = data.min()
 
    data = (data - datamin)/(datamax - datamin)
    return data
    
def split_train_test(data):
    return data[0:190,:], data[190:,:]



def cost_function(theta,X,Y):
    m = len(Y)
    predications = X@ theta.T
    cost = (1/m)*np.sum(np.square(predications-Y))
    return cost

def univariate_linear_regression_GD(x,y,learning_rate,iterations):
    m = len(y)
    assert(x.shape == (m,1),'data can only have one dimension')
    theta = np.array([-1,-0.5])
    cost_history=np.zeros(iterations)
    X = np.c_[np.ones(m),x]
    #theta = np.array([theta_0,theta_1])
    #theta_mat = np.repeat(theta, repeats=m, axis=0)
    for i in range(0,iterations):
        predication = X@ theta.T
        error=predication-y
        
        theta = theta-(1/m)*learning_rate*(error@X)
        cost_history[i] = cost_function(theta,X,y)
        
        
    return theta,cost_history
    


def RMSE_eval(theta,x,y):
    m = len(y)
    X = np.c_[np.ones(m),x]
    predications = X@ theta.T
    cost = np.sqrt((1/m)*np.sum(np.square(predications-y)))
    
    return cost


   
    
    