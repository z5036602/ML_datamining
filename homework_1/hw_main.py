#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 09:45:23 2019

@author: joshualiu
"""

import hw_mod
import matplotlib.pyplot as plt

total_matrix = hw_mod.data_extraction ("Advertising.csv")

data_matrix = total_matrix[:,1:]

data = hw_mod.normalization(data_matrix)

[train, test] = hw_mod.split_train_test(data)

x_TV_train = train[:,0]
x_Radio_train = train[:,1]
x_Newsp_train = train[:,2]
y = train[:,3]
x_TV_test = test[:,0]
x_Radio_test = test[:,1]
x_Newsp_test = test[:,2]
Y = test[:,3]


[theta,cost] = hw_mod.univariate_linear_regression_GD(x_TV_train,y,0.01,500)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(cost,label='Cost')
ax.title.set_text('Cost v.s Op iterations')
plt.xlabel('iteration')
plt.ylabel('cost')
# Add a legend
ax.legend()

# Show the plot
plt.show()

[theta_Radio,cost] = hw_mod.univariate_linear_regression_GD(x_Radio_train,y,0.01,500)
[theta_Newsp,cost] = hw_mod.univariate_linear_regression_GD(x_Newsp_train,y,0.01,500)

eval_metrics_TV_train = hw_mod.RMSE_eval(theta,x_TV_train,y)
print("TV Trained theta", theta, \
      "\nRMSE for training_set when using TV feature is", eval_metrics_TV_train)

eval_metrics_TV_test = hw_mod.RMSE_eval(theta,x_TV_test,Y)
print("TV Trained theta", theta, \
      "\nRMSE for test_set when using TV feature is", eval_metrics_TV_test)

eval_metrics_Radio = hw_mod.RMSE_eval(theta_Radio,x_Radio_test,Y)
print("Radio Trained theta", theta_Radio, \
      "\nRMSE for test_set when using Radio feature is", eval_metrics_Radio)




eval_metrics_Newsp = hw_mod.RMSE_eval(theta_Newsp,x_Newsp_test,Y)
print("Newsp Trained theta", theta_Newsp, \
      "\nRMSE for test_set when using Newspaper feature is", eval_metrics_Newsp)




