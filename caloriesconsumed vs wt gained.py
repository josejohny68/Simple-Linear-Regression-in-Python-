# Importing libraries required for Simple Linear Regression

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

# Reading the data set

calories_consumed=pd.read_csv("E:\\ExcelR\\Python codes and python datasets\\Assignments in Python\\Assignment 1 - Simple Linear Regression\\calories_consumed.csv")

calories_consumed.columns

# There is a space between the column names so we are not able to draw the graphs
cc=calories_consumed.rename(columns={"Weight gained (grams)":"WeightGained","Calories Consumed":"Caloriesconsumed"})

# Calculating the correlation between the variables- using graphs

plt.hist(cc.WeightGained)
plt.boxplot(cc.WeightGained)
plt.plot(cc.Caloriesconsumed,cc.WeightGained,"ro")
plt.hist(cc.Caloriesconsumed)
plt.boxplot(cc.Caloriesconsumed)

# Calculating the coefficient of correlation

cc.corr()
cc.WeightGained.corr(cc.Caloriesconsumed)
np.corrcoef(cc.Caloriesconsumed,cc.WeightGained) # Corelation is .94 which is strong correlation

import statsmodels.formula.api as smf

model1=smf.ols("WeightGained~Caloriesconsumed",data=cc).fit() 

type(model1)

model1.params

model1.summary() # P- Value of the coefficients and R squared value are good so we can stop our analysis here

# Coefficient values as a confidence 95%

model1.conf_int(0.95)

pred=model1.predict(cc)


plt.scatter(x=cc["Caloriesconsumed"],y=cc["WeightGained"],color='red');plt.plot(cc["Caloriesconsumed"],pred,color="Black")
