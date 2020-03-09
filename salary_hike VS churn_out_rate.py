# importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Reading the csv file
employee=pd.read_csv("E:\\ExcelR\\Python codes and python datasets\\Assignments in Python\\Assignment 1 - Simple Linear Regression\\emp_data.csv")

#EDA
employee.columns

plt.hist(employee.Salary_hike)# Not normally distributed
plt.boxplot(employee.Salary_hike)#Right skewed data

plt.hist(employee.Churn_out_rate)# Not normally distributed
plt.boxplot(employee.Churn_out_rate) # Right skewed data

plt.plot(employee.Salary_hike,employee.Churn_out_rate,"bo")


# Calculating the coefficient of Corellation 

employee.corr()
employee.Churn_out_rate.corr(employee.Salary_hike)
np.corrcoef(employee.Salary_hike,employee.Churn_out_rate) # Negatively correlated as 0.91


# Building the model

import statsmodels.formula.api as smf

model1=smf.ols("Churn_out_rate~Salary_hike",data=employee).fit()

model1.summary() #Rsquared value is 0.83

pred=model1.predict(employee)

#Plotting the best fit line
plt.scatter(x=employee["Salary_hike"],y=employee["Churn_out_rate"],color="blue");plt.plot(employee["Salary_hike"],pred,color="red")
