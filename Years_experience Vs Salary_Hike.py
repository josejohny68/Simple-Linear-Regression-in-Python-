# Building a simple Linear prediction model for salary Hike
# Importing the required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

salary=pd.read_csv("E:\\ExcelR\\Python codes and python datasets\\Assignments in Python\\Assignment 1 - Simple Linear Regression\\Salary_Data.csv")

# EDA
salary.columns
plt.hist(salary.YearsExperience)
plt.boxplot(salary.YearsExperience)
plt.plot(salary.YearsExperience,salary.Salary,"ro")
plt.hist(salary.Salary)
plt.boxplot(salary.Salary)

# Identifing coefficient of corellation

salary.corr()
salary.Salary.corr(salary.YearsExperience)
np.corrcoef(salary.YearsExperience,salary.Salary) # 97% corellated

# Building the model

import statsmodels.formula.api as smf

model1=smf.ols("Salary~YearsExperience",data=salary).fit()

model1.summary() # R squared is 95% and the p value<0.05

pred=model1.predict(salary.YearsExperience)

plt.scatter(x=salary["YearsExperience"],y=salary["Salary"],color="red");plt.plot(salary["YearsExperience"],pred,color="green")
