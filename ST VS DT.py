#  Delivery_time -> Predict delivery time using sorting time 

# Import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Reading the dataset

delivery_time= pd.read_csv("E:\\ExcelR\\Python codes and python datasets\\Assignments in Python\\Assignment 1 - Simple Linear Regression\\delivery_time.csv")

# EDA for the dataset
# The names of the columns are having space we need to remove that
delivery_time.columns

delivery_time=delivery_time.rename(columns={"Delivery Time":"DT","Sorting Time":"ST"})

# EDA

plt.hist(delivery_time.DT) # Do not follow a normal distribution
plt.boxplot(delivery_time.DT) # Right viscus more than left viscus
plt.hist(delivery_time.ST) # Do not follow normal distribution
plt.boxplot(delivery_time.ST)
plt.plot(delivery_time.ST,delivery_time.DT,"ro")# scatter plot

# Calculating coefficient of corellation

delivery_time.corr()
delivery_time.DT.corr(delivery_time.ST)
np.corrcoef(delivery_time.ST,delivery_time.DT) # 82.5% corellated which gives us the go ahead for building the model

# Building the model

import statsmodels.formula.api as smf

model1=smf.ols("DT~ST",data=delivery_time).fit()

model1.params

model1.summary() # R squared value is 68% 

# Building the second model

model2=smf.ols("np.log(DT)~np.log(ST)",data=delivery_time).fit()
model2.summary() # Rsquared value has increased to 77% which is ok

pred=model2.predict(delivery_time)

# plotting model2 to show the best fit line

plt.scatter(x=np.log(delivery_time["ST"]),y=np.log(delivery_time["DT"]),color="red");plt.plot(np.log(delivery_time.ST),pred,color="green")
