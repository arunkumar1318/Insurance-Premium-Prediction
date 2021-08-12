#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing required libraries

import pandas as pd
import numpy as np

import pandasql as psql

import matplotlib.pyplot as plt

#importing warnings 

import warnings
warnings.filterwarnings("ignore")


# In[2]:


#loading the 'HealthIns' data set 

HealthIns=pd.read_csv(r"C:\Users\ARUN KUMAR\Downloads\archive\insurance.csv")
HealthIns.head()


# In[3]:


#checking the null values

HealthIns.isnull().sum()


# In[4]:


#displaying the information of dataset

HealthIns.info()


# In[5]:


#correlation of variables

HealthIns.corr()


# In[6]:


#describing the variables

HealthIns.describe()


# In[7]:


#displaying duplicate values

HealthIns[HealthIns.duplicated()]


# In[8]:


#droping the duplicate values

HealthIns=HealthIns.drop_duplicates()


# In[9]:


#initialising cols1 and cols2 variables 

cols1=['age','bmi']  #using normalisation
cols2=['sex', 'children', 'smoker', 'region']  #using dummies


# In[10]:


#using dummies on cols2

HealthIns=pd.get_dummies(HealthIns,columns=cols2)


# In[11]:


#displaying the values in transpose format

HealthIns.head().T


# In[12]:


#identifying the dependent and independent variables

x= HealthIns.drop(columns='expenses')
y=HealthIns[['expenses']]


# In[13]:


# Splitting the dataset into train and test 

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size = 0.30,random_state=7)


# In[14]:


# Scaling the features by using MinMaxScaler

from sklearn.preprocessing import MinMaxScaler

mmscaler = MinMaxScaler(feature_range=(0, 1))

x_train[cols1] = mmscaler.fit_transform(x_train[cols1])
x_train = pd.DataFrame(x_train)

x_test[cols1] = mmscaler.fit_transform(x_test[cols1])
x_test = pd.DataFrame(x_test)


# # LINEAR REGRESSION

# In[15]:


# Train the algorithm and build the model with train dataset

from sklearn.linear_model import LinearRegression

modelREG = LinearRegression()
modelREG.fit(x_train, y_train)

y_pred=modelREG.predict(x_test)

# Evaluation metrics for Regression analysis

from sklearn import metrics

print('Mean Absolute Error (MAE):', round(metrics.mean_absolute_error(y_test, y_pred),3))
print('Mean Squared Error (MSE):', round(metrics.mean_squared_error(y_test, y_pred),3))
print('Root Mean Squared Error (RMSE):', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),3))
print('Mean Absolute Percentage Error (MAPE):', round(metrics.mean_absolute_percentage_error(y_test, y_pred), 3) * 100, '%')
print('R2_score:', round(metrics.r2_score(y_test, y_pred),6))

# Calculate Adjusted R squared values

r_squared = round(metrics.r2_score(y_test, y_pred),3)
adjusted_r_squared = round(1 - (1-r_squared)*(len(y)-1)/(len(y)-x.shape[1]-1),3)
print('Adj R Square: ', adjusted_r_squared)


# # DecisionTreeRegressor

# In[16]:


# Build the model with RandomForestRegressor Regressor

from sklearn.tree import DecisionTreeRegressor

modelDT = DecisionTreeRegressor()
modelDT.fit(x_train,y_train)

# Predict the model with test dataset

y1_pred = modelDT.predict(x_test)

# Evaluation metrics for Regression analysis

from sklearn import metrics

print('Mean Absolute Error (MAE):', round(metrics.mean_absolute_error(y_test, y1_pred),3))
print('Mean Squared Error (MSE):', round(metrics.mean_squared_error(y_test, y1_pred),3))
print('Root Mean Squared Error (RMSE):', round(np.sqrt(metrics.mean_squared_error(y_test, y1_pred)),3))
print('Mean Absolute Percentage Error (MAPE):', round(metrics.mean_absolute_percentage_error(y_test, y1_pred), 3) * 100, '%')
print('R2_score:', round(metrics.r2_score(y_test, y1_pred),3))

# Calculate Adjusted R squared values

r_squared = round(metrics.r2_score(y_test, y1_pred),6)
adjusted_r_squared = round(1 - (1-r_squared)*(len(y)-1)/(len(y)-x.shape[1]-1),3)
print('Adj R Square: ', adjusted_r_squared)


# # RandomForestRegressor

# In[17]:


# Build the model with RandomForestRegressor Regressor

from sklearn.ensemble import RandomForestRegressor

modelRF = RandomForestRegressor()
modelRF.fit(x_train,y_train)

# Predict the model with test dataset

y2_pred = modelRF.predict(x_test)

# Evaluation metrics for Regression analysis

from sklearn import metrics

print('Mean Absolute Error (MAE):', round(metrics.mean_absolute_error(y_test, y2_pred),3))
print('Mean Squared Error (MSE):', round(metrics.mean_squared_error(y_test, y2_pred),3))
print('Root Mean Squared Error (RMSE):', round(np.sqrt(metrics.mean_squared_error(y_test, y2_pred)),3))
print('Mean Absolute Percentage Error (MAPE):', round(metrics.mean_absolute_percentage_error(y_test, y2_pred), 3) * 100, '%')
print('R2_score:', round(metrics.r2_score(y_test, y2_pred),3))

# Calculate Adjusted R squared values

r_squared = round(metrics.r2_score(y_test, y2_pred),6)
adjusted_r_squared = round(1 - (1-r_squared)*(len(y)-1)/(len(y)-x.shape[1]-1),3)
print('Adj R Square: ', adjusted_r_squared)


# # GradientBoostingRegressor

# In[18]:


# Build the model with Gradient Boosting Regressor

from sklearn.ensemble import GradientBoostingRegressor

modelGBR = GradientBoostingRegressor()
modelGBR.fit(x_train,y_train)

# Predict the model with test dataset

y3_pred = modelGBR.predict(x_test)

# Evaluation metrics for Regression analysis

from sklearn import metrics

print('Mean Absolute Error (MAE):', round(metrics.mean_absolute_error(y_test, y3_pred),3))
print('Mean Squared Error (MSE):', round(metrics.mean_squared_error(y_test, y3_pred),3))
print('Root Mean Squared Error (RMSE):', round(np.sqrt(metrics.mean_squared_error(y_test, y3_pred)),3))
print('Mean Absolute Percentage Error (MAPE):', round(metrics.mean_absolute_percentage_error(y_test, y3_pred), 3) * 100, '%')
print('R2_score:', round(metrics.r2_score(y_test, y3_pred),3))

# Calculate Adjusted R squared values

r_squared = round(metrics.r2_score(y_test, y3_pred),6)
adjusted_r_squared = round(1 - (1-r_squared)*(len(y)-1)/(len(y)-x.shape[1]-1),3)
print('Adj R Square: ', adjusted_r_squared)


# In[19]:


import pickle
# open a file, where you ant to store the data
file = open('random_forest_regression_model.pkl', 'wb')

# dump information to that file
pickle.dump(modelRF, file)


# In[ ]:




