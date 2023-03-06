#!/usr/bin/env python
# coding: utf-8

# # TASK 1 (Prediction using Supervised ML)

# # By: Farah Bassem Mahmoud Ghanima.

# Simple linear regression to predict the percentage of a student based on the no. of study hours.

# In[123]:


#Importing required libraries

import pandas as pd
import matplotlib.pyplot as plt 


# In[124]:


#Reading the data

data= pd.read_csv('E:\\task1.csv')


# In[125]:


data.head()


# In[126]:


#Shape of the dataset

data.shape


# In[127]:


#Checking for Nullvalues in each column

data.isnull().sum()


# There are no null values in the data.

# In[128]:


# Plotting the distribution of scores

data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# In[129]:


#Dividing the data into independent and dependent variables

X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values  


#Splitting the data into train and test sets

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size = 0.3, random_state=0)


# In[130]:


#Shape of each set

X_train.shape, y_train.shape, X_test.shape, y_test.shape


# # Training the model

# In[131]:


#Importing Linear Regression model from scikit learn
from sklearn.linear_model import LinearRegression

#Fit the model
linear_reg = LinearRegression()
model = linear_reg.fit(X_train,y_train)


# In[132]:


#Regression line(y=mx+b)
line = model.coef_ * X + model.intercept_


# Plotting the actual data and the regression line
plt.scatter(X, y,color="Black")
plt.plot(X, line,color="Red");
plt.show()


# # Predicting the scores

# In[133]:


print("The predicted y-values of the test data are:")
y_pred=model.predict(X_test)
y_pred


# In[134]:


#Comparing the actual marks with the predicted marks in a dataframe

comparing=pd.DataFrame({'Actual': y_test ,'Predicted': y_pred})
comparing


# In[135]:


# Visualizing the comparison between the actual marks(points) and the predicted marks(line)

plt.scatter(X_test,y_test , color="black")
plt.plot(X_test,y_pred , color="orange")


# In[136]:


#What will be predicted score if a student studies for 9.25 hrs/ day?

hours = 9.25
answer=model.predict([[hours]])
print("Predicted mark for",hours,"studying hours is:",answer)


# # Model Evaluation

# In[137]:


#Evaluating the model using mean squared error

from sklearn.metrics import mean_squared_error

mse=mean_squared_error(y_test,y_pred)
print("Mean squared error:",mse)

