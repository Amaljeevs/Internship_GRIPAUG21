#!/usr/bin/env python
# coding: utf-8

# # THE SPARK FOUNDATION.
# 
# 
# ### Data Science and Buisness Analytics Intern GRIP(August 2021)![logo_small.png](attachment:logo_small.png)  
# 
# ## TASK 1 : Prediction Using Supervised Machine Learning
# 
# ### Author : Amal jeev S
# 
# ## Problem Statement
# 
#    > Predict the percentage of a student based on the number of study hours
#    
#    > What will be the predicted score if a student studies for 9.25 hrs?

# ### Import the libraries

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Loading the Data set

# In[6]:


url = 'https://bit.ly/w-data'
data = pd.read_csv(url)


# ### Exploratory Data Analysis

# In[7]:


# first five rows of dataframe
data.head()


# In[8]:


# concise summary of dataframe 
data.info()


# In[9]:


# descriptive statistics of data
data.describe()


# In[10]:


# checking correlation
data.corr(method= 'pearson')


# In[11]:


# checking for null values
data.isnull().sum()


# ### Visualizing Data

# In[12]:


# Plotting the distribution of scores
data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage', fontsize=16)  
plt.xlabel('Hours Studied',fontsize=12)  
plt.ylabel('Percentage Score', fontsize=12)  
plt.show()


# ### Preparing the data

# In[13]:


X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values


# ### Train Test Split

# In[14]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0)


# ### Training the Model

# In[15]:


# Linear Regression
from sklearn.linear_model import LinearRegression  

regressor = LinearRegression()  
regressor.fit(X_train, y_train)


# ### Plotting the regression line

# In[16]:


line = regressor.coef_ * X + regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.title('Hours vs Percentage', fontsize=16)  
plt.xlabel('Hours Studied',fontsize=12)  
plt.ylabel('Percentage Score', fontsize=12)  
plt.plot(X, line);
plt.show()


# ### Making Predictions

# In[17]:


# test data
print(X_test)


# In[18]:


# Comparing Actual vs Predicted

y_pred = regressor.predict(X_test)
y_pred = regressor.predict(X_test)
df = pd.DataFrame({ 'Actual_Score': y_test, 'Predicted_Score': y_pred})  
df


# ### Evaluating the Model

# In[19]:


from sklearn import metrics
from sklearn.metrics import r2_score
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print ('R2 Score:', r2_score(y_test,y_pred))


# ### What will be predicted score if a student studies for 9.25 hrs/ day?

# In[20]:


hours = 9.25
own_pred = regressor.predict([[hours]])
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))

