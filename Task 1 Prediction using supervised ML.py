#!/usr/bin/env python
# coding: utf-8

# # The percentage of an student based on the no. of study hours.

# Linear Regression involving two variables using Python.
# 
# Simple Linear Regression
# In this regression, we will predict the percentage of marks of students based on the numbers of study hours.

# In[2]:


#importing all the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as mt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[27]:


#reading the data
link='http://bit.ly/w-data'
data=pd.read_csv(link)
data.head(25)


# Now we will be ploting our data points in 2D graph to find any relationship between the data.

# In[26]:


data.plot(x="Hours", y="Scores", style="o")
mt.title("Study_Hours vs Percentage_Score")
mt.xlabel("Study_Hours")
mt.ylabel("Percentage_Score")
mt.show()


# From the above graph, we can conclude that there is positive relation between the number of hours studies and percentage of score.
# Now dividing the data into attributes and lables.

# In[20]:


X=data.iloc[:,:-1].values
y=data.iloc[:,1].values


# In the next step, we will use Scikit-Learn's built-in train_set_split() method to split our data into training and test sets.

# In[21]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# # Training the Algorithm 

# As we have splitted our data into training and testing sets, we will now train our algorithm.

# In[22]:


from sklearn.linear_model import LinearRegression
R1=LinearRegression()
R1.fit(X,y)
print("Training the model")


# In[23]:


#plotting regression line
line=R1.coef_*X+R1.intercept_
mt.scatter(X,y)
mt.plot(X,line)
mt.show()


# # Predictions

# In[24]:


#testing data
print(X_test)
y_pred=R1.predict(X_test)


# In[29]:


#comparing actual vs predicted
df=pd.DataFrame({"Actual":y_test, "Predicted":y_pred})
df


# In[30]:


print("Training Score:",R1.score(X_train,y_train))
print("Testing Score: ",R1.score(X_test,y_test))


# In[37]:


#plotting the actual and predicted value into bar graph
df.plot(kind="bar", figsize=(7,5))
mt.show()


# In[38]:


#predicting for 9.25 hrs per day
hours=9.25
test=np.array([hours])
test=test.reshape(-1,1)
pred_val=R1.predict(test)
print("No. of hours=", hours)
print("Predicted Score=", pred_val[0])


# # Evaluating the Model

# In[40]:


import numpy as np
from sklearn import metrics
print("Mean Absolute Error: ",metrics.mean_absolute_error(y_test,y_pred))
print("Mean Squared Error: ",metrics.mean_squared_error(y_test,y_pred))
print("Root Mean Sqaured Error: ",np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
print("Explained Variance Score: ",metrics.explained_variance_score(y_test,y_pred))


# The above step of calculating error is to evaluate the performance of algorithm. And its is important as it tells how it differently perform on different dataset.
