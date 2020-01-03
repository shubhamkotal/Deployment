#!/usr/bin/env python
# coding: utf-8

# In[46]:


import pandas as pd
import numpy as np
import os
os.chdir('E:\\Land Pricing')


# In[47]:


data=pd.read_csv('Land_Price.csv')


# In[48]:


data


# In[49]:


X = data.iloc[:, :2]
y = data.iloc[:, -1]


# In[50]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X, y)


# In[51]:


pickle.dump(regressor, open('model.pkl','wb'))


# In[54]:


# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[500072, 1350]]))


# In[ ]:




