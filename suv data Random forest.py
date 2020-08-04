#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df=pd.read_csv('suv_data.csv')


# In[4]:


df.info()


# In[9]:


df["Gender"]= df["Gender"].replace("Male",1).replace("Female",0)


# In[10]:


df.head()


# In[13]:


X=df.drop(["Purchased"],axis=1)
y=df.Purchased


# In[11]:


from sklearn.model_selection import train_test_split


# In[22]:


get_ipython().run_line_magic('pinfo2', 'RandomForestClassifier')


# In[14]:


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.40, random_state=0)


# In[23]:


from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=10, max_depth=10,min_samples_split=5)


# In[24]:


model.fit(X_train, y_train)


# In[25]:


model.score(X_train, y_train)


# In[26]:


pred=model.predict(X_test)


# In[21]:


pred


# In[27]:


actual=y_test
model.score(X_test,y_test)


# In[28]:


from sklearn import metrics
print("Accuracy=",metrics.accuracy_score(y_test,pred))


# In[30]:


from sklearn.metrics import mean_absolute_error,mean_squared_error
#from sklearn.metrics import mean_absolute_percentage_error


# In[31]:


mse=mean_squared_error(actual,pred)


# In[32]:


mse


# In[33]:


mae=mean_absolute_error(actual,pred)


# In[34]:


mae


# In[ ]:




