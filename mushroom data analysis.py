#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sn


# In[3]:


df=pd.read_csv('mushrooms.csv')


# In[28]:


df.isnull().sum()


# In[30]:


any(df.duplicated())


# In[4]:


df.info()


# In[31]:


df.describe()


# In[5]:


from sklearn.preprocessing import LabelEncoder
L=LabelEncoder()


# In[6]:


for i in df:
    if df[i].dtypes=='object':
       df[i]=L.fit_transform(df[i])


# In[7]:


corr_values=df.corr()


# In[8]:


corr_values


# In[9]:


sn.heatmap(df.corr())


# In[10]:


df.head()
df.isnull().sum()


# In[18]:


from sklearn.ensemble import RandomForestClassifier
model3=RandomForestClassifier(max_depth=10,n_jobs=-1, min_samples_split=5)


# In[19]:


X=df.drop(['class'],axis=1)
y=df['class'] 


# In[20]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=100)


# In[21]:


model3.fit(X_train,y_train)


# In[22]:


model3.score(X_train,y_train)


# In[23]:


y_pred=model3.predict(X_test)


# In[24]:


model3.score(X_test,y_test)


# In[ ]:


predictions=accuracy_score(y_test,y_pred)


# In[ ]:


print(predictions)


# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report


# In[25]:


confusion_matrix(y_test,y_pred)


# In[26]:


result=classification_report(y_test,y_pred)


# In[27]:


print(result)


# In[ ]:




