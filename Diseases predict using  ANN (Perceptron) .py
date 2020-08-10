#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow


# In[2]:


df=pd.read_csv('diabetes.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe().transpose()


# In[6]:


sns.countplot(x='Outcome',data=df)


# In[7]:


# checking the correlation of all the features with respect to the labels with sort values and plot the values adding [-1] for removing the label
df.corr()['Outcome'].sort_values().plot(kind='bar')


# In[8]:


# making the heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr())


# In[9]:


X=df.drop('Outcome',axis=1).values
y=df['Outcome'].values


# In[10]:


from sklearn.model_selection import train_test_split


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=100)


# In[12]:


from sklearn.preprocessing import StandardScaler
scale=StandardScaler()
for i in df:
    if df[i].dtypes!='object':
        df[i]=scale.fit_transform(df[[i]])


# In[13]:


df.head()


# In[14]:


X


# In[15]:


print(X_train.shape)
print(X_test.shape)


# In[16]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout


# In[17]:


model=Sequential()
model.add(Dense(8,activation='relu'))
model.add(Dense(4,activation='relu'))
# Since this is a binary classification so we use 'sigmoid ' as activation function

model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam')


# In[18]:


model.fit(x=X_train,y=y_train,epochs=200,validation_data=(X_test,y_test))


# In[19]:


losses=pd.DataFrame(model.history.history)


# In[20]:


losses.plot()


# In[21]:


# when the model fails to generalise or facing the overfitting problem then we can use call back function
#model=Sequential()
#model.add(Dense(8,activation='relu'))
#model.add(Dense(4,activation='relu'))
# Since this is a binary classification so we use 'sigmoid ' as activation function

#model.add(Dense(1,activation='sigmoid'))
#model.compile(loss='binary_crossentropy',optimizer='adam')
#from tensorflow.keras.callbacks import EarlyStopping
#earlystop=EarlyStopping(monitor='val_loss',verbose=1,mode='min',patience=25)
#model.fit(x=X_train,y=y_train,epochs=500,validation_data=(X_test,y_test),callbacks=[earlystop])

#after running this command it automatically stop when model shows overfit, so by using callbacjks function we can set even more no of epochs
#after this we need to again plot the loss and hence observe overfitting problem solved 


# In[22]:


# for regularixation we add a another function 'dropout'
from tensorflow.keras.layers import Dropout


# In[23]:


model=Sequential()
model.add(Dense(8,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(4,activation='relu'))
model.add(Dropout(0.5))
# Since this is a binary classification so we use 'sigmoid ' as activation function

model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam')


# In[24]:


model.fit(x=X_train,y=y_train,epochs=200,validation_data=(X_test,y_test))


# In[25]:


model_loss=pd.DataFrame(model.history.history)


# In[26]:


model_loss.plot()


# In[27]:


predictions=model.predict_classes(X_test)


# In[28]:


from sklearn.metrics import classification_report,confusion_matrix


# In[29]:


print(classification_report(y_test,predictions))


# In[30]:


print(confusion_matrix(y_test,predictions))


# In[ ]:




