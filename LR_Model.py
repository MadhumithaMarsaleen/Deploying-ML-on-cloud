#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pickle


# In[18]:


data = pd.read_csv("C:\\Users\\mmarsaleen\\Desktop\\New folder\\Azure\\mpg.csv")
data.head()


# In[19]:


data=data.drop(["name"],axis=1)


# In[20]:


data.head()


# In[21]:


data.shape


# In[22]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() 
data_scaled = scaler.fit_transform(data)
df = pd.DataFrame.from_records(data_scaled)


# In[39]:


x=data.iloc[:,1:9]
x


# In[41]:


y=data.iloc[:,0]
y


# In[42]:



from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()
lr_model.fit(x, y)


# In[44]:


from sklearn.metrics import r2_score
y_predict = lr_model.predict(x)
r2 = r2_score(y, y_predict)
print('R2 score is {}'.format(r2))


# In[45]:


pickle.dump(lr_model, open("C:\\Users\\mmarsaleen\\Desktop\\New folder\\Azure\\lr_model.pkl",'wb'))


# In[ ]:




