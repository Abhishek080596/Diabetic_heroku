#!/usr/bin/env python
# coding: utf-8

# In[4]:


#we'll put this in production


# # load the model

# In[5]:


import joblib


# In[6]:


model2 = joblib.load('diabetic_80.pkl')
data = model2.predict([[1,2,3,4,5,6,7,8]])


# In[7]:


if data[0] == 0:
    print('person in not diabetic')
else:
    print('person is diabetic')


# In[ ]:





# In[11]:


# load the model



model2 = joblib.load('diabetic_80.pkl') #just filename
data = model2.predict([[1,2,3,4,5,6,7,8]])

if data[0] == 0:
    print('person in not diabetic')
else:
    print('person is diabetic')


# In[ ]:


# i can do model.fit() on new data here after loading to retrain the model


# In[ ]:




