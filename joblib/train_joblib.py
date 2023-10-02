#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression


# In[3]:


import joblib

#to save model


# In[4]:


url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'


# In[5]:


names = ['preg' , 'plas' , 'pres' , 'skin' , 'test' , 'mass' , 'pedi' , 'age' , 'class']


# In[6]:


df = pd.read_csv(url , names = names)


# In[7]:


print(df.head())


# In[8]:


arr = df.values


# In[9]:


X , Y = arr[:,0:8] , arr[:,8]


# In[10]:


X_train , X_test , Y_train , Y_test = model_selection.train_test_split(X,Y,test_size=0.2,random_state=101)


# In[11]:


model = LogisticRegression()
model.fit(X_train,Y_train)
print('INFO - model has trained')


# In[12]:


result = model.score(X_test,Y_test)


# In[13]:


print(f'model accuracy is {result}')


# In[14]:


model.predict(X_test)


# In[15]:


#take live data

data = model.predict([[1,1,1,1,1,1,1,1]])    
if data[0] == 0:
    print('person in not diabetic')
else:
    print('person is diabetic')


# # model saving

# In[16]:


filename = 'diabetic_80.pkl' #or .sav

joblib.dump(model , filename)

#model we want to save , filename only

