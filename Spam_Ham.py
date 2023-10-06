#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# In[2]:


data=pd.read_csv('spambase.csv')


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data.shape


# In[6]:


data.isnull().sum()


# In[7]:


data['class'].value_counts()


# In[8]:


labels = ["spam","ham"]
msg_count = [data['class'].value_counts()[1],data['class'].value_counts()[0]]

plt.pie(x=msg_count,explode = [0,0.1],labels=labels,autopct='%.0f%%')
plt.title("spam and ham in percentage")
plt.legend()
plt.show()


# In[9]:


a=data[['word_freq_make','class']].corr()
print(a)
a.iloc[1,0]


# In[10]:


l=list(data.columns)
l.pop()
l


# In[11]:


for i in l:
    corr=data[[i,'class']].corr()
    a=corr.iloc[1,0]
    if abs(a)<0.1:
        data=data.drop([i],axis=1)


# In[12]:


data.columns


# In[13]:


data.shape


# In[14]:


x=data.drop(['class'],axis=1)
y=data['class']


# In[15]:


S=StandardScaler()
x=S.fit_transform(x)


# In[16]:


x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=11,test_size=0.2)


# In[17]:


#finding optimal value of k
k=0
M=-1
for i in range (1,10):
    ModelK=KNeighborsClassifier(n_neighbors=i)
    ModelK.fit(x_train,y_train)
    S=ModelK.score(x_test,y_test)
    if S>M:
        M=S
        k=i
print(k)


# In[18]:


model=KNeighborsClassifier(n_neighbors=k)


# In[19]:


model.fit(x_train,y_train)


# In[20]:


prediction=model.predict(x_test)


# In[21]:


accuracy_score(y_test,prediction)


# In[22]:


confusion_matrix(y_test,prediction)


# In[23]:


Prediction=model.predict(x_test)
Prediction=pd.Series(Prediction)


# In[24]:


y_test.plot(kind='kde',label='Actual Values')
Prediction.plot(kind='kde',label='Predicted Values')
plt.legend()


# In[25]:


accuracy = accuracy_score(y_test, Prediction)
print(f'Accuracy: {accuracy * 100:.2f}%')

