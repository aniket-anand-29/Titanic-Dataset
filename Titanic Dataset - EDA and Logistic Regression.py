#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
titanic_training_data = pd.read_csv("D:/CN DATA SCIENCE/titanic_training_data.csv")


# In[2]:


titanic_training_data.head(10)


# In[3]:


titanic_training_data.info()


# ## DATA ANALYSIS
# 

# In[4]:


sns.countplot( x = 'Survived' , data = titanic_training_data)


# In[5]:


sns.countplot( x = 'Survived' , data = titanic_training_data, hue = 'Sex')


# In[6]:


sns.countplot( x = 'Survived' , data = titanic_training_data, hue = 'Pclass')


# In[7]:


sns.countplot(x = 'SibSp' , data = titanic_training_data)


# ## DATA WRANGLING AND CLEANING

# In[8]:


titanic_training_data.isnull().sum()


# In[9]:


sns.heatmap(titanic_training_data.isnull() , yticklabels = False)


# #### The heatmap above shows us that the column named 'Cabin' mostly has null / nan values and hence may be dropped

# In[10]:


titanic_training_data.drop('Cabin', axis = 1, inplace=True)


# In[11]:


sns.boxplot(x = 'Pclass', y = 'Age', data = titanic_training_data)


# #### The box plot above shows us that the first and second class passengers are elder to the passengers in the third class

# In[12]:


titanic_training_data['Sex_0/1'] = titanic_training_data['Sex'].map({"female":0, "male":1})
titanic_training_data.drop('Sex', axis=1, inplace=True)


# In[13]:


titanic_training_data.head()


# #### Name, Ticket, PassengerID can be dropped because they do not provide any specific insight

# In[14]:


titanic_training_data.drop(['PassengerId', 'Name', 'Ticket'], axis = 1, inplace = True)


# In[15]:


Embark = pd.get_dummies(titanic_training_data['Embarked'], drop_first=True)
Embark.head()


# In[16]:


Class = pd.get_dummies(titanic_training_data['Pclass'] ,drop_first=True)
Class.head()


# In[17]:


titanic_training_data = pd.concat([titanic_training_data, Class, Embark], axis=1)


# In[18]:


titanic_training_data.dropna(inplace=True)


# In[19]:


titanic_training_data.isnull().sum()


# In[20]:


titanic_training_data.drop(['Pclass', 'Embarked'], axis = 1, inplace = True)


# ## TRAIN AND TEST

# In[21]:


X = titanic_training_data.drop('Survived' , axis = 1)
Y = titanic_training_data['Survived']
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(X,Y, test_size=0.3 , random_state=1)


# In[22]:


from sklearn.linear_model import LogisticRegression


# In[23]:


model = LogisticRegression()


# In[24]:


model.fit(x_train, y_train)


# In[25]:


y_test_pred = model.predict(x_test)


# In[26]:


from sklearn.metrics import classification_report


# In[32]:


print(classification_report(y_test,y_test_pred))


# In[28]:


from sklearn.metrics import confusion_matrix


# In[29]:


confusion_matrix(y_test, y_test_pred)


# In[30]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_test_pred)


# In[31]:


model.score(x_train,y_train)


# In[ ]:




