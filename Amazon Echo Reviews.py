#!/usr/bin/env python
# coding: utf-8

# # AMAZON ECHO REVIEW CLASSIFICATION AND ANALYSIS

# This dataset consists of 3000 Amazon customer reviews, star ratings, date of review, variant and feedback of various Amazon Alexa products like Alexa Echo, Echo dots.
# The objective is to discover insights into consumer reviews and perform sentiment analysis on the data.
# 
# Dataset: www.kaggle.com/sid321axn/amazon-alexa-reviews

# # IMPORT LIBRARIES

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# # IMPORT DATASET

# In[3]:


echo = pd.read_csv('amazon_alexa.tsv',sep='\t')


# In[4]:


echo


# In[21]:


echo.info()


# In[22]:


echo.describe()


# In[23]:


echo.head()


# In[24]:


echo.tail()


# In[25]:


echo['verified_reviews']


# Checking Null Values

# In[6]:


sns.heatmap(echo.isnull(),cmap='Blues',cbar=False)


# # DATA VISUALIZATION

# Model Variation Count

# In[20]:


plt.figure(figsize=[40,14])
sns.countplot(x='variation',data=echo,palette='dark')


# Rating

# In[19]:


echo['rating'].hist(bins=5)


# In[30]:


positive = echo[echo['feedback']==1]


# In[31]:


negative = echo[echo['feedback']==0]


# In[32]:


positive


# In[33]:


negative


# POSITIVE FEEDBACK COUNT WRT. VARIATION

# In[35]:


plt.figure(figsize=[40,14])
sns.countplot(x='variation',data=positive,palette='Blues')


# NEGATIVE FEEDBACK COUNT WRT. VARIATION

# In[36]:


plt.figure(figsize=[40,14])
sns.countplot(x='variation',data=negative,palette='Greens')


# FEEDBACK COUNT

# In[38]:


sns.countplot(x='feedback',data=echo,palette='colorblind')


# RATING COUNT

# In[39]:


sns.countplot(x = 'rating', data = echo)


# RATING WRT. VARIATIONS

# In[44]:


plt.figure(figsize = (40,15))

sns.barplot(x = 'variation', y='rating', data=echo, palette = ("RdGy"))


# # FEATURE ENGINEERING

# In[45]:


echo


# DROPPING UNWANTED COLUMNS

# In[46]:


echo.drop('date',axis=1,inplace=True)


# In[47]:


echo


# In[48]:


var = pd.get_dummies(echo['variation'],drop_first=True)


# In[49]:


var


# In[50]:


echo.drop('variation',axis=1,inplace=True)


# In[51]:


echo = pd.concat([echo,var],axis=1)


# In[52]:


echo


# # TRAINING THE DATA

# In[53]:


from sklearn.feature_extraction.text import CountVectorizer


# In[54]:


cv = CountVectorizer()


# In[55]:


alexa = cv.fit_transform(echo['verified_reviews'])


# In[56]:


alexa


# In[58]:


print(cv.get_feature_names())


# In[59]:


print(alexa.toarray())


# In[60]:


echo.drop(['verified_reviews'],axis=1,inplace=True)


# In[61]:


reviews = pd.DataFrame(alexa.toarray())


# In[64]:


echo = pd.concat([echo,reviews],axis=1)


# In[65]:


echo


# In[67]:


X = echo.drop('feedback',axis=1).values


# In[68]:


X


# In[69]:


y = echo['feedback'].values


# In[70]:


y


# In[72]:


from sklearn.model_selection import train_test_split


# In[73]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


# In[74]:


X_train.shape


# In[75]:


X_test.shape


# In[76]:


y_train.shape


# In[77]:


y_test.shape


# In[78]:


from sklearn.ensemble import RandomForestClassifier


# In[79]:


randomforest_classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy')
randomforest_classifier.fit(X_train, y_train)


# # TESTING THE DATA

# In[82]:


from sklearn.metrics import confusion_matrix,classification_report


# In[83]:


y_predict_train = randomforest_classifier.predict(X_train)
cm = confusion_matrix(y_train, y_predict_train)


# In[84]:


sns.heatmap(cm, annot=True)


# In[85]:


print(classification_report(y_train, y_predict_train))


# In[86]:


y_predict = randomforest_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict)


# In[87]:


sns.heatmap(cm, annot=True)


# In[88]:


print(classification_report(y_test, y_predict))

