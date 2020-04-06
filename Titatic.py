#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


train = pd.read_csv("/home/fakhredine/Documents/microsoft/DB/CSV/titatic/train.csv")
test = pd.read_csv("/home/fakhredine/Documents/microsoft/DB/CSV/titatic/test.csv")
gender = pd.read_csv("/home/fakhredine/Documents/microsoft/DB/CSV/titatic/gender_submission.csv")


# In[3]:


train.dtypes


# In[4]:


train


# In[6]:


f = plt.figure(figsize=(19, 15))
plt.matshow(train.corr(), fignum=f.number)
plt.xticks(range(train.shape[1]), train.columns, fontsize=14, rotation=45)
plt.yticks(range(train.shape[1]), train.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=18)
plt.title('Correlation Matrix', fontsize=16)


# In[7]:


rs = np.random.RandomState(0)
df = pd.DataFrame(rs.rand(10, 10))
corr = train.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)


# In[8]:


test


# In[9]:


gender


# In[10]:


NaNColumns = train.columns[train.isna().any()].tolist()
NaNColumns


# In[11]:


NaNColumns = test.columns[test.isna().any()].tolist()
NaNColumns


# In[12]:


train = train.drop(["Cabin", "Ticket", "Name"], axis=1)


# In[13]:


train


# In[14]:


train = train.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
train


# In[15]:


print(len(train[train['Survived'] == 1]))
print(len(train[train['Survived'] == 0]))


# In[16]:


train['Survived'].value_counts().plot.bar()


# In[17]:


train["Embarked"] = train["Embarked"].astype('category').cat.codes
train["Sex"] = train["Sex"].astype('category').cat.codes

train


# In[18]:


X = train.drop('Survived',axis=1)
y = train['Survived']
train.shape


# In[19]:


from sklearn.model_selection import train_test_split
np.random.seed(3456)
# data_split = train_test_split(np.asmatrix(data), test_size = 0.2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=64)



# X_train = data_split[0]
# y_train = np.ravel(data_split[0][:, 0])
# X_test = data_split[1]
# y_test = np.ravel(data_split[1][:, 0])

print(X_train.shape)
print(X_test.shape)
print()
print(y_train.shape)
print(y_test.shape)


# In[37]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)


# In[38]:


valid_test = pd.DataFrame(X_test, columns = train.columns)
valid_test['predicted'] = model.predict(X_test)
valid_test['correct'] = [1 if x == z else 0 for x, z in zip(valid_test['predicted'], y_test)]

score = model.score(X_test, y_test)
accuracy = float(sum(valid_test['correct'])) / float(valid_test.shape[0])

print(score)
print(accuracy)


# In[39]:


# Matrice de confusion
from sklearn.metrics import confusion_matrix
cm = pd.DataFrame(confusion_matrix(y_train, model.predict(X_train)), columns=['prédit ' + str(i) for i in model.classes_])
cm.index = ['vrai ' + str(i) for i in model.classes_]
cm


# In[54]:


# csvOutPut = valid_test['resilies'] + valid_test['predicted'] + valid_test['correct']
csvOutPut = train
csvOutPut['correct'] = valid_test['correct']
csvOutPut['prediction'] = valid_test['predicted']

csvOutPut[csvOutPut['prediction'] == 1]
csvOutPut[csvOutPut['Survived'] == 1]

# csvOutPut
# csvOutPut.to_csv("/home/fakhredine/Desktop/valid.csv", index=False)


# In[33]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)


# In[35]:


valid_test = pd.DataFrame(X_test, columns = train.columns)
valid_test['predicted'] = model.predict(X_test)
valid_test['correct'] = [1 if x == z else 0 for x, z in zip(valid_test['predicted'], y_test)]

score = model.score(X_test, y_test)
accuracy = float(sum(valid_test['correct'])) / float(valid_test.shape[0])

print(score)
print(accuracy)


# In[ ]:




