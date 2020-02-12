#!/usr/bin/env python
# coding: utf-8

# In[53]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[54]:


train = pd.read_csv('D:DS_TriS/titanic_data.csv')
train.head()


# In[55]:


train.info()


# In[56]:


train.describe()


# In[57]:


train.isnull().sum()


# In[58]:


#EDA
sns.countplot(x='Survived',data=train)


# In[59]:


sns.countplot(x='Survived',hue='Sex',data=train)


# In[60]:


sns.countplot(x='Survived',hue='Pclass',data=train)


# In[61]:


sns.distplot(train['Age'].dropna(),bins=30)


# In[62]:


sns.countplot(x='SibSp',data=train)


# In[63]:


train['Fare'].hist(color='green',bins=40,figsize=(8,4))


# In[64]:


sns.boxplot(x='Pclass',y='Age',data=train)


# In[65]:


sns.heatmap(train.isnull())


# In[66]:


sns.heatmap(train.corr())


# In[67]:


train['Age']=train['Age'].fillna(28.000000)


# In[68]:


train['Embarked']=train['Embarked'].fillna('S')


# In[69]:


X = train.iloc[:, [2, 4, 5, 6, 7, 9,11]]
y = train.iloc[:, 1]


# In[70]:


X.head()


# In[71]:


y.head()


# In[72]:


sex = pd.get_dummies(X['Sex'], prefix = 'Sex')


# In[73]:


embark = pd.get_dummies(X['Embarked'], prefix = 'Embarked')


# In[74]:


passenger_class = pd.get_dummies(X['Pclass'], prefix = 'Pclass')


# In[75]:


X = pd.concat([X,sex,embark, passenger_class],axis=1)
X.head()


# In[76]:


sns.boxplot(data= X).set_title("Outlier Box Plot")


# In[77]:


X.columns


# In[78]:


X=X.drop(['Sex','Embarked','Pclass'],axis=1)
X.head()


# In[79]:


X['travel_alone']=np.where((X['SibSp']+X['Parch'])>0,1,0)
X.corr()


# In[80]:


X.head()


# In[81]:


X=X.drop(['SibSp','Parch','Sex_male'],axis=1)
X.head()


# In[82]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=42)


# In[83]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train.iloc[:, [0,1]] = sc.fit_transform(X_train.iloc[:, [0,1]])
X_test.iloc[:, [0,1]] = sc.transform(X_test.iloc[:, [0,1]])


# In[84]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)


# In[85]:


classifier.score(X_test,y_test)


# In[86]:


classifier.score(X_train,y_train)


# In[87]:


from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV


# In[88]:


rfecv = RFECV(estimator=classifier, step=1, cv=StratifiedKFold(2), scoring='accuracy')
rfecv.fit(X_train, y_train)


# In[89]:


print("Optimal number of features : %d" % rfecv.n_features_)


# In[90]:


plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()


# In[91]:


from sklearn.feature_selection import RFE

rfe = RFE(classifier, rfecv.n_features_)
rfe = rfe.fit(X_train, y_train)
print(list(X.columns[rfe.support_]))


# In[92]:


x=X.drop(['Fare','Embarked_C','Pclass_2','travel_alone'],axis=1)
x.head()


# In[93]:


y.head()


# In[94]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,random_state=42)


# In[95]:


from sklearn.linear_model import LogisticRegression
model= LogisticRegression()
model.fit(X_train, y_train)


# In[96]:


model.score(X_test,y_test)


# In[97]:


y_pred = model.predict(X_test)
y_pred


# In[98]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
model_accuracy = accuracies.mean()
model_standard_deviation = accuracies.std()


# In[99]:


model_accuracy,model_standard_deviation


# In[48]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)


# In[49]:


confusion_matrix


# In[50]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[51]:


list(X.columns[rfe.support_])


# In[52]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
area_under_curve = roc_auc_score(y_test, model.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % area_under_curve)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# In[ ]:




