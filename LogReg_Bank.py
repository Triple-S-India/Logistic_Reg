#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
"""plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
"""


# In[14]:


data = pd.read_csv("D:DS_TriS/bankdata.csv",delimiter=";")


# In[15]:


data = data.dropna()


# In[16]:


data.head()


# In[17]:


data.shape


# In[18]:


type(data)


# In[19]:


data.columns


# In[26]:


data.dtypes


# In[27]:


data.info()


# In[29]:


data['education'].unique()


# In[30]:


data['y'].value_counts()


# In[31]:


import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


# In[32]:


sns.countplot(x = 'y',data = data,palette='hls')
plt.show()
plt.savefig('count_plot')


# In[60]:


data.groupby('y').size()


# In[61]:


count_no_sub = data.groupby('y').size()[0]
count_sub = data.groupby('y').size()[1]
pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
print("percentage of no subscription is", pct_of_no_sub*100)
pct_of_sub = count_sub/(count_no_sub+count_sub)
print("percentage of subscription", pct_of_sub*100)


# In[62]:


data.groupby('y').mean()


# In[63]:


data.groupby('job').mean()


# In[64]:


data.groupby('marital').mean()


# In[65]:


#Visualization
get_ipython().run_line_magic('matplotlib', 'inline')
pd.crosstab(data.job,data.y).plot(kind='bar')
plt.title('Purchase Frequency for Job Title')
plt.xlabel('Job')
plt.ylabel('Frequency of Purchase')
plt.savefig('purchase_fre_job')


# In[67]:


#The frequency of purchase of the deposit depends a great deal on the job title. 
#Thus, the job title can be a good predictor of the outcome variable.


# In[68]:


table=pd.crosstab(data.marital,data.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Marital Status vs Purchase')
plt.xlabel('Marital Status')
plt.ylabel('Proportion of Customers')
plt.savefig('mariral_vs_pur_stack')


# In[69]:


#The marital status does not seem a strong predictor for the outcome variable.


# In[70]:


table=pd.crosstab(data.education,data.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Education vs Purchase')
plt.xlabel('Education')
plt.ylabel('Proportion of Customers')
plt.savefig('edu_vs_pur_stack')


# In[71]:


data.age.hist()
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('hist_age')


# In[72]:


#Most of the customers of the bank in this dataset are in the age range of 30–40.


# In[73]:


#Create_Dummy_Variables


# In[75]:


dummy_bank = pd.get_dummies(data, columns=['job','marital',
                                         'education','default',
                                         'housing','loan',
                                         'contact','month',
                                         'poutcome'])
dummy_bank.shape
dummy_bank_train = dummy_bank[['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous','y',
       'job_admin.', 'job_blue-collar', 'job_entrepreneur', 'job_housemaid',
       'job_management', 'job_retired', 'job_self-employed', 'job_services',
       'job_student', 'job_technician', 'job_unemployed', 'job_unknown',
       'marital_divorced', 'marital_married', 'marital_single',
       'education_primary', 'education_secondary', 'education_tertiary',
       'education_unknown', 'default_no', 'default_yes', 'housing_no',
       'housing_yes', 'loan_no', 'loan_yes', 'contact_unknown', 'month_may',
       'poutcome_unknown']]
dummy_bank_train.shape


# In[76]:


dummy_bank_train.y.unique()
c = pd.value_counts(dummy_bank_train.y, sort = True).sort_index()
c


# In[78]:


data_y = pd.DataFrame(dummy_bank_train['y'])
data_X = dummy_bank_train.drop(['y'], axis=1)
print(data_X.count())
print(data_y.count())


# In[79]:


from sklearn.utils import shuffle


# In[80]:


train_data= pd.concat([data_X,data_y], axis=1)
train_data.y.replace(('yes','no'),(1,0), inplace=True)
X_1 =train_data[ train_data["y"]==1 ]
X_0=train_data[train_data["y"]==0]
X_0=shuffle(X_0,random_state=42).reset_index(drop=True)
X_1=shuffle(X_1,random_state=42).reset_index(drop=True)

ALPHA=1.5

X_0=X_0.iloc[:round(len(X_1)*ALPHA),:]
data_final=pd.concat([X_1, X_0])
d = pd.value_counts(data_final['y'])
d
c1 = d[0]/(d[0]+d[1] )
c2  = d[1]/(d[0]+d[1])
sizes = [c1,c2]
plot = plt.pie(sizes, labels = ['no','yes'],autopct='%1.1f%%',
        shadow=True, startangle=45 )
plt.axis('equal') 
plt.title("Class Imbalance Problem")
plt.show()


# In[81]:


data_final.head()


# In[82]:


dummy_bank.y.factorize()


# In[83]:


data_y = pd.DataFrame(data_final['y'])
data_X = data_final.drop(['y'], axis=1)
print(data_X.columns)
print(data_y.columns)


# In[84]:


#Train_Test_Splitting


# In[85]:


from sklearn.model_selection import train_test_split


# In[86]:


X_train,X_test,y_train,y_test = train_test_split(data_X, data_y, test_size=0.2,random_state=0)


# In[87]:


#Regression_Model(LOGISTIC)
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# In[88]:


classifier = LogisticRegression()
classifier.fit(X_train,y_train)


# In[90]:


#Predicting the set results
y_pred = classifier.predict(X_test)
y_pred


# In[91]:


#Accuracy
classifier.score(X_test,y_test)


# In[92]:


#confusion_metrix
metrics.confusion_matrix(y_test,y_pred)


# In[93]:


#Confusion metrix telling that 1372+725 correct and 319+229 incorrect predictions out of 1372+725+319+229 predictions


# In[98]:


#Compute precision, recall, F-measure and support
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[107]:


#ROC_Curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, classifier.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, classifier.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# In[ ]:


#The dotted line represents the ROC curve of a purely random classifier; 
#a good classifier stays as far away from that line as possible (toward the top-left corner).

