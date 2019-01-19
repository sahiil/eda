#!/usr/bin/env python
# coding: utf-8

# # Problem statement
# 
# #### A cash calculator has to be created in order to find the number of people spending on credit card and estimate the paydown curve
# 

# # Course of action
# 
# ## Exploratory data analysis
# ## Data Preparation
# ## Data Modelling
# ## Conclusion
# 
# 

# In[ ]:


#Import Libraries
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import scipy.stats as st
from sklearn import ensemble, tree, linear_model
import missingno as msno


# # Exploratory Data Analysis

# In[ ]:


#read file

df = pd.read_csv(r"C:\Users\sahil\Desktop\Amex\train.csv")

#describe the datatype
df.describe()


# In[ ]:


# Select numerical features
numeric_features = df.select_dtypes(include=[np.number])
numeric_features.columns

# Select categorical features
categorical_features = df.select_dtypes(include=[np.object])
categorical_features.columns

#Missing value visualize
msno.matrix(df.sample(250))
msno.heatmap(df)


# In[ ]:


df.skew()
#df.kurt()


# In[23]:


#plot single graph with transformation
y = df['is_click']
plt.figure(1); plt.title('Johnson SU')
sns.distplot(y, kde=False, fit=st.johnsonsu)
plt.figure(2); plt.title('Normal')
sns.distplot(y, kde=False, fit=st.norm)
plt.figure(3); plt.title('Log Normal')
sns.distplot(y, kde=False, fit=st.lognorm)


# In[24]:


sns.distplot(df.skew(),color='blue',axlabel ='Skewness')


# In[34]:


correlation = numeric_features.corr()
print(correlation['is_click'].sort_values(ascending = False),'\n')

f , ax = plt.subplots(figsize = (14,12))

plt.title('Correlation of Numeric Features with Sale Price',y=1,size=16)

sns.heatmap(correlation,square = True,  vmax=0.8)


# In[ ]:


#Pairwise plot

sns.set()
columns = correlation.columns.values
sns.pairplot(df[columns],size = 2 ,kind ='scatter',diag_kind='kde')
plt.show()


# In[37]:


# Missing value treatment numeric

total = numeric_features.isnull().sum().sort_values(ascending=False)
percent = (numeric_features.isnull().sum()/numeric_features.isnull().count()).sort_values(ascending=False)
missing_data_n = pd.concat([total, percent], axis=1,join='outer', keys=['Total Missing Count', '% of Total Observations'])
missing_data_n.index.name =' Numeric Feature'

missing_data_n.head(20)


# In[38]:



# Missing value treatment categorical
total = categorical_features.isnull().sum().sort_values(ascending=False)
percent = (categorical_features.isnull().sum()/categorical_features.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1,join='outer', keys=['Total Missing Count', ' % of Total Observations'])
missing_data.index.name ='Feature'
missing_data.head(20)


# # Data Manipulations

# In[46]:


# Define train and test variables
df.dropna(inplace=True)
X=df[['age_level','user_depth','city_development_index']]
y=df['is_click']


# In[47]:


# train test split
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[48]:



# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[59]:


# Define classifier
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0, probability=True)
classifier.fit(X_train, y_train)


# In[53]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


# In[72]:


import numpy as np
from sklearn import metrics
y_pred = classifier.predict_proba(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred[:,1])
metrics.auc(fpr, tpr)


# In[76]:


from matplotlib import pyplot as plt
plt.clf()
plt.plot(fpr, tpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
plt.show()

