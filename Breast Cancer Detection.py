#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from mpl_toolkits import mplot3d


# In[3]:


dataset = pd.read_csv('Downloads\Breast_cancer_data.csv')
dataset.head()


# In[4]:


dataset.describe()


# In[5]:


dataset.drop(['diagnosis','mean_area','mean_smoothness'], axis=1).plot.line(title='Cancer Dataset')


# In[6]:


dataset['mean_area'].hist()


# In[7]:


dataset.boxplot(column = 'mean_area')


# In[8]:


dataset['mean_smoothness'].hist()


# In[9]:


dataset.boxplot(column = 'mean_smoothness')


# In[10]:


# number of samples ! for infected and 0 for non-infected
temp3 = pd.crosstab(dataset['diagnosis'], dataset['diagnosis'])
temp3.plot(kind='bar', stacked=True, color=['pink','blue'], grid=False)


# In[11]:


#HeatMap of dataset corrilation
corr = dataset.corr()
fig, ax = plt.subplots()
# create heatmap
im = ax.imshow(corr.values)

# set labels
ax.set_xticks(np.arange(len(corr.columns)))
ax.set_yticks(np.arange(len(corr.columns)))
ax.set_xticklabels(corr.columns)
ax.set_yticklabels(corr.columns)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")


# In[12]:


print("Cancer data set dimensions : {}".format(dataset.shape))


# In[13]:


dataset.isnull().sum()
dataset.isna().sum()


# In[14]:


X = dataset.iloc[:, 0:5].values
Y = dataset.iloc[:,5].values


# In[15]:


# Spliting data into trainingset(75%) and testset(25%)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


# In[16]:


# Scaling the features

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[17]:


#Using Logistic Regression Algorithm to the Training Set

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)


# In[18]:


Y_pred = classifier.predict(X_test)


# In[19]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)


# In[20]:


print(cm)


# In[21]:


plt.imshow(cm, cmap='binary')


# In[22]:


#Using SVC method of svm class to use Kernel SVM Algorithm

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, Y_train)


# In[23]:


Y_pred = classifier.predict(X_test)


# In[24]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)


# In[25]:


print(cm)


# In[26]:


def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements 


# In[27]:


accur = accuracy(cm)
print("Model Accuracy: ", accur)


# In[56]:


fig = plt.figure(figsize=(10,6))
ax = plt.axes(projection='3d')
ax.plot_surface(dataset['mean_radius'], dataset['mean_texture'], np.array([dataset['mean_perimeter'],dataset['mean_area']]), rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('surface');


# In[64]:


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Get the data (csv file is hosted on the web)
url = 'Downloads\Breast_cancer_data.csv'
data = pd.read_csv(url)

# Transform it to a long format
df=data.unstack().reset_index()
df.columns=["X","Y","Z"]

# And transform the old column name in something numeric
df['X']=pd.Categorical(df['X'])
df['X']=df['X'].cat.codes

# We are going to do 20 plots, for 20 different angles
for angle in range(70,210,2):

# Make the plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.viridis, linewidth=0.2)

    ax.view_init(30,angle)

    filename='Downloads/Volcano_step'+str(angle)+'.png'
    plt.savefig(filename, dpi=96)
    plt.gca()


# In[ ]:




