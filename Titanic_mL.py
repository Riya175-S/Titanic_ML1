#!/usr/bin/env python
# coding: utf-8

# # Titanic Survival Prediction Model -
# # This model predicts whether a passenger on Titanic will survive or not 

# In[1]:


#IMPORTING LIBRARIES
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


#Loading the data-set
Titanic = sns.load_dataset('Titanic')


# In[3]:


#Printing the first 5 rows
Titanic.head(8)


# In[4]:


#By default .head() reads 5 lines (indexing 0 to 4)
Titanic.head()


# In[5]:


#Printing the firt 10 rows (Indexing 0 to 9)
Titanic.head(10)


# In[6]:


#Counting the nows of rows and columns in the dataset
Titanic.shape


# In[7]:


#Get sdme statistics
Titanic.describe()


# In[8]:


#Getting the number of survivors
Titanic['survived'].value_counts()
# 0 - Not survived / 1- Survived


# In[9]:


#Visualising the count of survivors
sns.countplot( Titanic['survived'] )


# In[10]:


#visualizing the count of urvivors for particular columns
cols = ['who' , 'sex', 'pclass', 'sibsp', 'parch', 'embarked']
n_rows = 3
n_cols = 2

#The subplot grid and figure size of each graph
fig, axs = plt.subplots(n_rows, n_cols, figsize = (n_cols*3.2, n_rows*3.2) )

# r-rows , c-columns
for r in range(0, n_rows):
    for c in range(0, n_cols):
        i = r*n_cols + c     #index to go through the number of columns
        ax = axs[r][c]       #show where to position each sub plot
        sns.countplot( Titanic[cols[i]] , hue = Titanic['survived'], ax = ax)
        ax.set_title(cols[i])
        ax.legend(title ='survived' , loc ='upper right')
        
plt.tight_layout()


# In[11]:


#looking at survival rate by gender
Titanic.groupby('sex')[['survived']].mean()


# In[12]:


Titanic.groupby('pclass')[['survived']].mean()


# In[13]:


Titanic.groupby('who')[['survived']].mean()


# In[14]:


#looking at survival rate by two fcators
Titanic.pivot_table('survived', index='sex', columns = 'class')


# In[15]:


#lookimg at survival rate by two factors visulally
# using matplotlib
Titanic.pivot_table('survived', index='sex', columns = 'class').plot()


# In[16]:


Titanic.pivot_table('survived', index='who', columns = 'class')


# In[17]:


Titanic.pivot_table('survived', index='who', columns = 'class').plot()


# In[18]:


#Plotting the survival rate of each class
sns.barplot(x ='class', y ='survived', data = Titanic)


# In[19]:


#Plotting the survival rate of each class
sns.barplot(x='sex', y='survived', data= Titanic)


# In[20]:


#Looking at survival rate by gender, age and class
age = pd.cut( Titanic['age'], [0,18,80])
Titanic.pivot_table("survived", ['sex', age],'class')


# In[21]:


age = pd.cut( Titanic['age'], [0,18,80])
Titanic.pivot_table("survived", ['who', age],'class')


# In[22]:


#Plotting the price paid by each class
plt.scatter(Titanic['fare'], Titanic['class'], color = 'green', label = 'Passenger paid')
plt.ylabel('class')
plt.xlabel("Price/Fare")
plt.title('Price of each class')
plt.legend()
plt.show()


# In[23]:


#Look at all the values in each column and count
for val in Titanic:
    print(Titanic[val].value_counts())
    print()


# In[24]:


#Count the empty values in each column
Titanic.isna().sum()


# In[25]:


#Dropping the columns
#Removing the rows with zero/ null values
Titanic = Titanic.dropna(subset = ['embarked', 'age'])


# In[26]:


#looking at hte data types
Titanic.dtypes


# In[27]:


#count the new number of rows and columns in the dataset
Titanic.shape


# In[28]:


from sklearn.preprocessing import LabelEncoder
lc = LabelEncoder()

#Encoding the gender column
Titanic.iloc[:, 2] = lc.fit_transform( Titanic.iloc[:, 2].values )

#Encoding the embarked column
Titanic.iloc[:, 7] = lc.fit_transform( Titanic.iloc[:, 7].values )


# In[29]:


#Printing unique values in each column
print(Titanic['sex'].unique())
print(Titanic['embarked'].unique())


# In[30]:


Titanic.dtypes


# In[31]:


Titanic.isna().sum()


# In[32]:


#splitting data into dependent and independent variables
# x-independent, y-dependent
x = Titanic.iloc[:,1:8].values
y = Titanic.iloc[:, 0].values


# In[33]:


#splitting the datatet into training and testing model
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 0)
#test size = 0.2 = 20% testing, 80% training


# In[34]:


#scale the data


# In[35]:


#creating a function with machine learning model
def models(x_train, y_train) :
    
    #using logistic regression
    from sklearn.linear_model import LogisticRegression
    z = LogisticRegression(random_state = 0)
    z.fit(x_train, y_train)
    
    #using concept of KNeighbours
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p=2)
    knn.fit(x_train, y_train)
    
    #printing training accuracy of each model
    print('[0] Logistic regresion training accuracy : ', z.score(x_train, y_train))
    print('[1] K Neigbors training accuracy : ', knn.score(x_train, y_train))
    
    return z, knn


# In[36]:


#Get and train all the 2 models
model = models(x_train, y_train)


# In[37]:


#printing prediction of logistic regression
pred = model[0].predict(x_test)
print(pred)

#printing the actual values
print(y_test)

#Reason for the accuracy

