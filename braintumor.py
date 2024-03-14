#!/usr/bin/env python
# coding: utf-8

# ## Brain Tumor Prediction
# 

# ## Business Problems and data understanding
# 
# ## Objective : 
# -  Develop predictive models to accurately classify brain tumor cases based on tumor-related features.
# ## Constraints:
# - Ensure that the data for tumor-related features (Area, Perimeter, Convex Area, Solidity, Equivalent Diameter, Major Axis, Minor Axis) is accurate, reliable, and comprehensive to facilitate effective model training and evaluation.
#     
# 
#  

# In[52]:


#importing required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression


# In[2]:


#loading dataset
df = pd.read_csv("Brain-tumor-detection.csv")


# In[3]:


df


# In[4]:


df.drop(columns=['Unnamed: 0'], axis=1, inplace=True)


# In[5]:


df.head()


# In[6]:


df.tail()


# In[7]:


df.shape


# In[16]:


df.info()


# In[ ]:





# ## EDA

# In[9]:


sns.boxplot(x="Class", y="Convex Area", data=df)


# In[ ]:





# In[11]:


# Plot pair plots for all numeric columns
sns.pairplot(df.select_dtypes(include=['float64', 'int64']))
plt.show()


# In[ ]:





# In[12]:


# Set the number of plots based on the number of numeric columns
num_plots = len(df.select_dtypes(include=['float64', 'int64']).columns)

# Create subplots
fig, axes = plt.subplots(1, num_plots, figsize=(20, 4))

# Plot histograms for each numeric column
for i, column in enumerate(df.select_dtypes(include=['float64', 'int64']).columns):
    sns.histplot(df[column], ax=axes[i], kde=True, color='skyblue')
    axes[i].set_title(f'Histogram of {column}')  # Set subplot title
    axes[i].set_xlabel(column)  # Set x-axis label
    axes[i].set_ylabel('Frequency')  # Set y-axis label

# Hide empty subplots
for j in range(i + 1, num_plots):
    axes[j].axis('off')

plt.tight_layout()
plt.show()



# In[ ]:





# In[13]:


# Set the number of plots based on the number of numeric columns
num_plots = len(df.select_dtypes(include=['float64', 'int64']).columns)

# Calculate the number of rows and columns for the subplots grid
num_rows = (num_plots + 2) // 3  # Adjust the number of rows as needed
num_cols = min(num_plots, 3)

# Create subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows))

# Flatten the axes array
axes = axes.flatten()

# Plot box plots for each numeric column
for i, col in enumerate(df.select_dtypes(include=['float64', 'int64']).columns):
    sns.boxplot(y=df[col], ax=axes[i], color='lightgreen')
    axes[i].set_title(f'Box plot of {col}')
    axes[i].set_ylabel(col)

# Hide empty subplots
for j in range(num_plots, num_rows * num_cols):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


#  ## Data Preprocessing

# In[25]:


df.info()


# In[26]:


df.isna().sum()


# In[27]:


df.duplicated().sum()


# In[28]:


df.drop_duplicates(inplace = True)


# In[29]:


df.duplicated().sum()


# In[30]:


df.describe().T


# In[31]:


def comp_eccentric(x):
    if x.isnumeric()==True:
        return abs(complex(x))
    else:
        return abs(complex(x[1:-1]))
df['Eccentricity'] = df['Eccentricity'].apply(comp_eccentric)


# In[32]:


df['Eccentricity']


# In[33]:


df.head()


# In[34]:


df.info()


# In[37]:


sns.heatmap(data=df.corr(),annot=True)


# ## Splitting the data

# In[38]:


X = df.drop("Class", axis=1)
y = df["Class"]


# In[39]:


from feature_engine.outliers import Winsorizer #removing outliers
for i in X.columns:
    mad_win=Winsorizer(capping_method='mad',tail='both',fold=1)
    mad_win.fit(X[[i]])
    X[[i]]=mad_win.transform(X[[i]])


# In[40]:


X_train,X_test, y_train, y_test= train_test_split(X,y,test_size = 0.2, random_state=42)


# In[43]:


sc= StandardScaler()
X_scaled = sc.fit_transform(X) # for standardising the features


# In[44]:


X_train


# ## Import the Model

# In[57]:


from sklearn.naive_bayes import GaussianNB


# ## Model Training

# In[60]:


gnb = GaussianNB()

gnb.fit(X_train, y_train)


# ## Prediction on Test Data

# In[61]:


y_pred = gnb.predict(X_test)


# ## Evaluating the Algorithms

# In[67]:


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy: {:.2f}%".format(accuracy*100))
print("Precision: {:.2f}%".format(precision*100))
print("Recall: {:.2f}%".format(recall*100))
print("F1-score: {:.2f}%".format(f1*100))


# In[ ]:





# Conclusion, the model developed for predicting Brain Tumor based on the provided dataset 
# - Accuracy: The model achieves an accuracy of 76.32%, indicating that approximately 76.32% of all predictions made by the model are correct.
# - Precision: With a precision of 85.71%, the GNB model demonstrates a high proportion of true positive predictions relative to all positive predictions made. This implies that when the model predicts a positive outcome, it is accurate approximately 85.71% of the time.
# - Recall: The recall of 82.76% suggests that the model effectively captures approximately 82.76% of all actual positive instances in the dataset. In other words, it demonstrates a good ability to identify positive instances, minimizing false negative predictions.
# - F1-score: The F1-score, calculated as 84.21%, provides a harmonic mean of precision and recall. This metric balances both precision and recall, offering a comprehensive assessment of the model's performance. An F1-score closer to 1 indicates better overall performance, considering both false positive and false negative predictions.

# In[63]:


conf_matrix = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:\n", conf_matrix)


# In[64]:


ConfusionMatrixDisplay(confusion_matrix(y_test,y_pred)).plot()


# In[65]:


class_report = classification_report(y_test, y_pred)

print("\nClassification Report:\n", class_report)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Model Building

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


knn = KNeighborsClassifier()


# In[ ]:


knn.fit(X_train,y_train)


# In[ ]:


y_pred = knn.predict(X_test.values)


# In[ ]:





# In[ ]:





# In[ ]:




