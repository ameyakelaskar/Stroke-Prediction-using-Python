#!/usr/bin/env python
# coding: utf-8

# In[220]:


#Analysis of Stroke Dataset and Building Logistic Regression and K Means Clustering Models

#By Ameya Rajesh Kelaskar



import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,explained_variance_score,confusion_matrix,accuracy_score,classification_report
from math import sqrt
from sklearn.cluster import KMeans, k_means
get_ipython().run_line_magic('matplotlib', 'inline')



#Import dataset
df = pd.read_csv('/Users/ameyakelaskar/Downloads/healthcare-dataset-stroke-data.csv')

print(df)

print(df.shape)

print(df.info())

print(df.head())

#Datatypes which are 'object' can be converted into categorical for now, later we will change them to numerical during feature selection
#Nulls in bmi column need to be imputed 

#We can drop id column as it is of no apparent use

df=df.drop('id',axis=1)

df


# In[221]:


#Replace nulls in bmi

df['bmi'].fillna((df['bmi'].mean()), inplace=True)

df.info()


# In[222]:


#Convert object datatypes to categorical

for i in ['gender','ever_married','work_type','Residence_type','smoking_status']:
     df[i] = df[i].astype('category')

df.info()


# In[223]:


#Investigate all elements within each feature

for column in df:
    unique_values=np.unique(df[column])
    nr_values=len(unique_values)
    if nr_values<10:
            print("The number of values for feature {} is: {} --{} ".format(column,nr_values,unique_values))
    else:
        print("The number of values for feature {} is {}".format(column,nr_values))
        
#From here we get an idea that average glucose level and bmi might not be important features


# In[225]:



#Seaborn pairplots

g = sns.pairplot(df[['age','bmi','avg_glucose_level']])
plt.show()
#Boxplots
sns.boxplot(data=df,x='stroke',y='bmi',)
plt.title('Stroke vs BMI')
plt.xlabel('Stroke')
plt.ylabel('BMI')
plt.show()

#The prople who had stroke seem to have a higher BMI

sns.boxplot(data=df,x='stroke',y='age')
plt.title('Stroke vs Age')
plt.xlabel('Stroke')
plt.ylabel('Age')
plt.show()
#The prople who had stroke seem to be much older than those who did not

sns.boxplot(data=df,x='stroke',y='avg_glucose_level')
plt.title('Stroke vs Glucose level')
plt.xlabel('Stroke')
plt.ylabel('Average Glucose Level')
plt.show()

fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(df.corr(),annot=True,ax=ax)
plt.title('Heatmap')


sns.displot(x='age', hue='stroke', data=df, alpha=0.6)
plt.show()

stroke = df[df['stroke']==1]
sns.displot(stroke.age, kind='kde')
plt.show()

sns.displot(stroke.age, kind='ecdf')
plt.grid(True)
plt.show()


#Age is clearly linked with stroke

features = ['gender', 'hypertension', 'heart_disease', 'ever_married',
       'work_type', 'Residence_type',
       'smoking_status']

for f in features:
 sns.countplot(x=f,data=df,hue='stroke')
 plt.show()


# In[172]:


#Find outliers based on bmi column using percentile

min_thresh,max_thresh=df.bmi.quantile([0.001,0.999]) #Calcuated threshold values using percentile

print('The minimum threshold is '+str(min_thresh)) #Prints the value of minimum threshold
print('The max threshold is '+str(max_thresh)) #Prints the value of maximum threshold

#New DF with ouliers removed

df2=df[(df.bmi>min_thresh) & (df.bmi<max_thresh)] #Creates a new dataframe with the outliers removed based on bmi column

print('The dataframe with outliers removed is: \n'+ str(df2)) #Prints the new dataframe with outliers removed

##impute outliers using median 

median = df2['bmi'].median()  #median of bmi column of new dataframe without outliers

df.loc[df.bmi < min_thresh, 'bmi'] = np.nan #Replace outliers in original dataframe with nan values
df.loc[df.bmi > max_thresh, 'bmi'] = np.nan #Replace outliers in original dataframe with nan values

df.bmi.fillna(median,inplace=True) #Fill the nan values with the median

#Original dataframe with outliers imputed with median

print('Original dataframe with outliers imputed with median is: \n'+str(df))


# In[173]:


##Investigate distribution of 'stroke'

sns.countplot(x='stroke',data=df)
plt.show()

#We can clearly see that the stroke variable is imbalanced 
#This will create problems in model formation
#Therefore, we will use oversampling to balance the dependent variable


# In[174]:


count_class_0,count_class_1=df.stroke.value_counts()

df_class_0=df[df['stroke']==0]
df_class_1=df[df['stroke']==1]


# In[175]:


df_class_1_over=df_class_1.sample(count_class_0,replace=True)

df_over=pd.concat((df_class_0,df_class_1_over),axis=0)

print(df_over.stroke.value_counts())

df_over #over sampled dataframe

sns.countplot(x='stroke',data=df_over)
plt.show()

#As we can see, the stroke variable has been balanced


# In[176]:


df.columns


# In[205]:



#Visualize data after balancing to find patterns

sns.boxplot(data=df,x='stroke',y='bmi',)
plt.title('Stroke vs BMI')
plt.xlabel('Stroke')
plt.ylabel('BMI')
plt.show()

#The prople who had stroke seem to have a higher BMI

sns.boxplot(data=df,x='stroke',y='age')
plt.title('Stroke vs Age')
plt.xlabel('Stroke')
plt.ylabel('Age')
plt.show()
#The prople who had stroke seem to be much older than those who did not

sns.boxplot(data=df,x='stroke',y='avg_glucose_level')
plt.title('Stroke vs Glucose level')
plt.xlabel('Stroke')
plt.ylabel('Average Glucose Level')
plt.show()


#Looping through all features by stroke variable and Visualize to check for relationships


features = ['gender', 'hypertension', 'heart_disease', 'ever_married',
       'work_type', 'Residence_type',
       'smoking_status']

for f in features:
 sns.countplot(x=f,data=df_over,hue='stroke')
 plt.show()
    
#More females have had strokes than males, however more percentage of males have had stroke

#Hypertension and heart disease seem to be correlated with stroke

#Married people are much more likely to have had a stroke

#Private and self employed categories have a higher rate of stroke

#Urban population has a slightly higher stroke rate

#Smoking habits are correlated to stroke


# In[178]:


#Convert categorical variables into numeric

new_df_over = pd.get_dummies(df_over,columns=features)
new_df_over


# In[179]:


#Feature selection

X=new_df_over.drop('stroke',axis=1).values
y=new_df_over['stroke']

print(X.shape)
print(y.shape)


# In[180]:


#Run a tree-based estimator

dt= DecisionTreeClassifier(random_state=15,criterion='entropy',max_depth=10)
dt.fit(X,y)


# In[181]:


#Running feature importance

fi_col=[]
fi=[]
for i,column in enumerate(new_df_over.drop('stroke',axis=1)):
    print("The feature importance for {} is: {}".format(column,dt.feature_importances_[i]))
    fi_col.append(column)
    fi.append(dt.feature_importances_[i])


# In[182]:


fi_df=zip(fi_col,fi)
fi_df=pd.DataFrame(fi_df,columns=['Feature','Feature Importance'])
print(fi_df)

#Ordering the data

fi_df=fi_df.sort_values('Feature Importance',ascending=False).reset_index()
print(fi_df)
#Create columns to keep

columns_to_keep=fi_df['Feature'][0:19]

columns_to_keep


# In[183]:


#Print shapes

print(new_df_over.shape)
print(new_df_over[columns_to_keep].shape)


# In[184]:


#Split the data into X and y

X = new_df_over[columns_to_keep].values
y = new_df_over['stroke']
y = y.astype(int)

print(X.shape)
print(y.shape)


# In[185]:


#Hold-out validation

#Train test split, 80%-20%

X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=15)

#Hold-out sample, 10%

X_train,X_valid,y_train,y_valid = train_test_split(X_train,y_train,train_size=0.9,test_size=0.1,random_state=15)

print(X_train.shape)
print(X_test.shape)
print(X_valid.shape)

print(y_train.shape)
print(y_test.shape)
print(y_valid.shape)


# In[186]:


#Investigation of all the y variables to check if there's an imbalance

ax = sns.countplot(x=y_train)
plt.show()

ax = sns.countplot(x=y_test)
plt.show()

ax = sns.countplot(x=y_valid)
plt.show()


# In[187]:


#Logistic Regression

#Training the model

lr = LogisticRegression(random_state=10,solver='lbfgs',max_iter=10000)
lr.fit(X_train,y_train)


# In[188]:


#Predict Class labels for samples in X

y_pred = lr.predict(X_train)
print(y_pred)


# In[255]:


#Evaluating the model

#Accuracy on training data

print("The Training Accuracy is: ", lr.score(X_train, y_train))

print("The test Accuracy is: ", lr.score(X_test, y_test))

print(classification_report(y_train,y_pred))

#The metrics indicate that the model is good


# In[190]:


# Confusion Matrix function

def plot_confusion_matrix(cm, classes=None, title='Confusion matrix'):
    """Plots a confusion matrix."""
    if classes is not None:
        sns.heatmap(cm, cmap="YlGnBu", xticklabels=classes, yticklabels=classes, vmin=0., vmax=1., annot=True, annot_kws={'size':50})
    else:
        sns.heatmap(cm, vmin=0., vmax=1.)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[191]:


# Visualizing the confusion matrix

cm = confusion_matrix(y_train, y_pred)
cm_norm = cm / cm.sum(axis=1).reshape(-1,1)

plot_confusion_matrix(cm_norm, classes = lr.classes_, title='Confusion matrix')


# In[192]:


# Calculating False Positives (FP), False Negatives (FN), True Positives (TP) & True Negatives (TN)

FP = cm.sum(axis=0) - np.diag(cm)
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)
TN = cm.sum() - (FP + FN + TP)


# Sensitivity, hit rate, recall, or true positive rate
TPR = TP / (TP + FN)
print("The True Positive Rate is:", TPR)

# Precision or positive predictive value
PPV = TP / (TP + FP)
print("The Precision is:", PPV)

# False positive rate or False alarm rate
FPR = FP / (FP + TN)
print("The False positive rate is:", FPR)


# False negative rate or Miss Rate
FNR = FN / (FN + TP)
print("The False Negative Rate is: ", FNR)



##Total averages :
print("")
print("The average TPR is:", TPR.sum()/2)
print("The average Precision is:", PPV.sum()/2)
print("The average False positive rate is:", FPR.sum()/2)
print("The average False Negative Rate is:", FNR.sum()/2)


# In[195]:


#We will use a dummy classifer to check if our model performs better than it
# Training a Dummy Classifier

from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
score = dummy_clf.score(X_test, y_test)


print("Testing Acc:", score)

#As per the accuracy score, the model performs better than the dummy classifier


# In[196]:


#Testing the model on a hold-out sample
#Earlier, we had kept 10% of the data as a hold-out sample
#Now we will test the model performanec on the hold-out sample

lr = LogisticRegression(random_state=10, solver = 'lbfgs',max_iter=10000)
lr.fit(X_train, y_train)
score = lr.score(X_valid, y_valid)


print("Testing Acc:", score)

#The model has good accuracy even on hold-out data


# In[117]:


#K-Means

X_train = new_df_over.values

#Finding the ideal number of Ks

no_of_clusters = range(2,20)
inertia=[]

for f in no_of_clusters:
    kmeans=KMeans(n_clusters=f,random_state=2)
    kmeans = kmeans.fit(X_train)
    u = kmeans.inertia_
    inertia.append(u)
    print("The inertia for",f,"Clusters is: ",u)


# In[123]:


#Plot inertia to determine ideal K value (Elbow Method)

fig, (ax1) = plt.subplots(1,figsize=(16,6))
xx = np.arange(len(no_of_clusters))
ax1.plot(xx,inertia)
ax1.set_xticks(xx)
ax1.set_xticklabels(no_of_clusters,rotation='vertical')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia Score')
plt.title('Inertia per K')

#From the plot, we see that K=5 can be used for clustering


# In[124]:


#Running K-Means on 5 clusters

kmeans=KMeans(n_clusters=5,random_state=2)
kmeans = kmeans.fit(X_train)

#Predictions for new data

predictions= kmeans.predict(X_train)

#Calculating cluster counts

unique, counts = np.unique(predictions, return_counts=True)
counts=counts.reshape(1,5)

#Creating a dataframe

countscldf = pd.DataFrame(counts,columns=['Cluster 1','Cluster 2','Cluster 3','Cluster 4','Cluster 5'])

countscldf


# In[236]:


#K-Means

#I will use two features so that I can visualize the clusters

#Age and BMI

X = new_df_over[['age','bmi']].copy()

#Finding the ideal number of Ks

no_of_clusters = range(2,20)
inertia=[]

for f in no_of_clusters:
    kmeans=KMeans(n_clusters=f,random_state=2)
    kmeans = kmeans.fit(X)
    u = kmeans.inertia_
    inertia.append(u)
    print("The inertia for",f,"Clusters is: ",u)


# In[237]:


#Plot inertia to determine ideal K value (Elbow Method)

fig, (ax1) = plt.subplots(1,figsize=(16,6))
xx = np.arange(len(no_of_clusters))
ax1.plot(xx,inertia)
ax1.set_xticks(xx)
ax1.set_xticklabels(no_of_clusters,rotation='vertical')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia Score')
plt.title('Inertia per K')

#From the plot, we see that K=5 can be used for clustering


# In[238]:


#Running K-Means on 4 clusters

kmeans=KMeans(n_clusters=4,random_state=2)
kmeans = kmeans.fit(X)

#Predictions for new data

predictions= kmeans.predict(X)

#Calculating cluster counts

unique, counts = np.unique(predictions, return_counts=True)
counts=counts.reshape(1,4)

#Creating a dataframe

countscldf = pd.DataFrame(counts,columns=['Cluster 1','Cluster 2','Cluster 3','Cluster 4'])

countscldf


# In[245]:


#Visualizing all the clusters 

X['Clusters']=predictions

print(X)

X1=X[X.Clusters==0]
X2=X[X.Clusters==1]
X3=X[X.Clusters==2]
X4=X[X.Clusters==3]
plt.scatter(X1.age,X1.bmi,color='green')
plt.scatter(X2.age,X2.bmi,color='red')
plt.scatter(X3.age,X3.bmi,color='blue')
plt.scatter(X4.age,X4.bmi,color='orange')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],marker='*',color='black',label='centroid')
plt.title('Age vs BMI')
plt.xlabel('Age')
plt.ylabel('BMI')

#Thus we see that people in the age range of approx 30-30 have greater chance of having BMI over the average


# In[246]:


#K-Means

#I will use two features so that I can visualize the clusters
#Age and average glucose level

X = new_df_over[['age','avg_glucose_level']].copy()

#Finding the ideal number of Ks

no_of_clusters = range(2,20)
inertia=[]

for f in no_of_clusters:
    kmeans=KMeans(n_clusters=f,random_state=2)
    kmeans = kmeans.fit(X)
    u = kmeans.inertia_
    inertia.append(u)
    print("The inertia for",f,"Clusters is: ",u)


# In[247]:


#Plot inertia to determine ideal K value (Elbow Method)

fig, (ax1) = plt.subplots(1,figsize=(16,6))
xx = np.arange(len(no_of_clusters))
ax1.plot(xx,inertia)
ax1.set_xticks(xx)
ax1.set_xticklabels(no_of_clusters,rotation='vertical')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia Score')
plt.title('Inertia per K')

#From the plot, we see that K=5 can be used for clustering


# In[248]:


#Running K-Means on 4 clusters

kmeans=KMeans(n_clusters=4,random_state=2)
kmeans = kmeans.fit(X)

#Predictions for new data

predictions= kmeans.predict(X)

#Calculating cluster counts

unique, counts = np.unique(predictions, return_counts=True)
counts=counts.reshape(1,4)

#Creating a dataframe

countscldf = pd.DataFrame(counts,columns=['Cluster 1','Cluster 2','Cluster 3','Cluster 4'])

countscldf


# In[249]:


X['Clusters']=predictions

print(X)

X1=X[X.Clusters==0]
X2=X[X.Clusters==1]
X3=X[X.Clusters==2]
X4=X[X.Clusters==3]
plt.scatter(X1.age,X1.avg_glucose_level,color='green')
plt.scatter(X2.age,X2.avg_glucose_level,color='red')
plt.scatter(X3.age,X3.avg_glucose_level,color='blue')
plt.scatter(X4.age,X4.avg_glucose_level,color='orange')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],marker='*',color='black',label='centroid')
plt.title('Age vs Average Glucose Level')
plt.xlabel('Age')
plt.ylabel('Average Glucose level')

#Thus we see that people in the older age range tend to have higher glucose levels than people in the lower age groups


# In[250]:


#K-Means

#I will use two features so that I can visualize the clusters
#BMI and average glucose level 

X = new_df_over[['bmi','avg_glucose_level']].copy()

#Finding the ideal number of Ks

no_of_clusters = range(2,20)
inertia=[]

for f in no_of_clusters:
    kmeans=KMeans(n_clusters=f,random_state=2)
    kmeans = kmeans.fit(X)
    u = kmeans.inertia_
    inertia.append(u)
    print("The inertia for",f,"Clusters is: ",u)


# In[251]:


#Plot inertia to determine ideal K value (Elbow Method)

fig, (ax1) = plt.subplots(1,figsize=(16,6))
xx = np.arange(len(no_of_clusters))
ax1.plot(xx,inertia)
ax1.set_xticks(xx)
ax1.set_xticklabels(no_of_clusters,rotation='vertical')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia Score')
plt.title('Inertia per K')

#From the plot, we see that K=5 can be used for clustering


# In[252]:


#Running K-Means on 4 clusters

kmeans=KMeans(n_clusters=4,random_state=2)
kmeans = kmeans.fit(X)

#Predictions for new data

predictions= kmeans.predict(X)

#Calculating cluster counts

unique, counts = np.unique(predictions, return_counts=True)
counts=counts.reshape(1,4)

#Creating a dataframe

countscldf = pd.DataFrame(counts,columns=['Cluster 1','Cluster 2','Cluster 3','Cluster 4'])

countscldf


# In[253]:


X['Clusters']=predictions

print(X)

X1=X[X.Clusters==0]
X2=X[X.Clusters==1]
X3=X[X.Clusters==2]
X4=X[X.Clusters==3]
plt.scatter(X1.bmi,X1.avg_glucose_level,color='green')
plt.scatter(X2.bmi,X2.avg_glucose_level,color='red')
plt.scatter(X3.bmi,X3.avg_glucose_level,color='blue')
plt.scatter(X4.bmi,X4.avg_glucose_level,color='orange')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],marker='*',color='black',label='centroid')
plt.title('BMI vs Average Glucose Level')
plt.xlabel('BMI')
plt.ylabel('Average Glucose level')

#We see that people with higher BMI also tend to have higher glucose levels

