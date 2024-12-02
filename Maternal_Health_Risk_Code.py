#!/usr/bin/env python
# coding: utf-8

# In[130]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().system('pip install category_encoders')
from scipy import stats
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.tree import export_text, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import roc_curve, accuracy_score, precision_score, recall_score


# In[131]:


maternal = pd.read_csv('Maternal_Health_Risk.csv')
maternal.head()


# In[132]:


maternal = pd.read_csv('Maternal_Health_Risk.csv')
summary_statistics = maternal.describe()
print(summary_statistics)


# In[133]:


plt.figure(figsize=(12, 6))
sns.boxplot(data=maternal[['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate', 'RiskLevel']])
plt.title('Boxplot of Numeric Features')
plt.show()


# In[134]:


# Using the Z-Score to identify outliers in the columns
numerical_columns = maternal.select_dtypes(include=['float64', 'int64']).columns
z_scores = stats.zscore(maternal[numerical_columns])
outliers = (np.abs(z_scores) > 3).any(axis=1)

# Calculating number of outliers
outliers_number = outliers.sum()
print("Outliers:", outliers_number)

# Features with outliers
outlier_mask = np.abs(z_scores) > 3
outlier_columns = outlier_mask.any()

print("Outliers Status")
print(outlier_columns)


# The Graphs below showcase the three columns containing outliers

# In[135]:


#Creating scatter plots graphs for columns with outliers:
columns_with_outliers = maternal[outlier_columns.index[outlier_columns]]
for column in columns_with_outliers:
    
    plt.figure(figsize=(8, 6))
    
    plt.scatter(maternal.index[~outlier_mask[column]], maternal[column][~outlier_mask[column]], marker='o', label='Non-Outliers', color='blue')
    plt.scatter(maternal.index[outlier_mask[column]], maternal[column][outlier_mask[column]], marker='x', label='Outliers', color='red')
    
    plt.xlabel('Index')
    plt.ylabel(column)
    plt.title(f'Scatter Plot for {column} With Outliers')
    
    plt.legend()
    plt.show()


# In[136]:


# Calculating z-scores for HeartRate
heartrate_z_scores = stats.zscore(maternal['HeartRate'])

# Identifying the outliers in HeartRate
heartrate_outliers = np.abs(heartrate_z_scores) > 3

# Removing outliers from HeartRate
cleaned_maternal = maternal[~heartrate_outliers]

# Using the Z-Score to identify outliers in the columns
numerical_columns = cleaned_maternal.select_dtypes(include=['float64', 'int64']).columns
z_scores = stats.zscore(cleaned_maternal[numerical_columns])

# This identifies the outliers
outliers = (np.abs(z_scores) > 3).any(axis=1)

# Calculates the number of outliers
outliers_number = outliers.sum()
print("Outliers:", outliers_number)

# Features with outliers
outlier_columns = (np.abs(z_scores) > 3).any(axis=0)

print("Outliers Status")
for column, status in outlier_columns.items():
    print(f"{column.ljust(12)} {status}")


# In[137]:


plt.figure(figsize=(10, 6))
plt.scatter(range(len(cleaned_maternal)), cleaned_maternal['HeartRate'], color='blue', label='HeartRate')
plt.title('Scatter Plot of HeartRate after Removing Outliers')
plt.xlabel('Index')
plt.ylabel('HeartRate')
plt.legend()
plt.show()


# In[138]:


#Checking the number of duplicate rows
duplicates = maternal.duplicated()

num_duplicates = duplicates.sum()

if num_duplicates > 0:
    print(f"Number of duplicate rows: {num_duplicates}")
    print("Duplicate rows:")
    print(maternal[duplicates])
else:
    print("No duplicate rows found.") 


# Since this dataset does not contain any missing value, there is no need to check for them. 

# In[139]:


# Check data types
print("\nData Types:")
print(maternal.dtypes)


# In[140]:


print(maternal['RiskLevel'].unique())


# In[141]:


plt.figure(figsize=(12, 6))
sns.boxplot(data=maternal)
plt.xticks(rotation=45)
plt.title('Boxplot of Maternal Health Risk Dataset')
plt.show()


# In[142]:


# Histograms for the numerical columns
plt.figure(figsize=(12, 6))
maternal.hist(bins=20, figsize=(12, 10))
plt.suptitle('Histograms of Maternal Health Risk Dataset')
plt.show()


# In[143]:


risk_level_counts = maternal['RiskLevel'].value_counts()

# Forming a Bar Plot
plt.figure(figsize=(10, 6))
plt.bar(risk_level_counts.index, risk_level_counts.values, color='skyblue')
plt.xlabel('Risk Level')
plt.ylabel('Count')
plt.title('Distribution of Risk Levels')
plt.show()


# In[144]:


#NEED TO ADD GROUP COLUMNS
sns.axes_style("whitegrid")
sns.pairplot(data=maternal, hue="RiskLevel", height=3, diag_kind="kde", markers=["o", "s", "D"], palette="Set1")
plt.show()


# In[145]:


X = maternal[['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']]
y = maternal[['RiskLevel']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[146]:


maternal = DecisionTreeClassifier(random_state=47)

# Train Decision Tree Classifier
tdt = maternal.fit(X_train, y_train)
print('Number of tree nodes:', tdt.tree_.node_count) 
plt.figure(figsize=(30,15), dpi=250)
tree.plot_tree(maternal, fontsize=7)
plt.show()


# In[147]:


#10 fold cross-validation score:
cv= cross_val_score(maternal,X,y,cv=10)

#Tune 'max_depth' parameter.
maxdepth_cv=[]
node_counts=[]
for k in range(1,10,1):
    maternal=DecisionTreeClassifier(max_depth=k,random_state=47)
    maternal.fit(X_train,y_train)
    predict=maternal.predict(X_test)
    cv= cross_val_score(maternal,X, y,cv=10)
    nodecount = maternal.tree_.node_count
    print("max_depth={}".format(k), "Average 10-Fold CV Score:{}".format(np.mean(cv)),
          "Node count:{}".format(nodecount))
    maxdepth_cv.append(np.mean(cv))
    node_counts.append(nodecount)  
    
#Plot averaged CV scores for all max_depth tunings
fig,axes=plt.subplots(1,1,figsize=(8,5))
axes.set_xticks(range(1,10,1))
k=range(1,10,1)
plt.plot(k,maxdepth_cv)
plt.xlabel("max_depth")
plt.ylabel("Averaged 10-fold CV score")
plt.show()


# In[148]:


dt_opt=DecisionTreeClassifier(max_depth=9,random_state=47)
dt_opt_fit=dt_opt.fit(X_train,y_train)


predict_opt=dt_opt.predict(X_test)
print('Number of tree nodes after optimizing max depth: ', dt_opt_fit.tree_.node_count)
plt.figure(dpi=150)
tree.plot_tree(dt_opt,filled=True)
plt.title("Decision tree trained on all the Maternal Health Risk features using max depth=9")
plt.show()

acc_score=accuracy_score(y_test, predict_opt)
print("Accuracy score of our model with Decision Tree:", '%.2f'%acc_score)
precision = precision_score(y_true=y_test, y_pred=predict_opt, average='micro')
print("Precision score of our model with Decision Tree :", '%.2f'%precision)
recall = recall_score(y_true=y_test, y_pred=predict_opt, average='micro')
print("Recall score of our model with Decision Tree :", '%.2f'%recall)


# In[149]:


importances = pd.DataFrame({'Feature':X.columns,'Importance':np.round(maternal.feature_importances_,3)})
importances = importances.sort_values('Importance',ascending=False)
importances


# In[150]:


optimal_max_depth = np.argmax(maxdepth_cv) + 1

max_leaf_nodes_range = range(2, 11)
averaged_cv_scores = []

for max_leaf_nodes in max_leaf_nodes_range:
    dt_opt = DecisionTreeClassifier(max_depth=optimal_max_depth, max_leaf_nodes=max_leaf_nodes, random_state=47)
    cv_scores = cross_val_score(dt_opt, X, y, cv=10)
    avg_cv_score = np.mean(cv_scores)
    averaged_cv_scores.append(avg_cv_score)

plt.plot(max_leaf_nodes_range, averaged_cv_scores)
plt.xlabel('max_leaf_nodes')
plt.ylabel('Averaged CV Score')
plt.title('Averaged CV Scores for Different max_leaf_nodes Values')
plt.xticks(max_leaf_nodes_range)
plt.grid(True)
plt.show()

best_avg_cv_score = max(averaged_cv_scores)
best_max_leaf_nodes = max_leaf_nodes_range[np.argmax(averaged_cv_scores)]

print('Best max score:', best_avg_cv_score)
print('Best leaf node value:', best_max_leaf_nodes)


# In[151]:


dt_tuned = DecisionTreeClassifier(max_depth=optimal_max_depth, max_leaf_nodes=best_max_leaf_nodes, random_state=47)
dt_tuned.fit(X_train, y_train)
plt.figure(figsize=(12, 8))
plot_tree(dt_tuned, filled=True, feature_names=list(X.columns), class_names=[str(i) for i in dt_tuned.classes_])
plt.show()


# In[152]:


num_nodes_tuned = dt_tuned.tree_.node_count
print('Number of nodes in the tuned tree:', num_nodes_tuned)

y_pred = dt_tuned.predict(X_test)

acc_score=accuracy_score(y_test, predict_opt)
print('Accuracy score of our model with Decision Tree:', '%.2f'%acc_score)

precision = precision_score(y_true=y_test, y_pred=predict_opt, average='micro')
print('Precision score of our model with Decision Tree :', '%.2f'%precision)

recall = recall_score(y_true=y_test, y_pred=predict_opt, average='micro')
print('Recall score of our model with Decision Tree :', '%.2f'%recall)


# In[153]:


importances = pd.DataFrame({'Feature':X.columns,'Importance':np.round(maternal.feature_importances_,3)})
importances = importances.sort_values('Importance',ascending=False)
importances


# In[154]:


conf_matrix = confusion_matrix(y_test, predict_opt)

plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt ='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


# In[155]:


accuracy = accuracy_score(y_test, predict_opt)
precision = precision_score(y_test, predict_opt, average='weighted')
recall = recall_score(y_test, predict_opt, average='weighted')
f1 = f1_score(y_test, predict_opt, average='weighted')

classification_rep = classification_report(y_test, predict_opt)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
print("\nClassification Report:\n", classification_rep)


# In[ ]:




