#!/usr/bin/env python
# coding: utf-8

# **Your challenge can be found toward the end of this notebook. The code below will be needed in order to begin the challenge. Read through and execute all necessary portions of this code to complete the tasks for this challenge.**

# ##### Import the necessary packages

# In[1]:


import numpy as np #numerical computation
import pandas as pd #data wrangling
import matplotlib.pyplot as plt #plotting package
#Next line helps with rendering plots
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl #add'l plotting functionality
mpl.rcParams['figure.dpi'] = 400 #high res figures
from IPython.display import Image #to visualize decision trees
import pickle

# ##### Cleaning the Dataset

# In[ ]:


df_orig = pd.read_excel('D:\EDUCATION\Internship_DataScience\Task_1\default_of_credit_card_clients.xls')


# In[ ]:


df_zero_mask = df_orig == 0


# In[ ]:


feature_zero_mask = df_zero_mask.iloc[:,1:].all(axis=1)


# In[ ]:


sum(feature_zero_mask)
# 315


# Remove all the rows with all zero features and response, confirm this that gets rid of the duplicate IDs.

# In[ ]:


df_clean = df_orig.loc[~feature_zero_mask,:].copy()


# In[ ]:


df_clean.shape
# (29685, 25)


# In[ ]:


df_clean['ID'].nunique()
# 29685


# Clean up the `EDUCATION` and `MARRIAGE` features as in Chapter 1

# In[ ]:


df_clean['EDUCATION'].value_counts()
# 2    13884
# 1    10474
# 3     4867
# 5      275
# 4      122
# 6       49
# 0       14
# Name: EDUCATION, dtype: int64


# "Education (1 = graduate school; 2 = university; 3 = high school; 4 = others)"

# Assign unknown categories to other.

# In[ ]:


df_clean['EDUCATION'].replace(to_replace=[0, 5, 6], value=4, inplace=True)


# In[ ]:


df_clean['EDUCATION'].value_counts()
# 2    13884
# 1    10474
# 3     4867
# 4      460
# Name: EDUCATION, dtype: int64


# Examine and clean marriage feature as well:

# In[ ]:


df_clean['MARRIAGE'].value_counts()
# 2    15810
# 1    13503
# 3      318
# 0       54
# Name: MARRIAGE, dtype: int64


# In[ ]:


#Should only be (1 = married; 2 = single; 3 = others).
df_clean['MARRIAGE'].replace(to_replace=0, value=3, inplace=True)


# In[ ]:


df_clean['MARRIAGE'].value_counts()
# 2    15810
# 1    13503
# 3      372
# Name: MARRIAGE, dtype: int64


# Now instead of removing rows with `PAY_1` = 'Not available', as done in Chapter 1, here select these out for addition to training and testing splits.

# In[ ]:


df_clean['PAY_1'].value_counts()
# 0                13087
# -1                5047
# 1                 3261
# Not available     3021
# -2                2476
# 2                 2378
# 3                  292
# 4                   63
# 5                   23
# 8                   17
# 6                   11
# 7                    9
# Name: PAY_1, dtype: int64


# In[ ]:


missing_pay_1_mask = df_clean['PAY_1'] == 'Not available'


# In[ ]:


sum(missing_pay_1_mask)
# 3021


# In[ ]:


df_missing_pay_1 = df_clean.loc[missing_pay_1_mask,:].copy()


# In[ ]:


df_missing_pay_1.shape
# (3021, 25)


# In[ ]:


df_missing_pay_1['PAY_1'].head(3)


# In[ ]:


df_missing_pay_1['PAY_1'].value_counts()


# In[ ]:


df_missing_pay_1.columns


# Load cleaned data

# In[ ]:


df = pd.read_csv('D:\EDUCATION\Internship_DataScience\Task_1\Data_Exploration_and_Cleaning\cleaned_data.csv')


# In[ ]:


df.columns


# In[ ]:


features_response = df.columns.tolist()


# In[ ]:


items_to_remove = ['ID', 'SEX', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                   'EDUCATION_CAT', 'graduate school', 'high school', 'none',
                   'others', 'university']


# In[ ]:


features_response = [item for item in features_response if item not in items_to_remove]
features_response


# ##### Mode and Random Imputation of `PAY_1`

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df[features_response[:-1]].values, df['default payment next month'].values,
test_size=0.2, random_state=24)


# In[ ]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
# (21331, 17)
# (5333, 17)
# (21331,)
# (5333,)


# In[ ]:


df_missing_pay_1.shape


# In[ ]:


features_response[4]


# In[ ]:


np.median(X_train[:,4])


# In[ ]:


np.random.seed(seed=1)
fill_values = [0, np.random.choice(X_train[:,4], size=(3021,), replace=True)]


# In[ ]:


fill_strategy = ['mode', 'random']


# In[ ]:


fill_values[-1]


# In[ ]:


fig, axs = plt.subplots(1,2, figsize=(8,3))
bin_edges = np.arange(-2,9)
axs[0].hist(X_train[:,4], bins=bin_edges, align='left')
axs[0].set_xticks(bin_edges)
axs[0].set_title('Non-missing values of PAY_1')
axs[1].hist(fill_values[-1], bins=bin_edges, align='left')
axs[1].set_xticks(bin_edges)
axs[1].set_title('Random selection for imputation')
plt.tight_layout()


# To do cross-validation on the training set, now we need to shuffle since all the samples with missing `PAY_1` were concatenated on to the end.

# In[ ]:


from sklearn.model_selection import KFold


# In[ ]:


k_folds = KFold(n_splits=4, shuffle=True, random_state=1)


# Don't need to do a grid search, so we can use `cross_validate`

# In[ ]:


from sklearn.model_selection import cross_validate


# For the estimator, set the optimal hyperparameters determined in previous chapter.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rf = RandomForestClassifier(n_estimators=200, criterion='gini', max_depth=9,
min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None,
random_state=4, verbose=1, warm_start=False, class_weight=None)


# In[ ]:


for counter in range(len(fill_values)):
    #Copy the data frame with missing PAY_1 and assign imputed values
    df_fill_pay_1_filled = df_missing_pay_1.copy()
    df_fill_pay_1_filled['PAY_1'] = fill_values[counter]
    
    #Split imputed data in to training and testing, using the same
    #80/20 split we have used for the data with non-missing PAY_1
    X_fill_pay_1_train, X_fill_pay_1_test, y_fill_pay_1_train, y_fill_pay_1_test =     train_test_split(
        df_fill_pay_1_filled[features_response[:-1]].values,
        df_fill_pay_1_filled['default payment next month'].values,
    test_size=0.2, random_state=24)
    
    #Concatenate the imputed data with the array of non-missing data
    X_train_all = np.concatenate((X_train, X_fill_pay_1_train), axis=0)
    y_train_all = np.concatenate((y_train, y_fill_pay_1_train), axis=0)
    
    #Use the KFolds splitter and the random forest model to get
    #4-fold cross-validation scores for both imputation methods
    imputation_compare_cv = cross_validate(rf, X_train_all, y_train_all, scoring='roc_auc',
                                       cv=k_folds, n_jobs=-1, verbose=1,
                                       return_train_score=True, return_estimator=True,
                                       error_score='raise-deprecating')
    
    test_score = imputation_compare_cv['test_score']
    print(fill_strategy[counter] + ' imputation: ' +
          'mean testing score ' + str(np.mean(test_score)) +
          ', std ' + str(np.std(test_score)))


# ##### A Predictive Model for `PAY_1`

# In[ ]:


pay_1_df = df.copy()


# In[ ]:


features_for_imputation = pay_1_df.columns.tolist()


# In[ ]:


items_to_remove_2 = ['ID', 'SEX', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                   'EDUCATION_CAT', 'graduate school', 'high school', 'none',
                   'others', 'university', 'default payment next month', 'PAY_1']


# In[ ]:


features_for_imputation = [item for item in features_for_imputation if item not in items_to_remove_2]
features_for_imputation


# ##### Building a Multiclass Classification Model for Imputation

# In[ ]:


X_impute_train, X_impute_test, y_impute_train, y_impute_test = train_test_split(
    pay_1_df[features_for_imputation].values,
    pay_1_df['PAY_1'].values,
test_size=0.2, random_state=24)


# In[ ]:


rf_impute_params = {'max_depth':[3, 6, 9, 12],
             'n_estimators':[10, 50, 100, 200]}


# In[ ]:


from sklearn.model_selection import GridSearchCV


# Need to use accuracy here as ROC AUC is not supported for multiclass. Need to use multiclass and not regression because need to limit to integer values of `PAY_1`.

# In[ ]:


cv_rf_impute = GridSearchCV(rf, param_grid=rf_impute_params, scoring='accuracy',
                            n_jobs=-1, iid=False, refit=True,
                            cv=4, verbose=2, error_score=np.nan, return_train_score=True)


# In[ ]:


cv_rf_impute.fit(X_impute_train, y_impute_train)


# In[ ]:


impute_df = pd.DataFrame(cv_rf_impute.cv_results_)
impute_df


# In[ ]:


cv_rf_impute.best_params_
# {'max_depth': 12, 'n_estimators': 100}


# In[ ]:


cv_rf_impute.best_score_
# 0.7337676389523727


# In[ ]:


pay_1_value_counts = pay_1_df['PAY_1'].value_counts().sort_index()


# In[ ]:


pay_1_value_counts


# In[ ]:


pay_1_value_counts/pay_1_value_counts.sum()


# In[ ]:


y_impute_predict = cv_rf_impute.predict(X_impute_test)


# In[ ]:


from sklearn import metrics


# In[ ]:


metrics.accuracy_score(y_impute_test, y_impute_predict)


# In[ ]:


fig, axs = plt.subplots(1,2, figsize=(8,3))
axs[0].hist(y_impute_test, bins=bin_edges, align='left')
axs[0].set_xticks(bin_edges)
axs[0].set_title('Non-missing values of PAY_1')
axs[1].hist(y_impute_predict, bins=bin_edges, align='left')
axs[1].set_xticks(bin_edges)
axs[1].set_title('Model-based imputation')
plt.tight_layout()


# In[ ]:


X_impute_all = pay_1_df[features_for_imputation].values
y_impute_all = pay_1_df['PAY_1'].values


# In[ ]:


rf_impute = RandomForestClassifier(n_estimators=100, max_depth=12)


# In[ ]:


rf_impute


# In[ ]:


rf_impute.fit(X_impute_all, y_impute_all)


# ##### Using the Imputation Model and Comparing it to Other Methods

# In[ ]:


df_fill_pay_1_model = df_missing_pay_1.copy()


# In[ ]:


df_fill_pay_1_model['PAY_1'].head()


# In[ ]:


df_fill_pay_1_model['PAY_1'] = rf_impute.predict(df_fill_pay_1_model[features_for_imputation].values)


# In[ ]:


df_fill_pay_1_model['PAY_1'].head()


# In[ ]:


df_fill_pay_1_model['PAY_1'].value_counts().sort_index()


# In[ ]:


X_fill_pay_1_train, X_fill_pay_1_test, y_fill_pay_1_train, y_fill_pay_1_test = train_test_split(
    df_fill_pay_1_model[features_response[:-1]].values,
    df_fill_pay_1_model['default payment next month'].values,
test_size=0.2, random_state=24)


# In[ ]:


print(X_fill_pay_1_train.shape)
print(X_fill_pay_1_test.shape)
print(y_fill_pay_1_train.shape)
print(y_fill_pay_1_test.shape)


# In[ ]:


X_train_all = np.concatenate((X_train, X_fill_pay_1_train), axis=0)
y_train_all = np.concatenate((y_train, y_fill_pay_1_train), axis=0)


# In[ ]:


print(X_train_all.shape)
print(y_train_all.shape)


# In[ ]:


rf


# In[ ]:


imputation_compare_cv = cross_validate(rf, X_train_all, y_train_all, scoring='roc_auc',
                                       cv=k_folds, n_jobs=-1, verbose=1,
                                       return_train_score=True, return_estimator=True,
                                       error_score='raise-deprecating')


# In[ ]:


imputation_compare_cv['test_score']
# array([0.76890992, 0.77309591, 0.77166336, 0.77703366])


# In[ ]:


np.mean(imputation_compare_cv['test_score'])
# 0.7726757126815554


# In[ ]:


np.std(imputation_compare_cv['test_score'])
# 0.002931480680760725


# Reassign values using mode imputation

# In[ ]:


df_fill_pay_1_model['PAY_1'] = np.zeros_like(df_fill_pay_1_model['PAY_1'].values)


# In[ ]:


df_fill_pay_1_model['PAY_1'].unique()


# In[ ]:


X_fill_pay_1_train, X_fill_pay_1_test, y_fill_pay_1_train, y_fill_pay_1_test = train_test_split(
    df_fill_pay_1_model[features_response[:-1]].values,
    df_fill_pay_1_model['default payment next month'].values,
test_size=0.2, random_state=24)


# In[ ]:


X_train_all = np.concatenate((X_train, X_fill_pay_1_train), axis=0)
X_test_all = np.concatenate((X_test, X_fill_pay_1_test), axis=0)
y_train_all = np.concatenate((y_train, y_fill_pay_1_train), axis=0)
y_test_all = np.concatenate((y_test, y_fill_pay_1_test), axis=0)


# In[ ]:


print(X_train_all.shape)
print(X_test_all.shape)
print(y_train_all.shape)
print(y_test_all.shape)


# In[ ]:


imputation_compare_cv = cross_validate(rf, X_train_all, y_train_all, scoring='roc_auc',
                                       cv=k_folds, n_jobs=-1, verbose=1,
                                       return_train_score=True, return_estimator=True,
                                       error_score='raise-deprecating')


# In[ ]:


np.mean(imputation_compare_cv['test_score'])


# ##### Confirming Model Performance on the Unseen Test Set

# In[ ]:


rf.fit(X_train_all, y_train_all)


# In[ ]:


y_test_all_predict_proba = rf.predict_proba(X_test_all)


# In[ ]:


from sklearn.metrics import roc_auc_score


# In[ ]:


roc_auc_score(y_test_all, y_test_all_predict_proba[:,1])
# 0.7696243835824927


# ##### Characterizing Costs and Savings

# In[ ]:


thresholds = np.linspace(0, 1, 101)


# Use mean bill amount to estimate savings per prevented default

# In[ ]:


df[features_response[:-1]].columns[5]


# In[ ]:


savings_per_default = np.mean(X_test_all[:, 5])
savings_per_default
# 51601.7433479286


# In[ ]:


cost_per_counseling = 7500


# In[ ]:


effectiveness = 0.70


# In[ ]:


n_pos_pred = np.empty_like(thresholds)
cost_of_all_counselings = np.empty_like(thresholds)
n_true_pos = np.empty_like(thresholds)
savings_of_all_counselings = np.empty_like(thresholds)


# In[ ]:


counter = 0
for threshold in thresholds:
    pos_pred = y_test_all_predict_proba[:,1]>threshold
    n_pos_pred[counter] = sum(pos_pred)
    cost_of_all_counselings[counter] = n_pos_pred[counter] * cost_per_counseling
    true_pos = pos_pred & y_test_all.astype(bool)
    n_true_pos[counter] = sum(true_pos)
    savings_of_all_counselings[counter] = n_true_pos[counter] * savings_per_default * effectiveness
    
    counter += 1


# In[ ]:


net_savings = savings_of_all_counselings - cost_of_all_counselings


# In[ ]:


# plt.plot(thresholds, cost_of_all_counselings)


# In[ ]:


# plt.plot(thresholds, savings_of_all_counselings)


# In[ ]:


mpl.rcParams['figure.dpi'] = 400
plt.plot(thresholds, net_savings)
plt.xlabel('Threshold')
plt.ylabel('Net savings (NT$)')
plt.xticks(np.linspace(0,1,11))
plt.grid(True)


# In[ ]:


max_savings_ix = np.argmax(net_savings)


# What is the threshold at which maximum savings is achieved?

# In[ ]:


thresholds[max_savings_ix]
# 0.2


# What is the maximum possible savings?

# In[ ]:


net_savings[max_savings_ix]
# 15446325.35991916


# ## Challenge: Deriving Financial Insights

#     Everything that is needed prior to this challenge has been included in the notebook above. You should run all the necessary portions of the provided code before beginning these tasks.

# In[ ]:


# This will autosave your notebook every ten seconds
get_ipython().run_line_magic('autosave', '10')


# **Using the testing set, calculate the cost of all defaults if there were no counseling program and output your result.**

# In[ ]:


sum(y_test_all)


# _______________________________________________________________________________________________
# **Next, calculate by what percent can the cost of defaults be decreased by the counseling program and output you result.**

# In[ ]:


cost_of_all_defaults=sum(y_test_all*savings_per_default)
cost_of_all_defaults


# _______________________________________________________________________________________________
# **Then, calculate the net savings per account at the optimal threshold and output your result.**

# In[ ]:


percentage_decrease=(net_savings[max_savings_ix]/cost_of_all_defaults)*100
percentage_decrease


# _______________________________________________________________________________________________
# **Now, plot the net savings per account against the cost of counseling per account for each threshold.**

# In[ ]:


len(y_test_all)


# _______________________________________________________________________________________________
# **Next, plot the fraction of accounts predicted as positive (this is called the "flag rate") at each threshold.**

# In[ ]:


mpl.rcParams['figure.dpi'] = 400
cost_of_counseling_per_account=cost_of_all_counselings/len(y_test_all)
net_savings_per_account = net_savings/len(y_test_all)
plt.plot(cost_of_counseling_per_account,net_savings_per_account)
plt.xlabel('Net Savings per account')
plt.ylabel('Cost of Counselling per account')
plt.grid(True)


# _______________________________________________________________________________________________
# **Next, plot a precision-recall curve for the testing data.**

# In[ ]:


plt.plot(n_pos_pred,thresholds)
plt.xlabel('Flag rate')
plt.ylabel('Threshold')
plt.grid(True)


# _______________________________________________________________________________________________
# **Finally, plot precision and recall separately on the y-axis against threshold on the x-axis.**

# In[ ]:


len(y_test_all)


# In[ ]:


# recall separately on the y-axis
recall = n_true_pos/sum(y_test_all)
precision=n_true_pos/n_pos_pred
plt.plot(precision,recall)
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.title('Precision-Recall')


# In[ ]:


#plot y-axis against threshold on the x-axis.
plt.plot(thresholds,precision,label='Precision')
plt.plot(thresholds,n_true_pos/sum(y_test_all),label='Recall')
plt.xlabel('Threshold')
plt.legend()
plt.show()


# In[ ]:

# saving model to disk
pickle.dump(rf, open('Deriving_Finanicial_Insights.pkl','wb'))




# In[ ]:

Deriving_Finanicial_Insights = pickle.load(open('Deriving_Finanicial_Insights.pkl','rb'))
print(Deriving_Finanicial_Insights.predict([[]]))



