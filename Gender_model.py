import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno 
import plotly.graph_objects as go 
import plotly.express as px 
# %matplotlib inline
import warnings
warnings.filterwarnings('ignore')

#Read the dataset/s

df = pd.read_csv('FBDt.csv')

df.shape

#(1)	The EDA starts with  understanding the basic description of data as described next:
#1. Checking description(first 5 and last 5 rows)
df.head()

df.tail()

print(df.shape)

df.columns

df.nunique()

df.info()

#3. Visualising data  distribution in detail
fig = plt.figure(figsize =(18,18))
ax=fig.gca()
df.hist(ax=ax,bins =30)
plt.show()

df.plot(kind='box', subplots=True,
layout=(2,7),sharex=False,sharey=False, figsize=(20, 10), color='deeppink');
#identify the outliers

continuous_features = ['age', 'dob_day', 'dob_year', 'dob_month', 'tenure', 'friend_count', 'friendships_initiated', 'likes', 'likes_received', 'mobile_likes', 'mobile_likes_received', 'www_likes', 'www_likes_received']

# Define function to identify and drop outliers
def outliers(df_out, drop=False):
    for each_feature in df_out.columns:
        feature_data = df_out[each_feature]
        Q1 = np.percentile(feature_data, 25.) 
        Q3 = np.percentile(feature_data, 75.) 
        IQR = Q3 - Q1 # Interquartile Range
        outlier_step = IQR * 1.5 # That's what we were talking about above
        outliers = feature_data[~((feature_data >= Q1 - outlier_step) & (feature_data <= Q3 + outlier_step))].index.tolist()
        if not drop:
            print('For the feature {}, No of Outliers is {}'.format(each_feature, len(outliers)))
        if drop:
            df.drop(outliers, inplace=True, errors='ignore')
            print('Outliers from {} feature removed'.format(each_feature))

# Identify and print outliers
outliers(df[continuous_features])

# Drop outliers
outliers(df[continuous_features], drop=True)

# Check if outliers were removed
df.plot(kind='box', subplots=True, layout=(2,7), sharex=False, sharey=False, figsize=(20,10), color='deeppink')

df.shape

#checking target value distribution
print(df.gender.value_counts())
fig, ax = plt.subplots(figsize=(7,4))
name = ["Male", "Female"]
ax = df.gender.value_counts().plot(kind='bar')
ax.set_title("Gender Classes", fontsize = 13, weight = 'bold')
ax.set_xticklabels (name, rotation = 0)

# To calculate the percentage
totals = []
for i in ax.patches:
    totals.append(i.get_height())
total = sum(totals)
for i in ax.patches:
    ax.text(i.get_x()+.09, i.get_height()-70, \
            str(round((i.get_height()/total)*100, 2))+'%', fontsize=14,
                color='white', weight = 'bold')
    
plt.tight_layout()

#check correlation between variables

#pre-processing
from sklearn.exceptions import DataDimensionalityWarning
#encode object columns to integers
from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder

for col in df:
  if df[col].dtype =='object':
    df[col]=OrdinalEncoder().fit_transform(df[col].values.reshape(-1,1))
df

class_label =df['gender']
df = df.drop(['gender'], axis =1)
df = (df-df.min())/(df.max()-df.min())
df['gender']=class_label
df

#pre-processing
pseudo_facebook = df.copy()
le = preprocessing.LabelEncoder()
userid = le.fit_transform(list(pseudo_facebook["userid"])) 
age = le.fit_transform(list(pseudo_facebook["age"])) 
dob_day = le.fit_transform(list(pseudo_facebook["dob_day"])) 
dob_year = le.fit_transform(list(pseudo_facebook["dob_year"])) 
dob_month = le.fit_transform(list(pseudo_facebook["dob_month"])) 
tenure = le.fit_transform(list(pseudo_facebook["tenure"])) 
friend_count= le.fit_transform(list(pseudo_facebook["friend_count"])) 
friendships_initiated = le.fit_transform(list(pseudo_facebook["friendships_initiated"]))
likes = le.fit_transform(list(pseudo_facebook["likes"]))
likes_received = le.fit_transform(list(pseudo_facebook["likes_received"]))
mobile_likes= le.fit_transform(list(pseudo_facebook["mobile_likes"]))
www_likes = le.fit_transform(list(pseudo_facebook["www_likes"]))
www_likes_received = le.fit_transform(list(pseudo_facebook["www_likes_received"]))
mobile_likes_received = le.fit_transform(list(pseudo_facebook["mobile_likes_received"]))

G = le.fit_transform(list(pseudo_facebook["gender"]))

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

x = list(zip(userid, age, dob_day, dob_year, dob_month, tenure,
       friend_count, friendships_initiated, likes, likes_received,
       mobile_likes, mobile_likes_received, www_likes,
       www_likes_received))
y = list(G)
# Test options and evaluation metric
num_folds = 5
seed = 7
scoring = 'accuracy'

# Model Test/Train
# Splitting what we are trying to predict into 4 different arrays -
# X train is a section of the x array(attributes) and vise versa for Y(features)
# The test data will test the accuracy of the model created
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.20, random_state=seed)
#splitting 20% of our data into test samples. If we train the model with higher data it already has seen that information and knows

#size of train and test subsets after splitting
np.shape(x_train), np.shape(x_test)
models = [
    DecisionTreeClassifier(),
    GaussianNB(),
    GradientBoostingClassifier(),
    RandomForestClassifier()
]
results = []
names = []
for model in models:
    cv_scores = cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy')
    results.append(cv_scores)
    names.append(model.__class__.__name__)

# Print the cross-validation results
for name, cv_scores in zip(names, results):
    print(f'{name} CV accuracy: {cv_scores.mean():.3f}')

# Train the best model on the full training set and make predictions on the test set
best_model = max(models, key=lambda m: cross_val_score(m, x_train, y_train, cv=5, scoring='accuracy').mean())
best_model.fit(x_train, y_train)
y_pred = best_model.predict(x_test)
model_accuracy = accuracy_score(y_test, y_pred)
print("Best Model Accuracy Score on Test Set:", model_accuracy)
report = classification_report(y_test, y_pred)
print(f'Best model: {best_model.__class__.__name__}')
print(report)


# Plot the accuracy scores using a box plot
fig = plt.figure()
fig.suptitle('Accuracy Scores of Models')
ax = fig.add_subplot(111)
ax.boxplot(results)
ax.set_xticklabels(names)
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.show()


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
best_model.fit(x_train, y_train)
rf_roc_auc = roc_auc_score(y_test,best_model.predict(x_test))
fpr,tpr,thresholds = roc_curve(y_test, best_model.predict_proba(x_test)[:,1
])
plt.figure()
plt.plot(fpr, tpr, label='{} (area = %0.2f)'.format(best_model) % rf_roc_auc)
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.savefig('LOC_ROC')
plt.show()
#Model Evaluation Metric 4-prediction report
for x in range(len(y_pred)):
  print("Predicted: ", y_pred[x], "Actual: ", y_test[x], "Data: ", x_test[x]) 