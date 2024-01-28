#-----------------------------------------------------------------------------------------------------------------------
# Vashu Patel
# CS 301 Section 003
# Final Project
#-----------------------------------------------------------------------------------------------------------------------
# Below are all the packages that are required for the project. Some packages remained unused and were imported only for
# experimental purposes.

import itertools
from itertools import cycle
import operator
import pandas as pd
import numpy as np
from scikitplot.metrics import plot_precision_recall_curve
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.feature_selection import SelectKBest, f_regression
from scipy.stats import pearsonr
from sklearn.metrics import recall_score,precision_recall_curve, roc_curve, precision_score, f1_score,accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, auc, classification_report
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import RandomOverSampler
import scikitplot as skplt
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)
data= pd.read_csv('fetal_health.csv')
df = data.copy(deep=True)
df['fetal_health'] = df['fetal_health'].replace({
    1.0:"NORMAL",
    2.0:"SUSPECT",
    3.0:"PATHOLOGICAL"
})
#-----------------------------------------------------------------------------------------------------------------------
# Task 1: Visualization of the 3 states. This was achieved by plotting a bar graph and pie chart.
print("Task 1: \n")

labels = ['SUSPECT',
         'NORMAL',
         'PATHOLOGICAL']
fig, ax = plt.subplots(figsize=(12,6))
sns.countplot(x=df['fetal_health'])
ax.set_xlabel('Fetal Health')
ax.set_xticklabels(labels)
plt.title('Distribution of Fetal Health Conditions')
plt.text(s = f"n = {df['fetal_health'].value_counts()[1]}", x = -0.25, y = 800)
plt.text(s = f"n = {df['fetal_health'].value_counts()[0]}", x = 1, y = 800)
plt.text(s = f"n = {df['fetal_health'].value_counts()[2]}", x = 2, y = 800)

plt.show()


#-----------------------------------------------------------------------------------------------------------------------
# Pie Chart

print(df['fetal_health'].value_counts(), "\n")

plt.pie(
    df['fetal_health'].value_counts(),
    autopct='%.2f%%',
    labels=['NORMAL','SUSPECT','PATHOLOGICAL'],
    colors = ["lightgreen", "Yellow", "Red"]
)
plt.title("Task 1: Distribution of the classes")
plt.show()
#-----------------------------------------------------------------------------------------------------------------------
# Task 2:  10 Best Features
target = data['fetal_health']
X = data.drop(['fetal_health'], axis=1)
y = df['fetal_health']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y,random_state=42)
print("X_train.shape  X_test.shape y_train.shape y_test.shape")
print(X_train.shape,"\t", X_test.shape,"\t" ,y_train.shape, "\t",y_test.shape)

print("-------------------------------------------------------------------------------------------------------------\n")

print("Task 2: \n")
corrMatrix = X_train.corr()



featureGT90 = set()
featureGT95 = set()
for i in range(len(corrMatrix .columns)):
    for j in range(i):
        if abs(corrMatrix.iloc[i, j]) > 0.9:
            colname = corrMatrix.columns[i]
            colVal = abs(corrMatrix.iloc[i,j])
            featureGT90.add(colname)

        elif abs(corrMatrix.iloc[i,j]) > 0.95:
            colname = corrMatrix.columns[i]
            colVal = abs(corrMatrix.iloc[i,j])
            featureGT95.add(colname)

print("Features with more that 0.90 correlation: ",featureGT90, "\n")
print("Features with more that 0.95 correlation: ",featureGT95, "\n")
dicti = {}
for i in range(len(corrMatrix .columns)):
    for j in range(i):
        abs(corrMatrix.iloc[i, j])
        colname = corrMatrix.columns[i]
        colVal = abs(corrMatrix.iloc[i,j])
        dicti.update({colname:colVal})
sortedDict = sorted(dicti.items())
x = itertools.islice(sorted(dicti.items(),key=operator.itemgetter(1), reverse=True),0,10)
print("Top 10 Features","\n")
print("{:<55} {:<55} ".format('Features', 'Correlation Value',"\n"))
for key, value in x:
    features = key
    corr = value
    print("{:<55} {:<55} ".format(key, value))
target = data['fetal_health']
X = data.drop(['fetal_health'], axis=1)
y = df['fetal_health']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y,random_state=42)

def conf_matrix(matrix,pred):
    class_names= [["NORMAL", "SUSPECT", "PATHOLOGICAL"]]
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()


#--------------------------------------------------------------------------------------------------\

# Task 3: Development of 2 Models
print("-------------------------------------------------------------------------------------------------------------\n")
print("Task 3: \n")
print("Gaussian NB Report: \n")
nb = GaussianNB()
nb.fit(X_train,y_train)
y_pred = nb.predict(X_test)
acc = metrics.accuracy_score(y_pred, y_test)
cnf_matrix = metrics.confusion_matrix(y_pred, y_test)
conf_matrix(cnf_matrix, y_test)
report = classification_report(y_pred,y_test)
print(report)
print("-------------------------------------------------------------------------------------------------------------\n")
print("Decision Tree Classifier Report: \n")
dc = DecisionTreeClassifier()
dc.fit(X_train, y_train)
y_pred2 = dc.predict(X_test)
acc2 = metrics.accuracy_score(y_pred2, y_test)
cnf_matrix2 = metrics.confusion_matrix(y_pred2, y_test)
conf_matrix(cnf_matrix2, y_test)
report2 = classification_report(y_pred2, y_test)
print(report2)

print("-------------------------------------------------------------------------------------------------------------\n")

#-----------------------------------------------------------------------------------------------------------------------

# Task 4: Confusion Matrix

print("Task 4: \n")
print("Gaussian NB Confusion Matrix:\n")
print(confusion_matrix(y_test,y_pred),"\n")
print("-------------------------------------------------------------------------------------------------------------\n")
print("Decision Tree Classifier Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred2),"\n")
print("-------------------------------------------------------------------------------------------------------------\n")
#-----------------------------------------------------------------------------------------------------------------------

# Task 5: Area under the ROC Curve, F1 Score, Area Under the Precision-Recall Curve

print("Task 5: \n")

print("Gaussian NB\nF1 Score",f1_score(y_test,nb.predict(X_test), average='weighted'))

y_score = nb.predict_proba(X_test)
skplt.metrics.plot_roc_curve(y_test, y_score)
plt.title("Gaussian NB ROC Curve")
plt.show()

plot_precision_recall_curve(y_test, y_score)
plt.title("Gaussian NB Precision vs. Recall")
plt.show()

print("-------------------------------------------------------------------------------------------------------------\n")

print("Decision Tree Classifier\nF1 Score", f1_score(y_test,dc.predict(X_test), average='weighted'))

y_score2 = dc.predict_proba(X_test)

skplt.metrics.plot_roc_curve(y_test, y_score2)
plt.title("Decision Tree Classifier ROC Curve")
plt.show()

plot_precision_recall_curve(y_test, y_score2)
plt.title("Decision Tree Classifier Precision vs. Recall")
plt.show()

print("-------------------------------------------------------------------------------------------------------------\n")

#-----------------------------------------------------------------------------------------------------------------------
# Task 6: K-Means Clustering of 5, 10, 15

print("Task 6: \n")


X = data.iloc[:, 0:21].values




fiveK = KMeans(n_clusters = 5, init='k-means++', random_state=42)
label = fiveK.fit_predict(X)
print(label)

plt.scatter(X[:, 0], X[:, 1], c=label, cmap=plt.cm.Paired)
plt.title("KMeans Clustering: K = 5")
plt.show()


tenK = KMeans(n_clusters=10, init='k-means++', random_state=42)
label = tenK.fit_predict(X)
print(label)

plt.scatter(X[:, 0], X[:, 1], c=label, cmap=plt.cm.Paired)
plt.title("KMeans Clustering: K = 10")
plt.show()

#K-means clustering for K=15
FifK = KMeans(n_clusters=15, init='k-means++', random_state=42)
label = FifK.fit_predict(X)
print(label)

plt.scatter(X[:, 0], X[:, 1], c=label, cmap=plt.cm.Paired)
plt.title("KMeans Clustering: K = 15")
plt.show()

print("-------------------------------------------------------------------------------------------------------------\n")

sc = StandardScaler()
featureScaled = sc.fit_transform(X)
kDict = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
 }

sse = []
for k in range(5,16,5):
    kmeans = KMeans(n_clusters=k, **kDict)
    kmeans.fit(featureScaled)
    sse.append(kmeans.inertia_)

plt.style.use("fivethirtyeight")
plt.plot(range(5,16,5),sse)
plt.xticks(range(5,16,5))
plt.xlabel("# of Clusters")
plt.ylabel("SSE")
plt.show()
