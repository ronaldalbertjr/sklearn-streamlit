from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

iris = load_iris(as_frame=True)
df = iris.data
target = iris.target

mi_array = mutual_info_regression(df.values, target, random_state=0)
mi_df = pd.DataFrame({'Mutual Information': mi_array}, index=df.columns)

gnb = GaussianNB()
gnb = gnb.fit(df.values, target)
target_gnb = gnb.predict(df.values)
gnb_matrix = confusion_matrix(target, target_gnb)


log = LogisticRegression(random_state=0)
log = log.fit(df.values, target)
target_log = log.predict(df.values)
log_matrix = confusion_matrix(target, target_log)


kernel_types = ['linear', 'poly', 'rbf', 'sigmoid']
svc = dict()
svc_matrix = dict()
svc_target = dict()

for i in kernel_types:
    curr_svc = SVC(kernel=i)
    curr_svc.fit(df.values, target)
    target_curr_svc = curr_svc.predict(df.values)
    curr_svc_matrix = confusion_matrix(target, target_curr_svc)
    svc[i] = curr_svc
    svc_matrix[i] = curr_svc_matrix
    svc_target[i] = target_curr_svc

criterions = ['gini', 'entropy']
tree = dict()
tree_matrix = dict()
tree_target = dict()

for i in criterions:
    curr_tree = DecisionTreeClassifier(criterion=i, random_state=0)
    curr_tree.fit(df.values, target)
    target_curr_tree = curr_tree.predict(df.values)
    curr_tree_matrix = confusion_matrix(target, target_curr_tree)
    tree[i] = curr_tree
    tree_matrix[i] = curr_tree_matrix
    tree_target[i] = target_curr_tree