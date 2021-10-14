from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LogisticRegression
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