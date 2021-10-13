from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report


import pandas as pd
import numpy as np

# Logistic良恶性乳腺癌预测
dates = pd.date_range('20130101', periods=3)
columns=range(11)
df = pd.DataFrame([[1,0,1],[0,0,1],[0,1,1]], index=dates, columns=list('ABC'))
# data=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',names=columns)
# data=data.replace(to_replace='?',value=np.nan)
# data=data.dropna(how='any')
# X_train,X_test,y_train,y_test=train_test_split(data.loc[:,0:9],data.loc[:,10],test_size=0.25,random_state=33)


# ss=StandardScaler()
# X_train=ss.fit_transform(X_train)
# X_test=ss.transform(X_test)
# lr=LogisticRegression()
# sgdc=SGDClassifier()
# lr.fit(X_train,y_train)
# lr_y_predict=lr.predict(X_test)
# sgdc.fit(X_train,y_train)
# sgdc_y_predict=sgdc.predict(X_test)
#
# print(lr.score(X_test,y_test))
# print(sgdc.score(X_test,y_test))
# print(classification_report(y_test,lr_y_predict,target_names=['Benign','Malignant']))
#
# print(classification_report(y_test,sgdc_y_predict,target_names=['Benign','Malignant']))

# SVM手写数字辨识
# from sklearn.datasets import load_digits
# from sklearn.svm import LinearSVC
# digits=load_digits()
# X_train,X_test,y_train,y_test=train_test_split(digits.data,digits.target,test_size=0.25,random_state=33)
# ss=StandardScaler()
# X_train=ss.fit_transform(X_train)
# X_test=ss.transform(X_test)
# lsvc=LinearSVC()
# lsvc.fit(X_train,y_train)
# y_predict=lsvc.predict(X_test)
# print(classification_report(y_test,y_predict,target_names=digits.target_names.astype(str)))

# 贝叶斯新闻文本数据细节抓取细节
# from sklearn.datasets import fetch_20newsgroups
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
# news=fetch_20newsgroups(subset='all')
# X_train,X_test,y_train,y_test=train_test_split(news.data,news.target,test_size=0.25,random_state=33)
# vec=CountVectorizer()
# X_train=vec.fit_transform(X_train)
# X_test=vec.transform(X_test)
# mnb=MultinomialNB()
# mnb.fit(X_train,y_train)
# print(mnb.score(X_test,y_test))
# y_predict=mnb.predict(X_test)
# print(classification_report(y_test,y_predict,target_names=news.target_names))

# Knn分类器分类iris数据集
# from sklearn.datasets import load_iris
# from sklearn.neighbors import KNeighborsClassifier
# iris=load_iris()
# X_train,X_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.25,random_state=33)
# ss=StandardScaler()
# X_train=ss.fit_transform(X_train)
# X_test=ss.transform(X_test)
# Knn=KNeighborsClassifier()
# Knn.fit(X_train,y_train)
# y_predict=Knn.predict(X_test)
# print(Knn.score(X_test,y_test))
# print(classification_report(y_test,y_predict,target_names=iris.target_names))

#决策树预测Titanic生存问题
titanic_train=pd.read_csv('train.csv')
titanic_test=pd.read_csv('test.csv')