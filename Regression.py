from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
import pandas as pd
import numpy as np


data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]


X=data
y=target
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=33,test_size=0.25)
ss_X=StandardScaler()
ss_Y=StandardScaler()
X_train=ss_X.fit_transform(X_train)
X_test=ss_X.transform(X_test)
y_train=ss_Y.fit_transform(y_train.reshape(-1,1))
y_train=y_train.ravel()
y_test=ss_Y.transform(y_test.reshape(-1,1))
y_test=y_test.ravel()


lr=LinearRegression()
sgdr=SGDRegressor()

lr.fit(X_train,y_train)
y1_pred=lr.predict(X_test)
print('lr:',lr.score(X_test,y_test))
print(r2_score(y_test,y1_pred))
print(mean_squared_error(ss_Y.inverse_transform(y_test.reshape(-1,1)),ss_Y.inverse_transform(y1_pred.reshape(-1,1))))

sgdr.fit(X_train,y_train)
y2_pred=sgdr.predict(X_test)
print('sgdr:',sgdr.score(X_test,y_test))
print(r2_score(y_test,y2_pred))
print(mean_squared_error(ss_Y.inverse_transform(y_test.reshape(-1,1)),ss_Y.inverse_transform(y2_pred.reshape(-1,1))))

from sklearn.tree import ExtraTreeRegressor
etr=ExtraTreeRegressor()
etr.fit(X_train,y_train)
y3_pred=etr.predict(X_test)
print('dtr:',etr.score(X_test,y_test))
print(np.sort(list(zip(etr.feature_importances_,range(13))),axis=0))