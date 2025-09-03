# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

step 1.start

step 2. Importing necessary liberaries

step 3. Data preprocessing

step 4. Spliting data int training and testing data

step 5. Performing SGD-Regressor

step 6. Calculating error

step 7. end

## Program:
```
Program to implement the multivariate linear regression

model for predicting the price of the house and number

of occupants in the house with SGD regressor.

Developed by: Ragala Sai Vivek

RegisterNumber:212223230163 
```
``` python
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
data=fetch_california_housing()
df=pd.DataFrame(data.data,columns=data.feature_names)
df['target']=data.target
print(df.head())
X=data.data[:,:3]
Y=np.column_stack((data.target,data.data[:,6]))
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
scaler_X=StandardScaler()
scaler_Y=StandardScaler()
X_train=scaler_X.fit_transform(X_train)
X_test=scaler_X.transform(X_test)
Y_train=scaler_Y.fit_transform(Y_train)
Y_test=scaler_Y.transform(Y_test)
sgd=SGDRegressor(max_iter=1000,tol=1e-3)
multi_output_sgd=MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train,Y_train)
Y_pred=multi_output_sgd.predict(X_test)
Y_pred=scaler_Y.inverse_transform(Y_pred)
Y_test=scaler_Y.inverse_transform(Y_test)
mse=mean_squared_error(Y_test,Y_pred)
print("Mean Sqaured Error:",mse)
print("\nPredictions:\n",Y_pred[:5])
```
## Output:

```
## Mean Sqaured Error: 2.5464944022481797

## Predictions:
 [[ 1.07950904 35.79410759]
 [ 1.51183585 35.77991986]
 [ 2.33422076 35.5771072 ]
 [ 2.67608454 35.39793682]
 [ 2.09214484 35.66633477]]
```
## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
