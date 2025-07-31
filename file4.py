#implement mlr using company dataset
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.core.common import random_state
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,root_mean_squared_error


df=pd.read_csv("Company_data.csv")
print(df.head())
print()
print("Missing Values")
print(df.isnull().sum())
x=df.drop('Sales',axis=1)
y=df['Sales']
print()
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.2,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)

print("Model Intercept: ", model.intercept_)
print("Model Coefficient: ")
features = x.columns
coefficients = model.coef_
coef_df = pd.DataFrame({
    'Features': features,
    'Coefficient':coefficients
})
print(coef_df)
print()
sales_price=model.predict(x_test)
print(sales_price)
print()
mae=mean_absolute_error(y_test,sales_price)
mse=mean_squared_error(y_test,sales_price)
rmse=root_mean_squared_error(y_test,sales_price)
r2=r2_score(y_test,sales_price)
print("Mean Squared Error = ",mse)
print("R2 Score = ",r2)
print("Mean Absolute Error = ",mae)
print("Root Mean Squared error = ",rmse)