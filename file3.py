import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, root_mean_squared_error
url="insurance_dataset (1).csv"
df=pd.read_csv(url)
print(df.head())
print("Missing Values")
print(df.isnull().sum())
plt.figure(figsize=(8,5))
sns.regplot(x='age',y='charges',data=df,color='green',line_kws={"color":"red"})
x=df[['age']].values.reshape(-1,1)
y=df['charges'].values.reshape(-1,1)
print("Shape of x:", x.shape)
print("Shape of y:", y.shape)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(x_train,y_train)

def charge_prediction(spend_charge):
    return model.predict([[spend_charge]])
print("Model coefficients:", model.coef_[0])
print("Model intercept:", model.intercept_)
charges_pred=model.predict(x_test)
print(charges_pred)
mse=mean_squared_error(y_test,charges_pred)
r2=r2_score(y_test,charges_pred)
mae=mean_absolute_error(y_test,charges_pred)
print("Mean Squared Error=",mse)
print("R2=",r2)
print("Mean Absolute Error=",mae)
RMSE=root_mean_squared_error(y_test,charges_pred)
print("RMSE",RMSE)

n=int(input("Age:"))
print(charge_prediction(n))

plt.grid(True)
plt.show()
