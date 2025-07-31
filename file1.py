import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
url="insurance_dataset (1).csv"
df=pd.read_csv(url)
print(df.head())
print("Missing Values")
print(df.isnull().sum())
plt.figure(figsize=(8,5))
sns.regplot(x='age',y='charges',data=df,color='green',line_kws={"color":"red"})
x=df[['age']]
y=df['charges']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
plt.grid(True)
plt.show()
