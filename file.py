import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
url="https://raw.githubusercontent.com/sreelakshmi-mca/Sales-dataset/refs/heads/main/insurance_dataset%20(1).csv"
df=pd.read_csv(url)
print(df.head())
print("Missing Values")
print(df.isnull().sum())
plt.figure(figsize=(8,5))
sns.regplot(x='age',y='charges',data=df,color='green',line_kws={"color":"red"})
plt.grid(True)
plt.show()
