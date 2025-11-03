import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer

trian_data = pd.read_csv(r"C:\Users\computer amir\Desktop\Titanic-Dataset.csv")
df = pd.DataFrame(trian_data)
A = df.info()
print("A")
B = df.describe()
print("B")
C = df.duplicated().sum()
print("C")

si = SimpleImputer(missing_values=np.nan, strategy='mean')
si.fit(df.iloc[:, 5:6]) 
df.iloc[:, 5] = si.transform(df.iloc[:, 5:6])


categorical_columns = df.select_dtypes(include=['object']).columns 
for col in categorical_columns :
    df[col] = pd.Categorical(df[col]).codes

scaler = MinMaxScaler()
numeric_cols = df.select_dtypes(include=[np.number]).columns                
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

df.to_csv(r"C:\Users\computer amir\Desktop\Titanic-Dataset_atiyehosinnezkhad.csv", index=False)
print(df.head())