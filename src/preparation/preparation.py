import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import sklearn as sk

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

# charger les données
df=pd.read_csv('../data/dataset.csv')

# transformer toutes les valeurs non numérique en entier
df["Gender"] = (df['Gender'] == 'Male').astype(int)
df.family_history_with_overweight = (df.family_history_with_overweight == 'yes').astype(int)
df.FAVC = (df.FAVC == 'yes').astype(int)
df.SMOKE = (df.SMOKE == 'yes').astype(int)
df.SCC = (df.SCC == 'yes').astype(int)

# arondi au centième la taille et au dixième le poids. Le reste arrondi à l'unité
for i in df.select_dtypes(exclude = ['object']).columns:
    if i == 'Height':
        df[i] = df[i].round(2)
    elif  i =='Weight':
        df[i] = df[i].round(1)
    elif i != 'NCP':
        df[i] = df[i].round(0)

def encode_NCP(code):
    if code < 2:
        return 1
    elif code < 3:
        return 2
    else :
        return 3
    
def encode_CAEC_CALC(code):
    if code == "no":
        return 0
    if code == "Sometimes":
        return 1
    if code == "Frequently":
        return 2
    if code == "Always":
        return 3
    
    raise Exception(f"Unknown code: {code}")

def encode_public_transport(code):
    if code == "Automobile":
        return 0
    if code == "Motorbike":
        return 1
    if code == "Bike":
        return 2
    if code == "Public_Transportation":
        return 3
    if code == "Walking":
        return 4
    
    raise Exception(f"Unknown code: {code}")
   
df["NCP"] = df["NCP"].apply(encode_NCP)
df["CAEC"] = df["CAEC"].apply(encode_CAEC_CALC)
df["CALC"] = df["CALC"].apply(encode_CAEC_CALC)
df["MTRANS"] = df["MTRANS"].apply(encode_public_transport)
df['Age'] = df['Age'].astype('int64')
df['FCVC'] = df['FCVC'].astype('int64')
df['FAF'] = df['FAF'].astype('int64')
df['CH2O'] = df['CH2O'].astype('int64')
df['TUE'] = df['TUE'].astype('int64')

for i in df.columns:
    print(i)
    print("----------")
    print(set(df[i].tolist()))
    print("-------------")

df.head()
