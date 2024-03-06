import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import pickle



def get_data(path):
    print("########## Data Preparation Started ###################")
    df = pd.read_csv(path)
    X = df.drop(columns = ["Outcome"],axis =1 )
    y = df["Outcome"]
   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 42) 

    scaler = StandardScaler()

    x_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns = X_train.columns)
    x_test_scaled = pd.DataFrame (scaler.transform(X_test), columns = X_test.columns)
    
    return x_train_scaled, y_train, x_test_scaled, y_test




