#!/usr/bin/env python
# coding: utf-8

# In[31]:


#Importing all the required libraries


# In[32]:


import argparse
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import datetime
from xgboost import XGBRegressor


# In[33]:


data_source = 'Real estate.csv'


# In[34]:


algorithm_list = [LinearRegression(), Lasso(alpha=1.0), KNeighborsRegressor(n_neighbors=5), DecisionTreeRegressor(), RandomForestRegressor()]


# In[35]:


#Functions

#1. for data preparation and feature engineering
#2. training and selecting the best model and deploying it-first time
#3. training and deployig a new model


# In[36]:


#Data Processing

def process_data(data_source):
    dataset = pd.read_csv(data_source)
    Features = ['X2 house age','X3 distance to the nearest MRT station','X4 number of convenience stores', 'X5 latitude', 'X6 longitude']
    target = ['Y house price of unit area']
    X_train, X_test, y_train, y_test = train_test_split(dataset[Features], dataset[target], test_size=0.10, random_state=42)
    print("Stage 1: Data Processing Completed Successfully" + "\u2705")
    print('\n')
    return X_train.values, X_test.values, y_train.values, y_test.values
    #return X_train = df[0], X_test= df[1], y_train= df[2], y_test= df[3]


# In[37]:


#Train Models

def train_models(X_train, y_train, X_test, y_test):
    tested_models = []
    tested_models.clear()
    for each_algorithm in algorithm_list:
        model = each_algorithm
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        RMSE_score_value = np.sqrt(mean_squared_error(y_true=y_test, y_pred=predictions))
        R2_score_value = r2_score(y_true=y_test, y_pred=predictions)
        tested_models.append({each_algorithm.__class__.__name__:{'RMSE':RMSE_score_value, 'R2':R2_score_value}})
    
    #PolynomialRegression
    polyfeatures = PolynomialFeatures(degree=2,include_bias=False)
    x_poly = polyfeatures.fit_transform(X_train)
    lr = LinearRegression()
    lr.fit(x_poly, y_train)
    poly_predictions = lr.predict(polyfeatures.fit_transform(X_test))

    poly_RMSE_score_value = np.sqrt(mean_squared_error(y_true=y_test, y_pred=poly_predictions))
    poly_R2_score_value = r2_score(y_true=y_test, y_pred=poly_predictions)
    tested_models.append({'PolynomialRegression':{'RMSE':poly_RMSE_score_value, 'R2':poly_R2_score_value}})

    #SVR
    svr = SVR(kernel='rbf')
    scaler = StandardScaler()
    scaler_y= StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    svr.fit(X_train_scaled, y_train_scaled)
    scaled_ytest_predictions = svr.predict(X_test_scaled)
    SVR_RMSE_score_value = np.sqrt(mean_squared_error(y_pred=scaler_y.inverse_transform(scaled_ytest_predictions.reshape(-1,1)), y_true = scaler_y.inverse_transform(y_test_scaled)))
    SVR_R2_score_value= r2_score(y_pred=scaler_y.inverse_transform(scaled_ytest_predictions.reshape(-1,1)), y_true = scaler_y.inverse_transform(y_test_scaled))
    tested_models.append({svr.__class__.__name__:{'RMSE':SVR_RMSE_score_value, 'R2':SVR_R2_score_value}})

    #Entire trained models with run parameters as a dataframe:

    rows = [{"Model": name, **metrics} for item in tested_models for name, metrics in item.items()]
    df = pd.DataFrame(rows)

    # 2. Sort by RMSE (Ascending) and R2 (Descending)
    df_sorted = df.sort_values(
        by=["RMSE", "R2"], 
        ascending=[True, False]
    ).reset_index(drop=True)

    # 3. Clean up the output visual formatting
    df_sorted[["RMSE", "R2"]] = df_sorted[["RMSE", "R2"]].round(4)
    
    #creating a csv database of models

    try:
        with open('models_database.csv',mode='r') as file:
            pass

    except FileNotFoundError:
        df_sorted.to_csv(path_or_buf='models_database.csv', index=False)
        print("Stage 2: Models Training Completed Successfully" + "\u2705")
        print('\n')
        return 'models_database.csv'

    else:
        df_sorted.to_csv(path_or_buf='models_database.csv', mode="a", header="False", index="False")
        print("Stage 2: Models Training Completed Successfully" + "\u2705")
        print('\n')
        return 'models_database.csv'


# In[38]:


#Deploying the best performing model - high R2 and low RMSE

def model_deployment(trained_models_database):
    models_df = pd.read_csv(trained_models_database)
    best_model = models_df.sort_values(by=['RMSE', 'R2'],ascending=[True, False]).reset_index(drop=True).head(1)

    #creating a deployed_model database

    best_model_df = pd.DataFrame(best_model)
    best_model_df['deployment_Status'] = 'In_Production'
    best_model_df['deployed_time'] = pd.Timestamp.now(tz='America/New_York').strftime('%Y-%m-%d %H:%M:%S')

    ## Deployment Database fields - ['Model', 'RMSE', 'R2', 'deployment_Status', 'deployed_time']


    try:
        with open('models_deployed.csv',mode='r') as file:
            pass

    except FileNotFoundError:
        best_model_df.to_csv(path_or_buf='models_deployed.csv', index=False)
        print(f'{best_model_df["Model"][0]} has been deployed to models_deployed database at {best_model_df["deployed_time"][0]}')
        print("Stage 3: Best Model deployed Successfully" + "\u2705")
        print('\n')

    else:
        best_model_df.to_csv(path_or_buf='models_deployed.csv', mode="a", header="False", index="False")
        remov_status_df = pd.read_csv('models_deployed.csv')
    #   remov_status_df.sort_values(by ='deployed_time', ascending=False).reset_index(drop=True).iloc[0]['deployment_Status'] == 'In_Production'
        remov_status_df.sort_values(by ='deployed_time', ascending=False).reset_index(drop=True).iloc[1:]['deployment_Status'] == 'Archived'
        remov_status_df.to_csv('models_deployed.csv', index = False)
        print(f'{best_model_df["Model"][0]} has been deployed to models_deployed database at {best_model_df["deployed_time"][0]}')
        print("Stage 3: Best Model deployed Successfully" + "\u2705")


# In[39]:


#Running Script


# In[40]:


df = process_data(data_source)
db = train_models(df[0], df[2], df[1], df[3])
model_deployment(db)


# In[ ]:





# In[ ]:




