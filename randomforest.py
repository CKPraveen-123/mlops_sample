#Importing Libraries
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import platform
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

#printing python version
print(f'The python version is {platform.python_version()}')

data_source = 'Real estate.csv'

#Data Processing
dataset = pd.read_csv(data_source)
Features = ['X2 house age','X3 distance to the nearest MRT station','X4 number of convenience stores', 'X5 latitude', 'X6 longitude']
target = ['Y house price of unit area']
X_train, X_test, y_train, y_test = train_test_split(dataset[Features], dataset[target], test_size=0.10, random_state=42)
print("Stage 1: Data Processing Completed Successfully" + "\u2705")

#Random Forest Model Training
RF_model = RandomForestRegressor()
RF_model.fit(X_train, y_train)
predictions = RF_model.predict(X_test)
RMSE_score_value = np.sqrt(mean_squared_error(y_true=y_test, y_pred=predictions))
R2_score_value = r2_score(y_true=y_test, y_pred=predictions)
print(f'RMSE_score_value: {RMSE_score_value}')
print(f'R2_score_value: {R2_score_value}')

# Save the trained model to a file
joblib.dump(RF_model, '/workspaces/mlops_sample/webservice/RandomForestRegressor_model.bin')
print('Model saved to path')

#Predicting using saved model
saved_model = joblib.load('/workspaces/mlops_sample/webservice/RandomForestRegressor_model.bin')
pred  = saved_model.predict(X_test.values[[0]])
truth_value = y_test.values[[0]]
print(f'Value Predicted: {pred}')
print(f'Value Original: {truth_value}')
print(f'Difference between predicted value and Original value: {pred-truth_value}')