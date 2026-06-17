#Importing Libraries
import requests
import joblib
import warnings
warnings.filterwarnings("ignore")
from flask import Flask, request

#Prediction Function by loading Model
def test_model(*features):
    rf_model = joblib.load('/workspaces/mlops_sample/webservice/RandomForestRegressor_model.bin')
    pred = rf_model.predict([features])
    return pred

#Main Function to call

app = Flask('House Price Unit Area Prediction')

@app.route("/predict", methods = ['POST'])

def predict():
    data = request.get_json()
    features = data["house_parameters"]
    house_price_per_unit_area = test_model(*features)
    return f'House_Price_Per_UnitArea: {house_price_per_unit_area[0]:.2f}'


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
    
