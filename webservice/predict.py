#IMPORTING LIBRARIES
import requests
import threading
import time
from flask import Flask, request, jsonify
import joblib
import warnings
warnings.filterwarnings("ignore")

#Prediction Function by loading Model - SERVER SIDE
def test_model(*features):
    rf_model = joblib.load('RandomForestRegressor_model.bin')
    pred = rf_model.predict([features])
    return pred


#SERVER SIDE FLASK APP
app = Flask('House Price Unit Area Prediction')

@app.route("/predict", methods = ['POST'])

def predict():
    data = request.get_json()
    features = data["house_parameters"]
    house_price_per_unit_area = test_model(*features) # see test_model function written above
    return f'House_Price_Per_UnitArea: {house_price_per_unit_area[0]:.2f}'

#Function to run the Flask Server
def run_server():
    #use 127.0.0.1 for internal loop back communication
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader = False)

if __name__ == "__main__":
    #start the server in a separate background thread
    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()

    time.sleep(1)
    



#INPUTS - CLIENT SIDE
house_age = float(input('enter house age'))
MRT_distance = float(input('mrt_distance'))
Convinience_stores = int(input('no_convinience_stores'))
latitude = float(input('latitude'))
longitude = float(input('longitude'))
features = [house_age, MRT_distance, Convinience_stores, latitude, longitude]
payload = {"house_parameters": features}



# url = 'http://10.0.4.81:9696/predict'
#Making the Client API call using localhost/127.0.0.1
url = "http://127.0.0.1:5000/predict"

try:
    response = requests.post(url, json=payload)
    print("----API Response Received----")
    print(response.text)
except Exception as e:
    print(f"Connection Failed: {e}")