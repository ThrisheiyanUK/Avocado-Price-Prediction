from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
from datetime import datetime

app = Flask(__name__)

model = joblib.load('random_forest_model.pkl')
region_encoder = joblib.load('region_encoder.pkl')

features = ['year', 'month', 'day_of_week', 'Total Volume', 'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags', 'region']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        date_str = request.form['date']
        date = datetime.strptime(date_str, '%Y-%m-%d')
        year = date.year
        month = date.month
        day_of_week = date.weekday() 

        total_volume = float(request.form['total_volume'])
        total_bags = float(request.form['total_bags'])
        small_bags = float(request.form['small_bags'])
        large_bags = float(request.form['large_bags'])
        xlarge_bags = float(request.form['xlarge_bags'])


        region = request.form['region'].strip()
        region_mapping = {
            'California': 0,
            'New York': 1,
            'Albany': 3, 
            'Atlanta': 4, 
            'BaltimoreWashington': 5, 
            'Boise': 6, 
            'Boston': 7,
            'BuffaloRochester': 8,
            'Charlotte': 9, 
            'Chicago': 10,
            'CincinnatiDayton': 11,
            'Columbus': 12, 
            'DallasFtWorth': 13, 
            'Denver': 14,
            'Detroit': 15, 
            'GrandRapids': 16, 
            'GreatLakes': 17, 
            'HarrisburgScranton': 18,
            'HartfordSpringfield': 19, 
            'Houston': 20, 
            'Indianapolis': 21, 
            'Jacksonville': 22,
            'LasVegas': 23, 
            'LosAngeles': 24, 
            'Louisville': 25, 
            'MiamiFtLauderdale': 26,
            'Midsouth': 27, 
            'Nashville': 28, 
            'NewOrleansMobile': 29,
            'Northeast': 30, 
            'NorthernNewEngland': 31, 
            'Orlando': 32, 
            'Philadelphia': 33,
            'PhoenixTucson': 34, 
            'Pittsburgh': 35, 
            'Plains': 36, 
            'Portland': 37,
            'RaleighGreensboro': 38, 
            'RichmondNorfolk': 39, 
            'Roanoke': 40, 
            'Sacramento': 41,
            'SanDiego': 42, 
            'SanFrancisco': 43, 
            'Seattle': 44, 
            'SouthCarolina': 45,
            'SouthCentral': 46, 
            'Southeast': 47, 
            'Spokane': 48, 
            'StLouis': 49, 
            'Syracuse': 50,
            'Tampa': 51, 
            'TotalUS': 52, 
            'West': 53, 
            'WestTexNewMexico': 54
        }

        if region not in region_mapping:
            return render_template('index.html', prediction_text="Region not recognized.")

        region_encoded = region_mapping[region]

        input_data = np.array([[year, month, day_of_week, total_volume, total_bags, small_bags, large_bags, xlarge_bags, region_encoded]])

        prediction = model.predict(input_data)

        return render_template('index.html', prediction_text=f'Predicted Average Price: ${prediction[0]:.2f}')

if __name__ == '__main__':
    app.run(debug=True)
