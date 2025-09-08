from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
from datetime import datetime
import traceback
import sys

app = Flask(__name__)

model = joblib.load('random_forest_model.pkl')
region_encoder = joblib.load('region_encoder.pkl')

features = ['year', 'month', 'day_of_week', 'Total Volume', 'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags', 'region']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            # Validate and extract date
            date_str = request.form.get('date', '').strip()
            if not date_str:
                return render_template('index.html', prediction_text="Error: Please select a date.")
            
            try:
                date = datetime.strptime(date_str, '%Y-%m-%d')
                year = date.year
                month = date.month
                day_of_week = date.weekday() 
            except ValueError:
                return render_template('index.html', prediction_text="Error: Invalid date format. Please use YYYY-MM-DD format.")

            # Validate numerical inputs
            try:
                total_volume = float(request.form.get('total_volume', 0))
                total_bags = float(request.form.get('total_bags', 0))
                small_bags = float(request.form.get('small_bags', 0))
                large_bags = float(request.form.get('large_bags', 0))
                xlarge_bags = float(request.form.get('xlarge_bags', 0))
                
                # Check for negative values
                if any(val < 0 for val in [total_volume, total_bags, small_bags, large_bags, xlarge_bags]):
                    return render_template('index.html', prediction_text="Error: Volume and bag counts cannot be negative.")
                    
            except (ValueError, TypeError):
                return render_template('index.html', prediction_text="Error: Please enter valid numbers for all volume and bag fields.")

            # Validate region
            region = request.form.get('region', '').strip()
            if not region:
                return render_template('index.html', prediction_text="Error: Please select a region.")
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
                error_message = f"Error: Region '{region}' not recognized. Please select a valid region from the dropdown."
                return render_template('index.html', prediction_text=error_message)

            region_encoded = region_mapping[region]

            input_data = np.array([[year, month, day_of_week, total_volume, total_bags, small_bags, large_bags, xlarge_bags, region_encoded]])

            prediction = model.predict(input_data)
            result_message = f'Predicted Average Price: ${prediction[0]:.2f}'
            
            return render_template('index.html', prediction_text=result_message)
            
    except Exception as e:
        error_message = f'Error occurred while predicting price: {str(e)}'
        return render_template('index.html', prediction_text=error_message)

if __name__ == '__main__':
    app.run(debug=True)
