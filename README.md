# Avocado-Price-Prediction
Avocado Price Prediction
## How It Works

1. **Data Preparation and Model Training**
   - The project uses historical avocado sales data (`avocado.csv`), which includes features like date, region, total volume, and bag sizes[5].
   - The `model.py` script preprocesses the data by:
     - Encoding the `region` column using a label encoder.
     - Extracting date features: year, month, and day of week.
     - Selecting relevant features for modeling.
     - Splitting the data into training and test sets.
   - A Random Forest Regressor is trained with hyperparameter tuning via RandomizedSearchCV to find the best model.
   - The trained model and region encoder are saved as `random_forest_model.pkl` and `region_encoder.pkl` for deployment[1].

2. **Web Application for Prediction**
   - The Flask web app (`app.py`) loads the trained model and encoder[2].
   - Users interact with a web form (`index.html`) to enter prediction details: date, region, total volume, and bag sizes[3].
   - Upon form submission:
     - The app extracts and processes the input (including encoding the region and extracting date features).
     - Features are arranged in the required order and passed to the trained model for prediction.
     - The predicted average avocado price is displayed on the web page.

3. **User Experience**
   - The user simply fills out the form and clicks "Predict Price."
   - The backend handles all data processing and prediction.
   - The result (predicted price) is instantly shown in the browser.

---

**Summary:**  
You enter avocado sales details in a web form, the backend processes your input and uses a trained machine learning model to predict and display the expected average price for your chosen region and date.
