# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize the Flask application
superkart_sales_predictor_api = Flask("SuperKart Sales Predictor")

# Load the trained machine learning model
model = joblib.load("superkart_sales_prediction_model_v1_0.joblib")

# Define a route for the home page (GET request)
@superkart_sales_predictor_api.get('/')
def home():
    """
    This function handles GET requests to the root URL ('/') of the API.
    It returns a simple welcome message.
    """
    return "Welcome to the SuperKart Sales Prediction API!"

# Define an endpoint for single store sales prediction (POST request)
@superkart_sales_predictor_api.post('/v1/sales')
def predict_store_sales():
    """
    This function handles POST requests to the '/v1/sales' endpoint.
    It expects a JSON payload containing store details and returns
    the predicted store sale as a JSON response.
    """
    # Get the JSON data from the request body
    store_data = request.get_json()

    # Extract relevant features from the JSON data
    sample = {
        'Product_Weight': store_data['Product_Weight'],
        'Product_Allocated_Area': store_data['Product_Allocated_Area'],
        'Product_MRP': store_data['Product_MRP'],
        'Store_Age_Years': store_data['Store_Age_Years'],
        'Product_Sugar_Content': store_data['Product_Sugar_Content'],
        'Product_Type_Category': store_data['Product_Type_Category'],
        'Store_Size': store_data['Store_Size'],
        'Store_Location_City_Type': store_data['Store_Location_City_Type'],
        'Store_Type': store_data['Store_Type'],
        'Product_Id_char': store_data['Product_Id_char']
    }


    # Convert the extracted data into a Pandas DataFrame
    input_data = pd.DataFrame([sample])

    # Make prediction (the model predicts sales directly in dollars)
    predicted_sale = model.predict(input_data)[0]

    # Convert predicted_sale to a Python float rounded to two decimal places
    predicted_sale = round(float(predicted_sale), 2)
    # The conversion above is needed because model.predict returns NumPy float32 values.
    # When we send this value directly within a JSON response, Flask's jsonify function encounters a datatype error

    # Return the predicted sale
    return jsonify({'Predicted Sale (in dollars)': predicted_sale})


# Define an endpoint for batch prediction (POST request)
@superkart_sales_predictor_api.post('/v1/salesbatch')
def predict_store_sales_batch():
    """
    This function handles POST requests to the '/v1/salesbatch' endpoint.
    It expects a CSV file containing store details for multiple stores
    and returns the predicted store sales as a dictionary in the JSON response.
    """
    # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the CSV file into a Pandas DataFrame
    input_data = pd.read_csv(file)

    # Make predictions for all rows in the DataFrame
    predictions = model.predict(input_data).tolist()

    # Round predictions to two decimal places and convert to Python floats for JSON serialization
    predicted_sales = [round(float(sale), 2) for sale in predictions]

    # Create a dictionary of predictions keyed by row number and product category,
    # so rows sharing the same Product_Id_char (e.g. FD) don't overwrite each other
    product_ids = [f"row_{i}_{pid}" for i, pid in enumerate(input_data['Product_Id_char'])]
    output_dict = dict(zip(product_ids, predicted_sales))

    # Return the predictions dictionary as a JSON response
    return jsonify(output_dict)

# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    superkart_sales_predictor_api.run(debug=True)
