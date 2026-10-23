import streamlit as st
import pandas as pd
import requests

# Base URL of the Flask backend
BACKEND_URL = "http://backend:7860"

# Set the title of the Streamlit app
st.title("SuperKart Sales Prediction")

# Section for online prediction
st.subheader("Online Prediction")

# Collect user input for product and store features
product_weight = st.number_input("Product Weight", min_value=1.0, max_value=30.0, value=12.5, step=0.1)
product_allocated_area = st.number_input("Product Allocated Area (ratio of display area)", min_value=0.001, max_value=1.0, value=0.05, step=0.001, format="%.3f")
product_mrp = st.number_input("Product MRP", min_value=1.0, max_value=500.0, value=145.0, step=1.0)
store_age_years = st.number_input("Store Age (years)", min_value=0, max_value=100, value=16, step=1)
product_sugar_content = st.selectbox("Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
product_type_category = st.selectbox("Product Type Category", ["Perishables", "Non Perishables"])
product_id_char = st.selectbox("Product Category Code (FD = Food, DR = Drinks, NC = Non-Consumable)", ["FD", "DR", "NC"])
store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
store_location_city_type = st.selectbox("Store City Type", ["Tier 1", "Tier 2", "Tier 3"])
store_type = st.selectbox("Store Type", ["Departmental Store", "Supermarket Type1", "Supermarket Type2", "Food Mart"])

# Convert user input into a DataFrame (column names must match the model features)
input_data = pd.DataFrame([{
    'Product_Weight': product_weight,
    'Product_Allocated_Area': product_allocated_area,
    'Product_MRP': product_mrp,
    'Store_Age_Years': store_age_years,
    'Product_Sugar_Content': product_sugar_content,
    'Product_Type_Category': product_type_category,
    'Store_Size': store_size,
    'Store_Location_City_Type': store_location_city_type,
    'Store_Type': store_type,
    'Product_Id_char': product_id_char
}])

# Make prediction when the "Predict" button is clicked
if st.button("Predict", type="primary"):
    response = requests.post(f"{BACKEND_URL}/v1/sales", json=input_data.to_dict(orient='records')[0])  # Send data to Flask API
    if response.status_code == 200:
        prediction = response.json()['Predicted Sale (in dollars)']
        st.success(f"Predicted Product Store Sales (in dollars): {prediction}")
    else:
        st.error(f"Prediction failed ({response.status_code}): {response.text}")

# Section for batch prediction
st.subheader("Batch Prediction")

# Allow users to upload a CSV file for batch prediction
uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=["csv"])

# Make batch prediction when the "Predict Batch" button is clicked
if uploaded_file is not None:
    if st.button("Predict Batch", type="primary"):
        response = requests.post(f"{BACKEND_URL}/v1/salesbatch", files={"file": uploaded_file})  # Send file to Flask API
        if response.status_code == 200:
            predictions = response.json()
            st.success("Batch predictions completed!")
            st.write(predictions)  # Display the predictions
        else:
            st.error(f"Batch prediction failed ({response.status_code}): {response.text}")
