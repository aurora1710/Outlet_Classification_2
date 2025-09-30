import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the model and scaler
try:
    model = joblib.load('logistic_regression_model.pkl')
    scaler = joblib.load('scaler.pkl')
    # Assuming X is available from previous steps or you have saved the columns list
    # For robustness, it's better to save the list of columns after training
    # For this example, I will assume the columns from the training data are needed for reindexing
    # In a real app, save the column names of the training data after one-hot encoding
    # Example:
    # train_cols = joblib.load('training_columns.pkl') # Load saved column names
    # For demonstration, let's define a placeholder for expected columns based on the notebook state
    # Replace with actual loading of saved column names in a production app
    expected_columns = ['No', 'SKU_count', 'Avg_Sales', 'Willingness_Organic', 'Willingness_Atta', 'Willingness_Ghee', 'Willingness_Oil', 'Willingness_Beverage', 'Willingness_Bakery', 'Importance_of_AMUL', 'Reputation_of_Store', 'Self_Service_Yes', 'Customer_Can_Browse_Partial View', 'Customer_Can_Browse_No Entry', 'Shop_Type_Modern trade', 'Cold_Storage_Yes', 'Shelf_Space_20-30ft', 'Shelf_Space_30-40ft', 'Shelf_Space_40-50ft', 'Shelf_Space_<20ft', 'Shelf_Space_>50ft'] # Example based on notebook's X.columns
#except FileNotFoundError:
    #st.error("Model or scaler file not found. Please ensure 'OC_logistic_regression_model.pkl' and 'scaler.pkl' are in the same directory.")
    #st.stop()


st.title('Outlet Classification Predictor')

st.write("""
Enter the details of the outlet to predict its classification (Premium or Non-Premium).
""")

st.header("Outlet Features")

no = st.number_input("No", min_value=1, value=1)
sku_count = st.number_input("SKU Count", min_value=0, value=13)
avg_sales = st.number_input("Average Sales", min_value=0, value=17)
willingness_organic = st.number_input("Willingness Organic (1-5)", min_value=1, max_value=5, value=1)
willingness_atta = st.number_input("Willingness Atta (1-5)", min_value=1, max_value=5, value=1)
willingness_ghee = st.number_input("Willingness Ghee (1-5)", min_value=1, max_value=5, value=1)
willingness_oil = st.number_input("Willingness Oil (1-5)", min_value=1, max_value=5, value=1)
willingness_beverage = st.number_input("Willingness Beverage (1-5)", min_value=1, max_value=5, value=3)
willingness_bakery = st.number_input("Willingness Bakery (1-5)", min_value=1, max_value=5, value=1)
importance_of_amul = st.number_input("Importance of AMUL (1-5)", min_value=1, max_value=5, value=4)
reputation_of_store = st.number_input("Reputation of Store (1-5)", min_value=1, max_value=5, value=2)

self_service = st.selectbox("Self Service", ['Yes', 'No'])
customer_can_browse = st.selectbox("Customer Can Browse", ['Full View and Choice', 'Partial View', 'No Entry'])
shop_type = st.selectbox("Shop Type", ['General trade', 'Modern trade'])
cold_storage = st.selectbox("Cold Storage", ['Yes', 'No'])
shelf_space = st.selectbox("Shelf Space", ['<20ft', '20-30ft', '30-40ft', '40-50ft', '>50ft'])

# Create a dictionary with the input values
input_data = {
    'No': no,
    'SKU_count': sku_count,
    'Avg_Sales': avg_sales,
    'Willingness_Organic': willingness_organic,
    'Willingness_Atta': willingness_atta,
    'Willingness_Ghee': willingness_ghee,
    'Willingness_Oil': willingness_oil,
    'Willingness_Beverage': willingness_beverage,
    'Willingness_Bakery': willingness_bakery,
    'Importance_of_AMUL': importance_of_amul,
    'Reputation_of_Store': reputation_of_store,
    'Self_Service': self_service,
    'Customer_Can_Browse': customer_can_browse,
    'Shop_Type': shop_type,
    'Cold_Storage': cold_storage,
    'Shelf_Space': shelf_space
}

# Convert input data to a DataFrame
input_df = pd.DataFrame([input_data])

# Perform one-hot encoding on the input data.
# Make sure the columns are in the same order as the training data after one-hot encoding.
categorical_cols = ['Self_Service', 'Customer_Can_Browse', 'Shop_Type', 'Cold_Storage', 'Shelf_Space']
input_df_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

# Reindex the input DataFrame to match the columns of the training data
# Add missing columns with a value of 0
for col in expected_columns:
    if col not in input_df_encoded.columns:
        input_df_encoded[col] = 0

# Ensure the order of columns in the input matches the training data
input_df_encoded = input_df_encoded[expected_columns]


# Scale the input data
#input_scaled = scaler.transform(input_df_encoded)

# Make a prediction
#prediction = model.predict(input_scaled)
#prediction_proba = model.predict_proba(input_scaled)
prediction = model.predict(input_df_encoded)
prediction_proba = model.predict_proba(input_df_encoded)
st.header("Prediction Result")

if st.button('Predict'):
    st.write(f"The predicted outlet classification is: **{prediction[0]}**")
    st.write("Prediction Probabilities:")
    proba_df = pd.DataFrame(prediction_proba, columns=model.classes_)
    st.write(proba_df)
