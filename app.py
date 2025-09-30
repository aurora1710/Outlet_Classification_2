import streamlit as st
import pickle
import pandas as pd

# Load the trained Logistic Regression model
with open('logistic_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Assuming you have the list of feature names used during training
# Replace this with the actual list of feature names from your training data (X.columns)
feature_names = ['SKU_count', 'Avg_Sales', 'Willingness_Organic', 'Willingness_Atta',
                 'Willingness_Ghee', 'Willingness_Oil', 'Willingness_Beverage',
                 'Willingness_Bakery', 'Importance_of_AMUL', 'Reputation_of_Store',
                 'Self_Service_No', 'Self_Service_Yes', 'Customer_Can_Browse_Full View and Choice',
                 'Customer_Can_Browse_No Entry', 'Shop_Type _Convenience store',
                 'Shop_Type _General trade', 'Shop_Type _Modern trade', 'Cold_Storage_No',
                 'Cold_Storage_Yes', 'Shelf_Space_20-30ft', 'Shelf_Space_30-40ft',
                 'Shelf_Space_40-50ft', 'Shelf_Space_<20ft', 'Shelf_Space_>50ft']


st.title('Outlet Quality Classification')

st.write('Enter the details of the outlet to predict its quality.')

# Create input fields for each feature
input_data = {}
for feature in feature_names:
    # You might need to customize the input type based on the feature's data type
    if feature in ['SKU_count', 'Avg_Sales', 'Willingness_Organic', 'Willingness_Atta', 'Willingness_Ghee', 'Willingness_Oil', 'Willingness_Beverage', 'Willingness_Bakery', 'Importance_of_AMUL', 'Reputation_of_Store']:
        input_data[feature] = st.number_input(f'Enter {feature}', value=0)
    elif feature in ['Self_Service_No', 'Self_Service_Yes', 'Cold_Storage_No', 'Cold_Storage_Yes']:
        input_data[feature] = st.selectbox(f'Select {feature.replace("_", " ").replace("No", "No").replace("Yes", "Yes")}', [0, 1])
    elif feature in ['Customer_Can_Browse_Full View and Choice', 'Customer_Can_Browse_No Entry']:
         input_data[feature] = st.selectbox(f'Select {feature.replace("Customer_Can_Browse_", "").replace("Full View and Choice", "Full View and Choice").replace("No Entry", "No Entry")}', [0, 1])
    elif feature in ['Shop_Type _Convenience store', 'Shop_Type _General trade', 'Shop_Type _Modern trade']:
         input_data[feature] = st.selectbox(f'Select {feature.replace("Shop_Type _", "")}', [0, 1])
    elif feature in ['Shelf_Space_20-30ft', 'Shelf_Space_30-40ft', 'Shelf_Space_40-50ft', 'Shelf_Space_<20ft', 'Shelf_Space_>50ft']:
         input_data[feature] = st.selectbox(f'Select {feature.replace("Shelf_Space_", "")}', [0, 1])
    else:
        input_data[feature] = st.text_input(f'Enter {feature}')


if st.button('Predict'):
    # Create a DataFrame from the input data
    input_df = pd.DataFrame([input_data])

    # Ensure the column order is the same as during training
    input_df = input_df[feature_names]

    # Make prediction
    prediction = model.predict(input_df)

    # Display the prediction
    # Assuming your target is encoded as 'Non-Premium' and 'Premium'
    predicted_class = 'Premium' if prediction[0] == 'Premium' else 'Non-Premium'
    st.write(f'The predicted outlet quality is: {predicted_class}')
