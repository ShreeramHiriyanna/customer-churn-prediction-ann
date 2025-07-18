"""
# Customer Churn Predictor (ANN + Streamlit)

An interactive web app built with Streamlit and TensorFlow that predicts the probability of a customer leaving (churning) based on demographic and account-related features. The model is trained using an Artificial Neural Network on historical banking data.
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load trained model and encoders
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# App title and description
st.title('Customer Churn Prediction')
st.markdown("""
This app predicts the probability of a customer churning using an Artificial Neural Network trained on historical bank data.  
Please enter the customer information below.
""")

# User input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance', min_value=0.0)
credit_score = st.number_input('Credit Score', min_value=0.0)
estimated_salary = st.number_input('Estimated Salary', min_value=0.0)
tenure = st.slider('Tenure (Years with Bank)', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)

has_cr_card_label = st.selectbox('Has Credit Card', ['No (0)', 'Yes (1)'])
has_cr_card = 0 if has_cr_card_label == 'No (0)' else 1
st.caption("0 = Customer doesn't own a credit card, 1 = Owns a credit card")

is_active_member_label = st.selectbox('Is Active Member', ['No (0)', 'Yes (1)'])
is_active_member = 0 if is_active_member_label == 'No (0)' else 1
st.caption("0 = Not actively using bank services, 1 = Actively engaged with the bank")

# Prepare input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode Geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the data
input_data_scaled = scaler.transform(input_data)

# Input summary
st.subheader("Input Summary")
st.write(input_data)

# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

# Show result
st.subheader("Prediction Result")
st.write(f'**Churn Probability:** {prediction_proba:.2f}')
st.caption("Higher values indicate higher likelihood of the customer leaving the bank.")

if prediction_proba > 0.75:
    st.error('High risk of churn. Immediate action may be needed.')
elif prediction_proba > 0.5:
    st.warning('Moderate risk of churn. Customer may be uncertain.')
else:
    st.success('Low risk of churn. Customer is likely to stay.')

# Footer
st.markdown("**Model Accuracy**: ~87% on test data")
