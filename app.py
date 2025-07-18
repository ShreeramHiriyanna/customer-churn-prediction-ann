"""
# Customer Churn Predictor (ANN + Streamlit)

This interactive app uses a trained Artificial Neural Network to predict the probability of a customer churning based on demographics and account details. Built with TensorFlow and deployed via Streamlit.
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
from PIL import Image

# Display banner image
image = Image.open("banner.png")
st.image(image, use_column_width=True)

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
This app predicts the probability of a customer churning using an Artificial Neural Network trained on historical bank data. Enter the customer information below to get a churn prediction.
""")

# User input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance', min_value=0.0)
credit_score = st.number_input('Credit Score', min_value=0.0)
estimated_salary = st.number_input('Estimated Salary', min_value=0.0)
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

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

# Display input summary
st.subheader("Input Summary")
st.write(input_data)

# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

# Display probability and interpretation
st.write(f'**Churn Probability:** {prediction_proba:.2f}')

if prediction_proba > 0.75:
    st.success('High risk of churn.')
elif prediction_proba > 0.5:
    st.warning('Moderate risk of churn.')
else:
    st.info('Low risk of churn.')

# Model accuracy note
st.markdown("**Model Accuracy**: ~87% on test set")
