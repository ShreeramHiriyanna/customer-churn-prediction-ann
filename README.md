# Customer Churn Prediction (ANN + Streamlit)

An interactive web app built using Streamlit and a trained Artificial Neural Network (ANN) to predict the likelihood of a customer churning based on banking data.

[Launch the Live App](https://your-streamlit-app-link-here) <!-- Replace with actual URL -->

---

## Project Summary

This project walks through the end-to-end process of building and deploying a neural network classification model:

### 1. Data Preprocessing
- Loaded customer bank data and handled missing values
- Encoded categorical features:
  - `Gender`: Label Encoding
  - `Geography`: One-Hot Encoding
- Scaled numerical features using StandardScaler

### 2. Model Development
- Built an Artificial Neural Network (ANN) using TensorFlow/Keras
- Trained model on historical customer data to predict churn (binary classification)
- Evaluated using metrics such as accuracy, precision, recall, and F1-score
- Achieved approximately 85% accuracy on test data

### 3. Serialization
- Saved the trained model (`model.h5`) and preprocessing tools (`scaler.pkl`, `label_encoder_gender.pkl`, `onehot_encoder_geo.pkl`) for deployment

### 4. Streamlit App
- Developed a front-end interface using Streamlit for real-time predictions
- Users input demographic and financial data
- App displays:
  - Churn probability (0 to 1)
  - A clear risk interpretation (Low / Moderate / High)
  - Summary of user inputs

---

## Files Overview

| File                          | Description |
|------------------------------|-------------|
| `app.py`                     | Streamlit UI for real-time prediction |
| `model.h5`                   | Trained ANN model |
| `label_encoder_gender.pkl`   | LabelEncoder for `Gender` |
| `onehot_encoder_geo.pkl`     | OneHotEncoder for `Geography` |
| `scaler.pkl`                 | Scaler used for feature normalization |
| `experiments_classification.ipynb` | Full notebook with EDA, preprocessing, model training and evaluation |
| `requirements.txt`           | Python dependencies used during development |

---

## Future Improvements
- Add SHAP or LIME for feature importance
- Extend to multi-page Streamlit app
- Deploy salary regression model as a companion app

---

## License

This project is open-source under the MIT License.
