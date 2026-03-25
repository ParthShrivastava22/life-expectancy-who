# main.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -------------------------------
# Custom Transformer (IMPORTANT)
# -------------------------------
# This must match your training code

from sklearn.base import BaseEstimator, TransformerMixin

class CustomFeatureAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Drop Country column
        if "Country" in X.columns:
            X = X.drop("Country", axis=1)
        
        # Add engineered features
        X["edu_income_interaction"] = (
            X["Schooling"] * X["Income composition of resources"]
        )
        
        X["nutrition_gap"] = (
            X["BMI"] - X["thinness  1-19 years"]
        )
        
        return X


# -------------------------------
# Load Model
# -------------------------------

@st.cache_resource
def load_model():
    return joblib.load("life_expectancy_model.joblib")

model = load_model()


# -------------------------------
# Streamlit UI
# -------------------------------

st.title("🌍 Life Expectancy Predictor")

st.write("Enter the details below to predict life expectancy.")

# --- Inputs ---
country = st.text_input("Country", "India")
status = st.selectbox("Status", ["Developing", "Developed"])
year = st.number_input("Year", 2000, 2015, 2015)

adult_mortality = st.number_input("Adult Mortality", value=150.0)
infant_deaths = st.number_input("Infant Deaths", value=10.0)
alcohol = st.number_input("Alcohol", value=2.0)
percentage_expenditure = st.number_input("Health Expenditure (%)", value=100.0)

hepatitis_b = st.number_input("Hepatitis B (%)", value=80.0)
measles = st.number_input("Measles Cases", value=100.0)
bmi = st.number_input("BMI", value=20.0)
under_five_deaths = st.number_input("Under-5 Deaths", value=15.0)

polio = st.number_input("Polio (%)", value=80.0)
total_expenditure = st.number_input("Total Expenditure", value=5.0)
diphtheria = st.number_input("Diphtheria (%)", value=80.0)

hiv_aids = st.number_input("HIV/AIDS", value=0.1)
gdp = st.number_input("GDP", value=1000.0)
population = st.number_input("Population", value=1000000.0)

thinness_5_9 = st.number_input("Thinness 5-9 Years", value=5.0)
thinness_1_19 = st.number_input("Thinness 1-19 Years", value=5.0)

schooling = st.number_input("Schooling", value=12.0)
income_composition = st.number_input("Income Composition", value=0.6)


# -------------------------------
# Prediction Button
# -------------------------------

if st.button("Predict Life Expectancy"):

    # Create input DataFrame (VERY IMPORTANT: column names must match training)
    input_data = pd.DataFrame([{
        "Country": country,
        "Year": year,
        "Status": status,
        "Adult Mortality": adult_mortality,
        "infant deaths": infant_deaths,
        "Alcohol": alcohol,
        "percentage expenditure": percentage_expenditure,
        "Hepatitis B": hepatitis_b,
        "Measles": measles,
        "BMI": bmi,
        "under-five deaths": under_five_deaths,
        "Polio": polio,
        "Total expenditure": total_expenditure,
        "Diphtheria": diphtheria,
        "HIV/AIDS": hiv_aids,
        "GDP": gdp,
        "Population": population,
        "thinness 5-9 years": thinness_5_9,
        "thinness  1-19 years": thinness_1_19,
        "Schooling": schooling,
        "Income composition of resources": income_composition
    }])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Display result
    st.success(f"Predicted Life Expectancy: {prediction:.2f} years")