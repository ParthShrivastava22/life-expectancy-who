import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


# -------------------------------
# Required for loading model
# -------------------------------

def encode_status(x):
    return (x == "Developed").astype(int)


class CustomFeatureAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if "Country" in X.columns:
            X = X.drop("Country", axis=1)

        X["edu_income_interaction"] = (
            X["Schooling"] * X["Income composition of resources"]
        )

        X["nutrition_gap"] = X["BMI"] - X["thinness  1-19 years"]

        return X


# -------------------------------
# Load Model
# -------------------------------

@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "life_expectation_model.joblib")
    return joblib.load(model_path)


model = load_model()


# -------------------------------
# UI Starts Here
# -------------------------------

st.set_page_config(page_title="Life Expectancy Predictor", layout="centered")

st.title("🌍 Life Expectancy Predictor")

st.markdown("""
This app predicts **Life Expectancy (in years)** using health, economic, and demographic indicators.

Fill in the values below to get a prediction.
""")

# -------------------------------
# Basic Info
# -------------------------------

st.header("📌 General Information")

col1, col2 = st.columns(2)

with col1:
    country = st.text_input("Country", "India")

with col2:
    status = st.selectbox("Development Status", ["Developing", "Developed"])

year = st.number_input(
    "Year",
    min_value=2000,
    max_value=2015,
    value=2015
)

# -------------------------------
# Health Indicators
# -------------------------------

st.header("🩺 Health Indicators")

col1, col2 = st.columns(2)

with col1:
    adult_mortality = st.number_input(
        "Adult Mortality (per 1000 population)",
        min_value=0.0,
        value=150.0,
        help="Probability of dying between ages 15–60 per 1000 population"
    )

    infant_deaths = st.number_input(
        "Infant Deaths (per 1000 population)",
        min_value=0.0,
        value=10.0,
        help="Number of infant deaths per 1000 population"
    )

    measles = st.number_input(
        "Measles Cases (per 1000 population)",
        min_value=0.0,
        value=100.0,
        help="Reported measles cases per 1000 population"
    )

    hiv_aids = st.number_input(
        "HIV/AIDS Prevalence (%)",
        min_value=0.0,
        value=0.1,
        help="Percentage of population living with HIV/AIDS"
    )

with col2:
    bmi = st.number_input(
        "Average BMI",
        min_value=0.0,
        max_value=100.0,
        value=20.0,
        help="Average Body Mass Index of population"
    )

    under_five_deaths = st.number_input(
        "Under-5 Deaths (per 1000 population)",
        min_value=0.0,
        value=15.0,
        help="Number of deaths of children under age 5 per 1000 population"
    )

    thinness_5_9 = st.number_input(
        "Thinness (5–9 years) (%)",
        min_value=0.0,
        value=5.0
    )

    thinness_1_19 = st.number_input(
        "Thinness (1–19 years) (%)",
        min_value=0.0,
        value=5.0
    )

# -------------------------------
# Vaccination & Lifestyle
# -------------------------------

st.header("💉 Vaccination & Lifestyle")

col1, col2 = st.columns(2)

with col1:
    hepatitis_b = st.number_input(
        "Hepatitis B Immunization (%)",
        min_value=0.0,
        max_value=100.0,
        value=80.0
    )

    polio = st.number_input(
        "Polio Immunization (%)",
        min_value=0.0,
        max_value=100.0,
        value=80.0
    )

with col2:
    diphtheria = st.number_input(
        "Diphtheria Immunization (%)",
        min_value=0.0,
        max_value=100.0,
        value=80.0
    )

    alcohol = st.number_input(
        "Alcohol Consumption (litres per capita)",
        min_value=0.0,
        value=2.0,
        help="Average alcohol consumption per adult (15+)"
    )

# -------------------------------
# Economic Indicators
# -------------------------------

st.header("📊 Economic Indicators")

col1, col2 = st.columns(2)

with col1:
    gdp = st.number_input(
        "GDP per Capita (USD)",
        min_value=0.0,
        value=1000.0
    )

    percentage_expenditure = st.number_input(
        "Health Expenditure (% of GDP)",
        min_value=0.0,
        value=100.0
    )

with col2:
    population = st.number_input(
        "Population",
        min_value=0.0,
        value=1000000.0
    )

    total_expenditure = st.number_input(
        "Total Expenditure (%)",
        min_value=0.0,
        value=5.0
    )

# -------------------------------
# Education
# -------------------------------

st.header("🎓 Education & Development")

schooling = st.number_input(
    "Schooling (years)",
    min_value=0.0,
    value=12.0
)

income_composition = st.number_input(
    "Income Composition of Resources (0–1)",
    min_value=0.0,
    max_value=1.0,
    value=0.6
)

# -------------------------------
# Prediction
# -------------------------------

if st.button("🚀 Predict Life Expectancy"):

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

    prediction = model.predict(input_data)[0]

    st.success(f"🎯 Predicted Life Expectancy: {prediction:.2f} years")
    st.caption("Typical global range: 40 – 90 years")