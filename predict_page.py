import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor_loaded = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

def show_predict_page():
    st.title("Software Developer Salary Predictions")
    st.write("""\t\tWe need some information to predict the salary""")
    countries = (
        "United States of America",
        "India",
        "United Kingdom",
        "Germany",
        "Canada",
        "Brazil",
        "France",
        "Spain",
        "Australia",
        "Netherlands",
        "Poland",
        "Italy",
        "Russian Federation",
        "Sweden",
    )

    education = (
        "Less than a Bachelors",
        "Bachelor’s degree",
        "Master’s degree",
        "Post grad",
    )

    country = st.selectbox("Country", countries)
    education_level = st.selectbox("Education", education)
    experience = st.slider("Years of Experience", 0, 50, 3)
    ok = st.button("Calculate Salary")
    
    if ok:
        # Prepare input data for prediction
        X = np.array([[countries.index(country), education.index(education_level), experience]])
        X = X.astype(float)

        # Combine labels from both training and test datasets
        all_country_labels = np.concatenate([le_country.classes_, X[:, 0]])
        all_education_labels = np.concatenate([le_education.classes_, X[:, 1]])

        # Convert labels to string type before fitting
        all_country_labels = all_country_labels.astype(str)
        all_education_labels = all_education_labels.astype(str)

        # Refit the LabelEncoders on the combined labels
        le_country.fit(all_country_labels)
        le_education.fit(all_education_labels)

        # Apply label encoding
        X[:, 0] = le_country.transform(X[:, 0])
        X[:, 1] = le_education.transform(X[:, 1])

        # Make prediction
        salary = regressor_loaded.predict(X)
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")
        st.subheader(f"The estimated Salary(In Indian Rupees) is Rs.{salary[0]*83.55:.0f}")
