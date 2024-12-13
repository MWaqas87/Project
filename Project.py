import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import streamlit as st

# Define the clean_sm function
def clean_sm(x):
    return np.where(x == 1, 1, 0)

# Load and preprocess data
def load_data():
    # Replace with the actual dataset path
    df = pd.read_csv("social_media_usage.csv")

    # Create target variable and clean features
    df['sm_li'] = clean_sm(df['web1h'])  # LinkedIn usage
    df['income'] = df['income'].apply(lambda x: x if x <= 9 else np.nan)  # Income
    df['education'] = df['educ2'].apply(lambda x: x if x <= 8 else np.nan)  # Education
    df['parent'] = clean_sm(df['par'])  # Parent (binary)
    df['married'] = clean_sm(df['marital'].apply(lambda x: 1 if x == 1 else 0))  # Married (binary)
    df['female'] = clean_sm(df['gender'].apply(lambda x: 1 if x == 2 else 0))  # Female (binary)
    df['age'] = df['age'].apply(lambda x: x if x <= 98 else np.nan)  # Age

    # Select columns of interest and drop missing values
    columns_of_interest = ['sm_li', 'income', 'education', 'parent', 'married', 'female', 'age']
    ss = df[columns_of_interest].dropna()

    # Split into features and target
    X = ss.drop(columns=['sm_li'])
    y = ss['sm_li']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train logistic regression model
def train_model(X_train, y_train):
    model = LogisticRegression(class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    return model

# Streamlit application
def main():
    st.title("LinkedIn Usage Prediction")

    # User inputs
    income = st.slider("Income Level (1-9)", min_value=1, max_value=9, value=5)
    education = st.slider("Education Level (1-8)", min_value=1, max_value=8, value=4)
    parent = st.selectbox("Are you a parent?", options=["No", "Yes"], index=0)
    married = st.selectbox("Are you married?", options=["No", "Yes"], index=0)
    female = st.selectbox("What is your gender?", options=["Male", "Female"], index=0)
    age = st.slider("Age", min_value=18, max_value=98, value=30)

    # Convert user inputs to model-ready format
    user_input = pd.DataFrame({
        'income': [income],
        'education': [education],
        'parent': [1 if parent == "Yes" else 0],
        'married': [1 if married == "Yes" else 0],
        'female': [1 if female == "Female" else 0],
        'age': [age]
    })

    # Load data and train model
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)

    # Predict probability for user input
    probabilities = model.predict_proba(user_input)
    prob_non_user, prob_user = probabilities[0]

    # Display results
    st.write("### Prediction Results")
    st.write(f"Probability of being a Non-LinkedIn User: {prob_non_user:.2f}")
    st.write(f"Probability of being a LinkedIn User: {prob_user:.2f}")

if __name__ == "__main__":
    main()
