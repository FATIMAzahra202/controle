import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load data
df = pd.read_csv("Heart_Disease_Prediction.csv")

# Features / Target
X = df.drop("target", axis=1)
y = df["target"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train
model = LogisticRegression()
model.fit(X_train, y_train)

# Interface
st.title("Heart Disease Prediction ❤️")

age = st.number_input("Age", 20, 100, 50)
chol = st.number_input("Cholesterol", 100, 400, 200)
thalach = st.number_input("Max Heart Rate", 60, 220, 150)

# Predict
if st.button("Predict"):

    data = pd.DataFrame([[age, chol, thalach]], 
                        columns=["age", "chol", "thalach"])

    for col in X.columns:
        if col not in data.columns:
            data[col] = 0

    data = data[X.columns]

    data = scaler.transform(data)

    result = model.predict(data)

    if result[0] == 1:
        st.error("High risk ❌")
    else:
        st.success("Low risk ✅")