import matplotlib
matplotlib.use("Agg")

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="Iris Naive Bayes Classifier", layout="wide")

st.title("Naive Bayes Classification on Iris Dataset")

st.write("This app trains a Naive Bayes model using the Iris dataset.")

# ------------------------------
# Load Dataset
# ------------------------------
try:
    df = pd.read_csv("iris.csv")
    st.success("iris.csv loaded successfully")
except:
    st.error("iris.csv not found in project directory.")
    st.stop()

# Remove unnamed columns if present
df = df.loc[:, ~df.columns.str.contains("Unnamed")]

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.write(f"Dataset Shape: {df.shape}")

# ------------------------------
# Target and Features
# ------------------------------
target_column = "species"

if target_column not in df.columns:
    st.error("Target column 'species' not found in dataset.")
    st.stop()

feature_columns = [col for col in df.columns if col != target_column]

st.subheader("Features Used")
st.write(feature_columns)

# ------------------------------
# Train/Test Settings
# ------------------------------
test_size = st.slider("Test Size (%)", 10, 50, 20) / 100
random_state = st.number_input("Random State", value=42, step=1)

# ------------------------------
# Train Model
# ------------------------------
if st.button("Train Model"):

    try:
        X = df[feature_columns].copy()
        y = df[target_column].copy()

        # Encode target
        le = LabelEncoder()
        y = le.fit_transform(y)

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=int(random_state)
        )

        # Train Model
        model = GaussianNB()
        model.fit(X_train, y_train)

        # Prediction
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        # ------------------------------
        # Results
        # ------------------------------
        st.subheader("Model Accuracy")
        st.metric("Accuracy", f"{accuracy:.4f} ({accuracy*100:.2f}%)")

        # ------------------------------
        # Confusion Matrix
        # ------------------------------
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        st.pyplot(fig)

        # Table version
        cm_df = pd.DataFrame(
            cm,
            columns=[f"Pred {i}" for i in range(cm.shape[1])],
            index=[f"Actual {i}" for i in range(cm.shape[0])]
        )

        st.dataframe(cm_df)

        # ------------------------------
        # Classification Report
        # ------------------------------
        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))

    except Exception as e:
        st.error(f"Model error: {e}")