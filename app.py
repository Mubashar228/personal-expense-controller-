```python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Personal Expense Tracker", layout="wide")

MODEL_FILE = "expense_model.pkl"

# -----------------------------
# Helper Functions
# -----------------------------
def train_model(df):
    """Train text classifier to predict expense category"""
    df = df.dropna(subset=["description", "category"])
    if df.empty:
        return None
    X = df["description"]
    y = df["category"]
    
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    pipeline.fit(X, y)
    
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(pipeline, f)
    
    return pipeline

def load_model():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            return pickle.load(f)
    return None

def forecast_expenses(df):
    """Simple linear trend forecast for weekly expenses"""
    df["date"] = pd.to_datetime(df["date"])
    weekly = df.groupby(pd.Grouper(key="date", freq="W"))["amount"].sum().reset_index()
    weekly["week_num"] = np.arange(len(weekly))
    
    if len(weekly) < 2:
        return weekly, pd.DataFrame()
    
    X = weekly["week_num"].values.reshape(-1, 1)
    y = weekly["amount"].values
    model = LinearRegression().fit(X, y)
    
    future_weeks = np.arange(len(weekly), len(weekly) + 4).reshape(-1, 1)
    preds = model.predict(future_weeks)
    
    forecast_df = pd.DataFrame({
        "date": pd.date_range(weekly["date"].iloc[-1] + pd.Timedelta(weeks=1), periods=4, freq="W"),
        "amount": preds
    })
    return weekly, forecast_df

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("💸 Personal Expense Tracker (User Input)")
st.write("Add your daily expenses below. AI will categorize (if missing) and forecast your spending!")

# Initialize session_state dataframe
if "expenses" not in st.session_state:
    st.session_state["expenses"] = pd.DataFrame(columns=["date", "description", "amount", "category"])

# Input form
with st.form("expense_form"):
    date = st.date_input("Date")
    description = st.text_input("Description (e.g., Tea, Uber Ride, Electricity Bill)")
    amount = st.number_input("Amount", min_value=0.0, step=50.0)
    category = st.selectbox("Category (optional)", ["", "Food", "Transport", "Bills", "Entertainment", "Groceries", "Other"])
    submitted = st.form_submit_button("Add Expense")
    
    if submitted:
        new_entry = {
            "date": pd.to_datetime(date),
            "description": description,
            "amount": amount,
            "category": category if category != "" else None
        }
        st.session_state["expenses"] = pd.concat([st.session_state["expenses"], pd.DataFrame([new_entry])], ignore_index=True)
        st.success("Expense added!")

df = st.session_state["expenses"]

if not df.empty:
    st.subheader("Your Expenses")
    st.dataframe(df)
    
    # Train model if categories exist
    model = None
    if df["category"].notna().any():
        model = train_model(df)
    
    # Predict missing categories
    if model is not None:
        missing_mask = df["category"].isna()
        if missing_mask.any():
            df.loc[missing_mask, "category"] = model.predict(df.loc[missing_mask, "description"])
    
    # Download button
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Expense Data", csv, "expenses.csv", "text/csv")
    
    # Forecast
    st.subheader("📈 Expense Forecast")
    weekly, forecast_df = forecast_expenses(df)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(weekly["date"], weekly["amount"], marker="o", label="Actual")
    if not forecast_df.empty:
        ax.plot(forecast_df["date"], forecast_df["amount"], marker="x", linestyle="--", label="Forecast")
    ax.set_title("Weekly Expenses")
    ax.set_xlabel("Date")
    ax.set_ylabel("Amount")
    ax.legend()
    st.pyplot(fig)
else:
    st.info("Please add some expenses to see results.")
```
