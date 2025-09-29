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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

st.set_page_config(page_title="Personal Expense Tracker", layout="wide")

# -----------------------------
# Helper Functions
# -----------------------------
MODEL_FILE = "expense_model.pkl"

def train_model(df):
    """Train text classifier to predict expense category"""
    df = df.dropna(subset=["description", "category"])
    X = df["description"]
    y = df["category"]
    
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    pipeline.fit(X, y)
    
    # Save the model
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
    
    # Linear regression
    X = weekly["week_num"].values.reshape(-1, 1)
    y = weekly["amount"].values
    
    if len(X) < 2:
        return weekly, pd.DataFrame()  # not enough data
    
    from sklearn.linear_model import LinearRegression
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
st.title("💸 Personal Expense Tracker with ML")
st.write("Upload your transactions CSV and let AI categorize + forecast your spending!")

# Upload section
uploaded_file = st.file_uploader("Upload transactions CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    if "date" not in df.columns or "description" not in df.columns or "amount" not in df.columns:
        st.error("CSV must have at least: date, description, amount, [category]")
    else:
        # Ensure correct dtypes
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        
        model = load_model()
        
        if "category" in df.columns and df["category"].notna().any():
            with st.expander("Train / Retrain Model"):
                if st.button("Train Model with Current Data"):
                    model = train_model(df)
                    st.success("✅ Model trained and saved!")
        
        if model:
            # Predict categories where missing
            missing_mask = df["category"].isna() if "category" in df.columns else np.ones(len(df), dtype=bool)
            if missing_mask.any():
                df_missing = df.loc[missing_mask, "description"]
                df.loc[missing_mask, "category"] = model.predict(df_missing)
            
            st.subheader("Predicted Transactions")
            st.dataframe(df.head(20))
            
            # Download button
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download categorized CSV", csv, "transactions_with_categories.csv", "text/csv")
            
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
            st.warning("⚠️ No model trained yet. Upload data with 'category' column to train.")
else:
    st.info("Please upload a CSV file to begin. Example columns: date, description, amount, category")
```
