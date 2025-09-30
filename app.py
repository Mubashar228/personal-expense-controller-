import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Advanced Expense Tracker", layout="wide")

MODEL_PATH = "category_model.joblib"

# -------------------------
# Keyword dictionary (expandable)
# -------------------------
KEYWORDS = {
    "Food": ["tea", "coffee", "lunch", "dinner", "pizza", "burger", "snack", "restaurant", "meal", "biryani", "kfc", "mcdonald", "cafe"],
    "Transport": ["uber", "careem", "bus", "taxi", "ride", "petrol", "fuel", "car", "rickshaw", "metro", "uberx"],
    "Bills": ["electricity", "gas", "internet", "wifi", "water", "bill", "mobile", "phone", "recharge", "subscription"],
    "Entertainment": ["movie", "cinema", "netflix", "game", "concert", "show", "music", "ticket"],
    "Groceries": ["milk", "bread", "egg", "vegetable", "fruit", "rice", "flour", "atta", "oil", "sugar", "grocery", "supermarket"],
    "Shopping": ["clothes", "shoes", "watch", "mobile", "laptop", "bag", "makeup", "jewelry", "store", "online shopping"],
    "Health": ["doctor", "medicine", "hospital", "clinic", "pharmacy", "tablet", "checkup"],
    "Education": ["book", "pen", "notebook", "school", "college", "university", "course", "fee", "tuition"],
    "Transport-Fuel": ["pump", "petrolpump", "petrolpump", "filling", "diesel"],
    "Other": []
}

# Flatten keywords -> quick lookup (word -> category)
KEYWORD_TO_CAT = {}
for cat, words in KEYWORDS.items():
    for w in words:
        KEYWORD_TO_CAT[w.lower()] = cat

# -------------------------
# Helpers: data & model
# -------------------------
def init_session():
    if "expenses" not in st.session_state:
        st.session_state["expenses"] = pd.DataFrame(columns=["date", "description", "amount", "category"])
    if "model" not in st.session_state:
        st.session_state["model"] = load_model_if_exists()

def load_model_if_exists():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            return None
    return None

def save_model(pipe):
    joblib.dump(pipe, MODEL_PATH)

def keyword_category(text):
    if not isinstance(text, str):
        return None
    t = text.lower()
    # simple tokenization by splitting on non-alpha characters
    tokens = [tok.strip() for tok in ''.join(ch if ch.isalnum() else ' ' for ch in t).split() if tok.strip()]
    for tok in tokens:
        if tok in KEYWORD_TO_CAT:
            return KEYWORD_TO_CAT[tok]
    # try substring matches for multi-word tokens present in keywords
    for kw, cat in KEYWORD_TO_CAT.items():
        if kw in t:
            return cat
    return None

def train_category_pipeline(df):
    # train only on rows with non-null category and description
    train_df = df.dropna(subset=["description", "category"]).copy()
    if train_df.empty or train_df["category"].nunique() < 2:
        return None, "Not enough labeled data (need >=2 distinct categories)."
    X = train_df["description"].astype(str)
    y = train_df["category"].astype(str)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=1)),
        ("nb", MultinomialNB())
    ])
    pipe.fit(X, y)
    save_model(pipe)
    st.session_state["model"] = pipe
    return pipe, f"Trained on {len(train_df)} rows. Classes: {sorted(train_df['category'].unique())}"

def hybrid_predict(desc):
    # priority: keywords -> model -> Unknown
    if pd.isna(desc) or str(desc).strip()=="":
        return None
    kw = keyword_category(desc)
    if kw:
        return kw
    model = st.session_state.get("model", None)
    if model is not None:
        try:
            pred = model.predict([str(desc)])[0]
            return pred
        except Exception:
            return None
    return None

# -------------------------
# Forecast helpers
# -------------------------
def monthly_aggregate(df):
    if df.empty:
        return pd.DataFrame(columns=["month","amount"])
    tmp = df.copy()
    tmp['date'] = pd.to_datetime(tmp['date'])
    tmp = tmp.set_index('date')
    m = tmp['amount'].resample('M').sum().reset_index()
    m['month'] = m['date'].dt.to_period('M').astype(str)
    m = m[['month','date','amount']]
    return m

def forecast_months(monthly_df, n_months=1):
    if monthly_df.empty or len(monthly_df) < 2:
        # fallback: use mean of existing months or zero
        mean_val = float(monthly_df['amount'].mean()) if not monthly_df.empty else 0.0
        future = []
        last_date = monthly_df['date'].max() if not monthly_df.empty else pd.Timestamp(datetime.today()).replace(day=1)
        for i in range(1, n_months+1):
            next_dt = (last_date + pd.DateOffset(months=i)).replace(day=1) + pd.offsets.MonthEnd(0)
            future.append({"date": next_dt, "amount": mean_val})
        return pd.DataFrame(future)
    # prepare numeric x = 0..len-1
    X = np.arange(len(monthly_df)).reshape(-1,1)
    y = monthly_df['amount'].values
    lr = LinearRegression()
    lr.fit(X, y)
    future_idx = np.arange(len(monthly_df), len(monthly_df)+n_months).reshape(-1,1)
    preds = lr.predict(future_idx)
    last_date = monthly_df['date'].max()
    future = []
    for i, p in enumerate(preds, start=1):
        next_dt = (last_date + pd.DateOffset(months=i)).replace(day=1) + pd.offsets.MonthEnd(0)
        future.append({"date": next_dt, "amount": float(max(0, p))})
    return pd.DataFrame(future)

# -------------------------
# UI: initialization
# -------------------------
init_session()
st.title("📊 Advanced Personal Expense Tracker")
st.markdown("Enter expenses manually, see monthly history, category breakdown, and predict next months' spending.")

col1, col2 = st.columns([2,1])

with col1:
    st.subheader("Add Expense")
    with st.form("add_expense_form", clear_on_submit=True):
        date = st.date_input("Date", value=pd.Timestamp.today().date())
        description = st.text_input("Description (e.g., 'Uber ride', 'Electricity bill')", max_chars=200)
        amount = st.number_input("Amount", min_value=0.0, step=50.0, format="%.2f")
        cat_input = st.text_input("Category (optional) — leave blank for auto-predict", value="")
        submitted = st.form_submit_button("Add entry")
        if submitted:
            new_row = {
                "date": pd.to_datetime(date),
                "description": str(description).strip(),
                "amount": float(amount),
                "category": cat_input.strip() if cat_input.strip() != "" else None
            }
            st.session_state["expenses"] = pd.concat([
                st.session_state["expenses"],
                pd.DataFrame([new_row])
            ], ignore_index=True)
            st.success("Expense added ✅")

    st.markdown("**Quick Actions**")
    if st.button("Clear all expenses (reset)"):
        st.session_state["expenses"] = pd.DataFrame(columns=["date","description","amount","category"])
        st.session_state["model"] = load_model_if_exists()
        st.success("All data cleared")

with col2:
    st.subheader("Model / Training")
    st.write("Hybrid strategy: keyword-match first, then ML (Naive Bayes) fallback.")
    if st.session_state.get("model") is not None:
        st.success("Category model loaded from disk.")
    else:
        st.info("No saved category model found.")
    if st.button("Train category model now"):
        pipe, message = train_category_pipeline(st.session_state["expenses"])
        if pipe is None:
            st.warning(message)
        else:
            st.success(message)
    if st.button("Delete saved model"):
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
            st.session_state["model"] = None
            st.info("Saved model deleted.")
        else:
            st.info("No saved model to delete.")

st.markdown("---")

# -------------------------
# Show expenses table with predicted categories (hybrid)
# -------------------------
df = st.session_state["expenses"].copy()
if not df.empty:
    st.subheader("All Entries")
    # Apply hybrid prediction in-memory for display (do not overwrite original unless user wants)
    display_df = df.copy()
    display_df['predicted_category'] = display_df.apply(
        lambda r: r['category'] if pd.notna(r.get('category')) and str(r.get('category')).strip() != "" else hybrid_predict(r['description']),
        axis=1
    )
    display_df['date'] = pd.to_datetime(display_df['date'])
    # sort desc
    display_df = display_df.sort_values('date', ascending=False).reset_index(drop=True)
    st.dataframe(display_df.style.format({"amount":"{:.2f}"}))

    # allow user to commit predicted categories to actual category column
    if st.button("Apply predicted categories to missing rows"):
        applied = 0
        for idx, row in st.session_state["expenses"].iterrows():
            if pd.isna(row.get("category")) or str(row.get("category")).strip()=="":
                pred = hybrid_predict(row.get("description"))
                if pred:
                    st.session_state["expenses"].at[idx, "category"] = pred
                    applied += 1
        st.success(f"Applied predicted categories to {applied} rows.")

    # download CSV
    csv = st.session_state["expenses"].to_csv(index=False).encode("utf-8")
    st.download_button("Download expenses CSV", data=csv, file_name="expenses.csv", mime="text/csv")
else:
    st.info("No expenses yet — add entries using the form above.")

st.markdown("---")

# -------------------------
# Monthly aggregation, history, charts
# -------------------------
st.subheader("Monthly Summary & History")
monthly = monthly_aggregate(st.session_state["expenses"])

if monthly.empty:
    st.info("No monthly data to show yet. Add expenses across at least one month.")
else:
    # show month selector
    months = monthly['month'].tolist()
    selected_month = st.selectbox("Select month to view details", options=months[::-1])  # most recent first
    month_rows = st.session_state["expenses"][pd.to_datetime(st.session_state["expenses"]["date"]).dt.to_period("M").astype(str) == selected_month].copy()
    st.markdown(f"**Summary for {selected_month}:**")
    total_month = float(month_rows['amount'].sum())
    st.metric(label=f"Total spend ({selected_month})", value=f"{total_month:.2f}")

    # category breakdown for that month
    cat_breakdown = month_rows.groupby('category')['amount'].sum().sort_values(ascending=False)
    if not cat_breakdown.empty:
        fig1, ax1 = plt.subplots(figsize=(6,4))
        ax1.pie(cat_breakdown.values, labels=cat_breakdown.index, autopct='%1.1f%%', startangle=140)
        ax1.axis('equal')
        st.pyplot(fig1)
        st.write("Category breakdown (amount):")
        st.dataframe(cat_breakdown.reset_index().rename(columns={0:"amount"}).style.format({"amount":"{:.2f}"}))
    else:
        st.info("No category data for this month (categories may be missing).")

    # show rows for selected month
    st.write("Records for selected month:")
    st.dataframe(month_rows.sort_values('date', ascending=False).reset_index(drop=True).style.format({"amount":"{:.2f}"}))

    # show monthly trend
    st.write("Monthly trend (last 12 months):")
    show_months = monthly.tail(12)
    fig2, ax2 = plt.subplots(figsize=(8,4))
    ax2.plot(show_months['date'], show_months['amount'], marker='o', label='Actual')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Amount')
    ax2.set_title('Monthly Spending (last 12 months)')
    ax2.tick_params(axis='x', rotation=30)
    ax2.grid(alpha=0.3)
    st.pyplot(fig2)

st.markdown("---")

# -------------------------
# Forecasting UI
# -------------------------
st.subheader("Multi-month Forecast")
with st.form("forecast_form"):
    n_months = st.number_input("Predict how many future months?", min_value=1, max_value=12, value=1)
    forecast_btn = st.form_submit_button("Forecast")
    if forecast_btn:
        monthly_df = monthly
        if monthly_df.empty:
            st.warning("Not enough monthly data — add expenses for at least one month.")
        else:
            future_df = forecast_months(monthly_df, n_months)
            st.success(f"Predicted next {n_months} month(s).")
            st.dataframe(future_df.style.format({"amount":"{:.2f}"}))
            # plot combined
            comb_dates = pd.concat([monthly_df[['date','amount']], future_df[['date','amount']]])
            fig3, ax3 = plt.subplots(figsize=(8,4))
            ax3.plot(monthly_df['date'], monthly_df['amount'], marker='o', label='Actual')
            ax3.plot(future_df['date'], future_df['amount'], marker='x', linestyle='--', label='Forecast')
            ax3.set_title('Actual + Forecast (monthly)')
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Amount')
            ax3.legend()
            ax3.grid(alpha=0.3)
            st.pyplot(fig3)

st.markdown("---")

# -------------------------
# Tips & next steps
# -------------------------
st.subheader("Tips to improve predictions")
st.markdown("""
- Enter categories manually for a few dozen rows across different categories — then train the model.
- Use descriptive text in `description` (e.g., \"Uber Ride - Airport\" instead of \"Payment\").
- If you have many historical transactions, upload them as CSV and then train the model (you can add upload feature later).
- For better forecasting you can integrate Prophet (seasonality) or use more months of history.
""")

# End of app
