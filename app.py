# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from io import StringIO

# Forecasting libraries (Prophet optional)
USE_PROPHET = False
try:
    from prophet import Prophet
    USE_PROPHET = True
except Exception:
    from sklearn.linear_model import LinearRegression

# -------------------------
# Helpers & Initialization
# -------------------------
st.set_page_config(page_title="Expense Tracker — Budget & Forecast", layout="wide")

def init_session():
    if "transactions" not in st.session_state:
        st.session_state["transactions"] = pd.DataFrame(columns=["date", "category", "description", "amount"])
    if "budget" not in st.session_state:
        st.session_state["budget"] = 0.0
    if "recurring" not in st.session_state:
        # recurring: list of dicts {name, amount, day, category}
        st.session_state["recurring"] = []
    if "last_applied_month" not in st.session_state:
        st.session_state["last_applied_month"] = None

init_session()

# -------------------------
# Utility functions
# -------------------------
def parse_date(x):
    try:
        return pd.to_datetime(x)
    except Exception:
        return pd.NaT

def monthly_aggregate(df):
    if df.empty:
        return pd.DataFrame(columns=["month", "date", "amount"])
    tmp = df.copy()
    tmp["date"] = pd.to_datetime(tmp["date"])
    m = tmp.set_index("date")["amount"].resample("M").sum().reset_index()
    m["month"] = m["date"].dt.to_period("M").astype(str)
    return m[["month", "date", "amount"]]

def category_breakdown(df, month=None):
    tmp = df.copy()
    tmp["date"] = pd.to_datetime(tmp["date"])
    if month:
        tmp = tmp[tmp["date"].dt.to_period("M").astype(str) == month]
    if tmp.empty:
        return pd.Series(dtype=float)
    return tmp.groupby("category")["amount"].sum().sort_values(ascending=False)

def apply_recurring_to_month(target_month):
    """Add recurring expenses into transactions for the month if not already applied for that month.
    target_month: string like '2025-09' (Period M format)"""
    applied = 0
    if st.session_state["recurring"] == []:
        return applied
    # Check last applied month to avoid duplicate
    if st.session_state.get("last_applied_month") == target_month:
        return applied
    # Add recurring entries: choose date as month-first + day-1 (clamped)
    year, mon = map(int, target_month.split("-"))
    for r in st.session_state["recurring"]:
        try:
            day = int(r.get("day") or 1)
        except Exception:
            day = 1
        # Build date safely
        try:
            dt = pd.Timestamp(year=year, month=mon, day=min(day,28))  # keep safe day
        except Exception:
            dt = pd.Timestamp(year=year, month=mon, day=1)
        new = {
            "date": dt,
            "category": r.get("category", "Other") or "Other",
            "description": r.get("name", "Recurring"),
            "amount": float(r.get("amount") or 0.0)
        }
        st.session_state["transactions"] = pd.concat([st.session_state["transactions"], pd.DataFrame([new])], ignore_index=True)
        applied += 1
    st.session_state["last_applied_month"] = target_month
    return applied

def forecast_months(monthly_df, n_months=1):
    """Forecast next n_months. Use Prophet if available else simple linear regression."""
    if monthly_df.empty or len(monthly_df) < 2:
        mean_val = float(monthly_df["amount"].mean()) if (not monthly_df.empty) else 0.0
        last_date = monthly_df["date"].max() if not monthly_df.empty else pd.Timestamp.today()
        rows = []
        for i in range(1, n_months+1):
            nxt = (last_date + pd.DateOffset(months=i)).replace(day=1) + pd.offsets.MonthEnd(0)
            rows.append({"date": nxt, "amount": mean_val})
        return pd.DataFrame(rows)

    dfp = monthly_df.rename(columns={"date":"ds","amount":"y"})[["ds","y"]]
    if USE_PROPHET:
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m.fit(dfp)
        future = m.make_future_dataframe(periods=n_months, freq="M")
        forecast = m.predict(future)
        out = forecast.tail(n_months)[["ds","yhat"]].rename(columns={"ds":"date","yhat":"amount"})
        out["amount"] = out["amount"].apply(lambda x: float(max(0, x)))
        return out
    else:
        # Linear trend fallback
        X = np.arange(len(dfp)).reshape(-1,1)
        y = dfp["y"].values
        lr = LinearRegression().fit(X,y)
        future_idx = np.arange(len(dfp), len(dfp)+n_months).reshape(-1,1)
        preds = lr.predict(future_idx)
        last_date = dfp["ds"].max()
        rows = []
        for i, p in enumerate(preds, start=1):
            nxt = (last_date + pd.DateOffset(months=i)).replace(day=1) + pd.offsets.MonthEnd(0)
            rows.append({"date": nxt, "amount": float(max(0,p))})
        return pd.DataFrame(rows)

# -------------------------
# Sidebar: controls
# -------------------------
st.sidebar.header("Control Panel")

# Budget
st.sidebar.subheader("Monthly Budget")
budget_input = st.sidebar.number_input("Set monthly budget (PKR)", min_value=0.0, value=float(st.session_state["budget"] or 0.0), step=100.0, format="%.2f")
if st.sidebar.button("Save Budget"):
    st.session_state["budget"] = float(budget_input)
    st.sidebar.success(f"Budget set: {st.session_state['budget']:.2f} PKR")

# Recurring expenses manager
st.sidebar.subheader("Recurring Expenses")
with st.sidebar.form("recurring_form", clear_on_submit=True):
    r_name = st.text_input("Name (e.g., Rent, Internet)")
    r_amount = st.number_input("Amount", min_value=0.0, step=50.0, format="%.2f")
    r_day = st.number_input("Day of month (1-28)", min_value=1, max_value=28, value=1)
    r_category = st.selectbox("Category", ["Bills","Rent","Utilities","Subscription","Other"])
    r_submit = st.form_submit_button("Add Recurring")
    if r_submit:
        st.session_state["recurring"].append({"name": r_name or "Recurring", "amount": r_amount, "day": int(r_day), "category": r_category})
        st.sidebar.success("Recurring expense added")

st.sidebar.markdown("**Existing recurring items**")
if st.session_state["recurring"]:
    for i, r in enumerate(st.session_state["recurring"]):
        st.sidebar.write(f"{i+1}. {r['name']} — {r['amount']:.2f} PKR on day {r['day']} ({r['category']})")
    if st.sidebar.button("Clear recurring items"):
        st.session_state["recurring"] = []
        st.sidebar.info("Recurring items cleared")
else:
    st.sidebar.write("_No recurring items added_")

st.sidebar.markdown("---")
st.sidebar.subheader("Data")
uploaded = st.sidebar.file_uploader("Upload transactions CSV (date,category,description,amount)", type=["csv"])
if uploaded is not None:
    try:
        dfu = pd.read_csv(uploaded, parse_dates=["date"])
        # Basic validation
        if set(["date","category","description","amount"]).issubset(dfu.columns):
            st.session_state["transactions"] = pd.concat([st.session_state["transactions"], dfu[["date","category","description","amount"]]], ignore_index=True)
            st.sidebar.success("CSV uploaded and merged")
        else:
            st.sidebar.error("CSV must contain columns: date, category, description, amount")
    except Exception as e:
        st.sidebar.error("Failed to read CSV: " + str(e))

st.sidebar.markdown("---")
st.sidebar.write("Forecast engine:", "Prophet" if USE_PROPHET else "LinearRegression (fallback)")
st.sidebar.write("Tip: If Prophet is not installed and you want seasonal forecasts, add `prophet` to requirements.")
st.sidebar.markdown("---")
if st.sidebar.button("Apply recurring to this month"):
    current_month = str(pd.Timestamp.today().to_period("M"))

    added = apply_recurring_to_month(current_month)
    if added:
        st.sidebar.success(f"Applied {added} recurring items to {current_month}")
    else:
        st.sidebar.info("No recurring items to apply or already applied for this month")

# -------------------------
# Main area
# -------------------------
st.title("🏦 Expense Tracker — Budget Goals & Forecast")
st.markdown("Add expenses, set your monthly budget, and see forecasts & charts.")

# Main: add expense form
st.subheader("Add Expense")
with st.form("add_form", clear_on_submit=True):
    date_in = st.date_input("Date", value=pd.Timestamp.today().date())
    cat_in = st.selectbox("Category", ["Food","Transport","Bills","Entertainment","Groceries","Shopping","Healthcare","Other"])
    desc_in = st.text_input("Description (e.g., 'Uber - Airport')")
    amt_in = st.number_input("Amount (PKR)", min_value=0.0, step=50.0, format="%.2f")
    add_submit = st.form_submit_button("Add Expense")
    if add_submit and amt_in > 0:
        new = {"date": pd.to_datetime(date_in), "category": cat_in, "description": desc_in or "N/A", "amount": float(amt_in)}
        st.session_state["transactions"] = pd.concat([st.session_state["transactions"], pd.DataFrame([new])], ignore_index=True)
        st.success("Expense added")

# Show and manage transactions
st.subheader("Transactions (most recent first)")
if st.session_state["transactions"].empty:
    st.info("No transactions yet. Add using the form or upload a CSV.")
else:
    tx = st.session_state["transactions"].copy()
    tx["date"] = pd.to_datetime(tx["date"])
    tx = tx.sort_values("date", ascending=False).reset_index(drop=True)
    st.dataframe(tx)

    # Download CSV
    csv = tx.to_csv(index=False).encode("utf-8")
    st.download_button("Download transactions CSV", data=csv, file_name="transactions.csv", mime="text/csv")

# -------------------------
# Analytics, Budget check & Visuals
# -------------------------
st.markdown("---")
st.subheader("Monthly Summary & Budget Check")

# Aggregate monthly
monthly = monthly_aggregate(st.session_state["transactions"])
if monthly.empty:
    st.info("No monthly data yet. Add some transactions.")
else:
    # show months selector
    months = monthly["month"].tolist()
    selected_month = st.selectbox("Select month", options=months[::-1])
    # apply recurring for selected month for comparison if desired (doesn't double-apply)
    if st.button("Auto-apply recurring to selected month"):
        added = apply_recurring_to_month(selected_month)
        if added:
            st.success(f"Applied {added} recurring items to {selected_month}.")
            monthly = monthly_aggregate(st.session_state["transactions"])  # recompute
        else:
            st.info("No recurring items applied (maybe none or already applied).")

    # compute selected month total
    sel_rows = st.session_state["transactions"][pd.to_datetime(st.session_state["transactions"]["date"]).dt.to_period("M").astype(str) == selected_month]
    sel_total = float(sel_rows["amount"].sum()) if not sel_rows.empty else 0.0

    # show metrics
    colA, colB, colC = st.columns(3)
    colA.metric("Selected month", selected_month)
    colB.metric("Total spent", f"{sel_total:.2f} PKR")
    budget_val = float(st.session_state.get("budget", 0.0) or 0.0)
    colC.metric("Budget", f"{budget_val:.2f} PKR")

    # Budget alert
    if budget_val > 0:
        if sel_total > budget_val:
            st.error(f"⚠️ You exceeded the budget by {sel_total - budget_val:.2f} PKR in {selected_month}")
        else:
            st.success(f"✅ You are within budget. Remaining: {budget_val - sel_total:.2f} PKR")

    # Category breakdown pie
    cat_break = category_breakdown(st.session_state["transactions"], month=selected_month)
    if not cat_break.empty:
        figp, axp = plt.subplots(figsize=(6,4))
        axp.pie(cat_break.values, labels=cat_break.index, autopct='%1.1f%%', startangle=90)
        axp.set_title(f"Category split — {selected_month}")
        st.pyplot(figp)
    else:
        st.info("No category breakdown data for this month.")

    # Monthly trend + budget line
    show_months = monthly.tail(12)  # last 12 months
    figm, axm = plt.subplots(figsize=(10,4))
    axm.plot(show_months["date"], show_months["amount"], marker="o", label="Actual")
    if budget_val > 0:
        # draw budget as horizontal line at budget value
        axm.axhline(budget_val, color="red", linestyle="--", label="Monthly Budget")
    axm.set_title("Monthly Spending (last 12 months)")
    axm.set_xlabel("Month")
    axm.set_ylabel("Amount (PKR)")
    axm.legend()
    axm.grid(alpha=0.3)
    st.pyplot(figm)

# -------------------------
# Forecast
# -------------------------
st.markdown("---")
st.subheader("Forecast Future Months")

n_predict = st.slider("How many months to predict?", min_value=1, max_value=12, value=3)
if st.button("Compute Forecast"):
    monthly_now = monthly_aggregate(st.session_state["transactions"])
    forecast_df = forecast_months(monthly_now, n_predict)
    st.success("Forecast computed")
    st.dataframe(forecast_df.assign(amount=lambda d: d["amount"].map(lambda x: f"{x:.2f}")))
    # Plot with forecast
    figf, axf = plt.subplots(figsize=(10,4))
    if not monthly_now.empty:
        axf.plot(monthly_now["date"], monthly_now["amount"], marker="o", label="Actual")
    axf.plot(forecast_df["date"], forecast_df["amount"], marker="x", linestyle="--", color="orange", label="Forecast")
    if budget_val > 0:
        axf.axhline(budget_val, color="red", linestyle="--", label="Monthly Budget")
    axf.set_title("Actual + Forecast (monthly)")
    axf.set_xlabel("Month")
    axf.set_ylabel("Amount")
    axf.legend()
    st.pyplot(figf)

# -------------------------
# Tips and next steps
# -------------------------
st.markdown("---")
st.subheader("Tips")
st.markdown("""
- Enter descriptive descriptions (e.g., 'Uber - Airport') so categories and insights work better.
- Add at least 2–3 months of data for better forecasts.
- Use recurring feature for fixed monthly bills (rent, internet).
- To enable seasonal forecasts install Prophet (add `prophet` to requirements).
""")
