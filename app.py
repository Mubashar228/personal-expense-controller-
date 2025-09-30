
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# -------------------------------
# Initialize session state
# -------------------------------
if "transactions" not in st.session_state:
    st.session_state["transactions"] = pd.DataFrame(columns=["date", "category", "description", "amount"])

# -------------------------------
# Prophet Forecast Function
# -------------------------------
def forecast_months(monthly_df, n_months=1):
    if monthly_df.empty or len(monthly_df) < 2:
        mean_val = float(monthly_df["amount"].mean()) if not monthly_df.empty else 0.0
        last_date = monthly_df["date"].max() if not monthly_df.empty else pd.Timestamp.today()
        future = []
        for i in range(1, n_months + 1):
            next_dt = (last_date + pd.DateOffset(months=i)).replace(day=1) + pd.offsets.MonthEnd(0)
            future.append({"date": next_dt, "amount": mean_val})
        return pd.DataFrame(future)

    df_prophet = monthly_df.rename(columns={"date": "ds", "amount": "y"})[["ds", "y"]]
    model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=n_months, freq="M")
    forecast = model.predict(future)

    forecast_result = forecast.tail(n_months)[["ds", "yhat"]]
    forecast_result = forecast_result.rename(columns={"ds": "date", "yhat": "amount"})
    forecast_result["amount"] = forecast_result["amount"].apply(lambda x: float(max(0, x)))
    return forecast_result

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("💰 Advanced Personal Expense Tracker with Prophet")

# Expense Entry Section
st.header("➕ Add New Expense Manually")
with st.form("expense_form", clear_on_submit=True):
    date = st.date_input("Date")
    category = st.selectbox(
        "Category",
        ["Food", "Transport", "Bills", "Entertainment", "Groceries", "Shopping", "Healthcare", "Other"]
    )
    description = st.text_input("Description (e.g., 'Uber Ride - Airport', 'KFC Dinner with Friends')")
    amount = st.number_input("Amount (PKR)", min_value=0.0, step=50.0)
    submitted = st.form_submit_button("Add Expense")

    if submitted and amount > 0:
        new_row = {
            "date": pd.to_datetime(date),
            "category": category,
            "description": description if description else "N/A",
            "amount": amount,
        }
        st.session_state["transactions"] = pd.concat(
            [st.session_state["transactions"], pd.DataFrame([new_row])],
            ignore_index=True
        )
        st.success("✅ Expense added successfully!")

# CSV Upload Section
st.header("📂 Upload Historical Transactions (CSV)")
uploaded_file = st.file_uploader("Upload a CSV file with columns: date, category, description, amount", type=["csv"])
if uploaded_file is not None:
    uploaded_df = pd.read_csv(uploaded_file, parse_dates=["date"])
    st.session_state["transactions"] = pd.concat(
        [st.session_state["transactions"], uploaded_df],
        ignore_index=True
    )
    st.success("✅ CSV data uploaded successfully!")

# Show Current Data
st.header("📊 Your Expense Records")
if st.session_state["transactions"].empty:
    st.info("No expenses recorded yet. Add manually above or upload a CSV 👆")
else:
    df = st.session_state["transactions"].copy()
    df = df.sort_values("date")
    st.dataframe(df)

    # Monthly aggregation
    monthly_expenses = df.groupby(pd.Grouper(key="date", freq="M")).sum().reset_index()

    # Plot historical expenses (line + bar)
    st.subheader("📈 Monthly Expenses Overview")
    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 3))
        ax1.plot(monthly_expenses["date"], monthly_expenses["amount"], marker="o", label="Actual")
        ax1.set_xlabel("Month")
        ax1.set_ylabel("Total Expense (PKR)")
        ax1.set_title("Monthly Trend (Line Chart)")
        ax1.legend()
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        ax2.bar(monthly_expenses["date"].dt.strftime("%b-%Y"), monthly_expenses["amount"], color="skyblue")
        ax2.set_xlabel("Month")
        ax2.set_ylabel("Total Expense (PKR)")
        ax2.set_title("Monthly Totals (Bar Chart)")
        plt.xticks(rotation=45)
        st.pyplot(fig2)

    # Category distribution
    st.subheader("📊 Category-wise Expense Distribution")
    category_expenses = df.groupby("category")["amount"].sum()
    fig3, ax3 = plt.subplots()
    ax3.pie(category_expenses, labels=category_expenses.index, autopct="%1.1f%%", startangle=90)
    ax3.set_title("Expenses by Category")
    st.pyplot(fig3)

    # Forecast
    st.subheader("🔮 Forecast Future Expenses")
    n_months = st.slider("Select number of months to forecast", 1, 12, 3)
    forecast_df = forecast_months(monthly_expenses, n_months=n_months)

    st.write("### Forecasted Expenses")
    st.dataframe(forecast_df)

    # Plot with forecast
    st.subheader("📉 Actual vs Forecast")
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    ax4.plot(monthly_expenses["date"], monthly_expenses["amount"], marker="o", label="Actual")
    ax4.plot(forecast_df["date"], forecast_df["amount"], marker="x", linestyle="--", color="red", label="Forecast")
    ax4.set_xlabel("Month")
    ax4.set_ylabel("Expense (PKR)")
    ax4.legend()
    st.pyplot(fig4)
