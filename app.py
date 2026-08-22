import streamlit as st
import plotly.express as px
import pandas as pd
from src.data_loader import fetch_market_data
from src.returns import calculate_returns, resample_returns
from src.statistics import compute_summary_statistics, calculate_max_drawdown, run_normality_tests
from src.distributions import plot_return_distribution, plot_rolling_moments
from src.risk import calculate_historical_var, calculate_historical_cvar

st.set_page_config(page_title="Quant Dashboard", page_icon="📈", layout="wide")

st.title("📈 Quantitative Financial Analysis Platform")
st.caption("Historical statistical behavior, distribution analysis, and risk metrics.")

st.sidebar.header("Asset Selection")
ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL").upper()
time_period = st.sidebar.selectbox("Lookback Period", ["1M", "3M", "6M", "1Y", "3Y", "5Y", "10Y", "MAX"], index=3)

st.divider()

# --- STAGE 10: UI POLISH (CSV EXPORT) ---
@st.cache_data
def convert_df(df):
    # Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=True).encode('utf-8')
# ----------------------------------------

if ticker:
    with st.spinner(f"Loading market data for {ticker}..."):
        raw_data = fetch_market_data(ticker, time_period)

    if raw_data.empty:
        st.error(f"Could not load data for symbol '{ticker}'.")
    else:
        data = calculate_returns(raw_data, price_col="Close")
        stats_dict = compute_summary_statistics(data["Daily_Return"])
        max_dd, dd_series = calculate_max_drawdown(data["Close"])
        data["Drawdown"] = dd_series

        var_95 = calculate_historical_var(data["Daily_Return"], confidence_level=0.95)
        cvar_95 = calculate_historical_cvar(data["Daily_Return"], confidence_level=0.95)

        # UI Polish: Download Data Button
        csv_data = convert_df(data)
        st.download_button(
            label=f"📥 Download {ticker} Cleaned Data (CSV)",
            data=csv_data,
            file_name=f"{ticker}_quant_data.csv",
            mime="text/csv",
        )
        st.write("") # small spacing

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Ann. Return", f"{stats_dict['Annualized Return']*100:+.2f}%")
        col2.metric("Ann. Volatility", f"{stats_dict['Annualized Volatility']*100:.2f}%")
        col3.metric("Sharpe Ratio", f"{stats_dict['Sharpe Ratio']:.2f}")
        col4.metric("Daily VaR (95%)", f"{var_95*100:.2f}%")
        col5.metric("Max Drawdown", f"{max_dd*100:.2f}%")

        st.subheader("Price, Returns & Risk Analysis")

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Price History", "Daily Returns", "Drawdowns", "Statistical Moments", "Distribution Analysis", "Tail Risk"
        ])

        with tab1:
            fig_price = px.line(data, x=data.index, y="Close", title=f"{ticker} Price History")
            st.plotly_chart(fig_price, use_container_width=True)

        with tab2:
            fig_daily = px.bar(data, x=data.index, y="Daily_Return", title=f"{ticker} Daily Returns")
            st.plotly_chart(fig_daily, use_container_width=True)

        with tab3:
            fig_dd = px.area(data, x=data.index, y="Drawdown", title=f"{ticker} Drawdown Profile")
            fig_dd.update_traces(fillcolor="rgba(255, 0, 0, 0.3)", line_color="red")
            st.plotly_chart(fig_dd, use_container_width=True)

        with tab4:
            stats_df = pd.DataFrame(list(stats_dict.items()), columns=["Metric", "Value"])
            stats_df["Formatted Value"] = stats_df.apply(
                lambda row: f"{row['Value']*100:.4f}%" if "Daily" in row["Metric"] or "Return" in row["Metric"] or "Volatility" in row["Metric"]
                else f"{row['Value']:.4f}", axis=1
            )
            st.table(stats_df[["Metric", "Formatted Value"]])

        with tab5:
            # 1. Static Histogram
            fig_dist = plot_return_distribution(data["Daily_Return"], ticker)
            st.plotly_chart(fig_dist, use_container_width=True)
            
            st.markdown("### Formal Normality Tests")
            normality_df = run_normality_tests(data["Daily_Return"])
            st.dataframe(normality_df, use_container_width=True, hide_index=True)
            
            st.divider()
            
            # 2. Dynamic Rolling Moments Visualization
            st.markdown("### Dynamic Evolution of Tail Risk")
            st.markdown("""
            Standard statistics assume risk is constant. By plotting a 6-month rolling window, we can visualize regime changes:
            *   **Skewness (Purple):** When this dips deeply below the zero line, downside risk dominates.
            *   **Kurtosis (Orange):** When this spikes above zero, the market is experiencing extreme "fat-tailed" volatility events (market shocks).
            """)
            
            fig_moments = plot_rolling_moments(data["Daily_Return"], ticker)
            st.plotly_chart(fig_moments, use_container_width=True)

        with tab6:
            st.markdown(f"### Historical Tail Risk for {ticker}")
            st.markdown(f"**95% Value at Risk (VaR):** `{var_95*100:.2f}%`")
            st.markdown(f"**95% Expected Shortfall (CVaR):** `{cvar_95*100:.2f}%`")

        # --- NEW SECTION: MULTIPLE TIME HORIZONS ---
        st.subheader("Historical Returns by Time Horizon")
        st.caption("Actual point-to-point historical returns calculated from settled market prices.")
        
        rt1, rt2, rt3, rt4 = st.tabs(["Weekly", "Monthly", "Quarterly", "Annually"])
        
        with rt1:
            weekly_df = resample_returns(raw_data, freq="W-FRI", price_col="Close") * 100
            st.dataframe(weekly_df.style.format("{:+.2f}%"), use_container_width=True)
            
        with rt2:
            monthly_df = resample_returns(raw_data, freq="ME", price_col="Close") * 100
            st.dataframe(monthly_df.style.format("{:+.2f}%"), use_container_width=True)
            
        with rt3:
            quarterly_df = resample_returns(raw_data, freq="QE", price_col="Close") * 100
            st.dataframe(quarterly_df.style.format("{:+.2f}%"), use_container_width=True)
            
        with rt4:
            annual_df = resample_returns(raw_data, freq="YE", price_col="Close") * 100
            st.dataframe(annual_df.style.format("{:+.2f}%"), use_container_width=True)