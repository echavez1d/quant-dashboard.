import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from src.data_loader import fetch_market_data
from src.returns import calculate_returns, resample_returns
from src.statistics import compute_summary_statistics, calculate_max_drawdown, run_normality_tests
from src.distributions import plot_return_distribution, plot_rolling_moments
from src.risk import calculate_historical_var, calculate_historical_cvar
from src.levels import calculate_price_zones

st.set_page_config(page_title="Quant Dashboard", page_icon="📈", layout="wide")

st.title("📈 Quantitative Financial Analysis Platform")
st.caption("Historical statistical behavior, distribution analysis, and risk metrics.")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Asset Selection")
ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL").upper()
time_period = st.sidebar.selectbox("Lookback Period", ["1M", "3M", "6M", "1Y", "3Y", "5Y", "10Y", "MAX"], index=3)

st.sidebar.divider()

st.sidebar.header("S&R Engine Parameters")
swing_length = st.sidebar.slider(
    "Swing Pivot Length (Days)", 
    min_value=2, max_value=20, value=5, 
    help="Number of days before and after a high/low to define a valid structural turning point."
)
cluster_tol_pct = st.sidebar.slider(
    "Zone Cluster Tolerance (%)", 
    min_value=0.1, max_value=3.0, value=0.5, step=0.1, 
    help="Max percentage difference between price points to group them into a single support/resistance zone."
)
# Convert percentage back to decimal for the engine math
cluster_tolerance = cluster_tol_pct / 100.0

st.divider()

# --- STAGE 10: UI POLISH (CSV EXPORT) ---
@st.cache_data
def convert_df(df):
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

        # --- CALCULATE S&R ZONES ONCE FOR ALL TABS (USING SIDEBAR DYNAMIC INPUTS) ---
        zones_df = calculate_price_zones(data, left_bars=swing_length, right_bars=swing_length, cluster_tolerance=cluster_tolerance)

        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "Price History", "Daily Returns", "Drawdowns", "Statistical Moments", "Distribution Analysis", "Tail Risk", "S&R Zones"
        ])

        with tab1:
            # 1. Base Price Chart (Now Candlesticks!)
            fig_price = go.Figure(data=[go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Candlestick'
            )])
            
            fig_price.update_layout(
                title=f"{ticker} Price History with Major S&R Zones",
                xaxis_rangeslider_visible=False,
                yaxis_title="Price ($)"
            )
            
            # 2. Color-coded Significance Palettes
            zone_styles = [
                {"fill": "rgba(245, 158, 11, 0.22)", "border": "rgba(245, 158, 11, 0.85)", "label": "🥇 Tier 1 Strongest Zone"},
                {"fill": "rgba(59, 130, 246, 0.20)", "border": "rgba(59, 130, 246, 0.85)", "label": "🥈 Tier 2 Major Zone"},
                {"fill": "rgba(168, 85, 247, 0.18)", "border": "rgba(168, 85, 247, 0.85)", "label": "🥉 Tier 3 Key Zone"}
            ]

            # 3. Add shaded horizontal bands with price bounds and significance colors
            if not zones_df.empty:
                top_zones = zones_df.head(3)
                for idx, (_, zone) in enumerate(top_zones.iterrows()):
                    style = zone_styles[idx] if idx < len(zone_styles) else zone_styles[-1]
                    
                    bottom_str = f"${zone['Zone_Bottom']:.2f}"
                    top_str = f"${zone['Zone_Top']:.2f}"
                    center_str = f"${zone['Zone_Center']:.2f}"
                    reversals = int(zone['Reversal_Touches'])
                    sig_score = zone['Significance']

                    # Shaded Zone with Price Bounds
                    fig_price.add_hrect(
                        y0=zone["Zone_Bottom"], 
                        y1=zone["Zone_Top"], 
                        line_width=1.5,
                        line_color=style["border"],
                        fillcolor=style["fill"], 
                        layer="below",
                        annotation_text=f"{style['label']} | Range: {bottom_str} - {top_str} (Mid: {center_str}) | {reversals} Reversals ({sig_score:.1f}% Sig)",
                        annotation_position="top left",
                        annotation_font_color=style["border"],
                        annotation_font_size=11
                    )

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
            fig_dist = plot_return_distribution(data["Daily_Return"], ticker)
            st.plotly_chart(fig_dist, use_container_width=True)
            
            st.markdown("### Formal Normality Tests")
            normality_df = run_normality_tests(data["Daily_Return"])
            st.dataframe(normality_df, use_container_width=True, hide_index=True)
            
            st.divider()
            
            st.markdown("### Dynamic Evolution of Tail Risk")
            st.markdown("""
            Standard statistics assume risk is constant. By plotting a 6-month rolling window, we can visualize regime changes:
            *   **Skewness (Purple):** When this dips deeply below the zero line, downside risk dominates.
            *   **Kurtosis (Orange):** When this spikes above zero, the market is experiencing extreme "fat-tailed" volatility events.
            """)
            
            fig_moments = plot_rolling_moments(data["Daily_Return"], ticker)
            st.plotly_chart(fig_moments, use_container_width=True)

        with tab6:
            st.markdown(f"### Historical Tail Risk for {ticker}")
            st.markdown(f"**95% Value at Risk (VaR):** `{var_95*100:.2f}%`")
            st.markdown(f"**95% Expected Shortfall (CVaR):** `{cvar_95*100:.2f}%`")


        with tab7:
            st.markdown("### Advanced Heuristic S&R Engine")
            st.markdown("""
            This engine detects **objective price levels** using a multi-factor quantitative model:
            * **Score:** The composite Zone Strength Score (0-100).
            * **Concentration (C):** Recency and volume-weighted turnaround density.
            * **Rejection (R):** Win-rate of the zone holding when tested.
            * **Reaction (S):** ATR-normalized strength of the bounce.
            * **Confidence (F):** Bayesian sample-size adjustment.
            """)
            
            col_search, col_stats = st.columns([1, 2])
            
            with col_search:
                st.info("🔍 **Level Search Engine**")
                current_price = data["Close"].iloc[-1]
                target_price = st.number_input("Enter a Price Level to test ($):", value=float(current_price), step=1.0)
                
                match = zones_df[(target_price >= zones_df["Zone_Bottom"]) & (target_price <= zones_df["Zone_Top"])]
                
                if not match.empty:
                    touches = match["Reversal_Touches"].values[0]
                    score = match["Score"].values[0]
                    z_bot = match['Zone_Bottom'].values[0]
                    z_top = match['Zone_Top'].values[0]
                    st.success(f"**Valid Zone Found!**\\nThe price **${target_price:.2f}** falls inside a historic zone: **${z_bot:.2f} –${z_top:.2f}**.\\n\\n"
                               f"• **Pivot Reversals:** {touches}\\n"
                               f"• **Strength Score:** {score:.2f}/100")
                else:
                    st.warning("No historical swing pivots recorded at this exact price level.")

            with col_stats:
                st.markdown("**Top Historical Reversal Zones (Ranked by Strength)**")
                
                # Safety check: Only try to display the table if zones actually exist
                if zones_df.empty:
                    st.info("No historical S&R zones detected for this timeframe and sensitivity. Try adjusting your sidebar parameters.")
                else:
                    top_zones_table = zones_df.head(10).copy()
                    top_zones_table["Price Band"] = top_zones_table.apply(lambda x: f"${x['Zone_Bottom']:,.2f} – ${x['Zone_Top']:,.2f}", axis=1)
                    
                    # Select and rename columns for a clean UI
                    display_df = top_zones_table[[
                        "Price Band", "Reversal_Touches", "Score", "Concentration", "Rejection_Rate", "Reaction_Str", "Confidence"
                    ]].rename(columns={
                        "Reversal_Touches": "Pivots",
                        "Score": "Total Score",
                        "Reaction_Str": "Reaction"
                    })
                    
                    st.dataframe(
                        display_df.style.format({
                            "Total Score": "{:.1f}",
                            "Concentration": "{:.2f}",
                            "Rejection_Rate": "{:.2f}",
                            "Reaction": "{:.2f}",
                            "Confidence": "{:.2f}"
                        }).background_gradient(subset=["Total Score"], cmap="viridis"),
                        use_container_width=True, hide_index=True
                    )


        # --- MULTIPLE TIME HORIZONS ---
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