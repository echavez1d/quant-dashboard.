import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

def plot_return_distribution(returns_series: pd.Series, ticker: str) -> go.Figure:
    """
    Creates an interactive histogram of empirical returns overlaid 
    with a theoretical Gaussian (Normal) distribution.
    """
    clean_returns = returns_series.dropna()
    mean_ret = clean_returns.mean()
    std_ret = clean_returns.std()

    fig = go.Figure()

    # 1. Empirical Histogram (Actual Data)
    fig.add_trace(go.Histogram(
        x=clean_returns,
        histnorm='probability density',
        name='Actual Returns',
        marker_color='royalblue',
        opacity=0.7,
        nbinsx=100
    ))

    # 2. Theoretical Gaussian Curve
    x_min, x_max = clean_returns.min(), clean_returns.max()
    x_axis = np.linspace(x_min, x_max, 500)
    y_axis = norm.pdf(x_axis, mean_ret, std_ret)

    fig.add_trace(go.Scatter(
        x=x_axis, y=y_axis,
        mode='lines',
        name='Normal Distribution',
        line=dict(color='firebrick', width=3)
    ))

    # 3. Add Vertical Lines for Standard Deviations
    colors = ['green', 'orange', 'red']
    for i in [1, 2, 3]:
        fig.add_vline(x=mean_ret + i*std_ret, line_dash="dot", line_color=colors[i-1], opacity=0.8)
        fig.add_vline(x=mean_ret - i*std_ret, line_dash="dot", line_color=colors[i-1], opacity=0.8)
        
        fig.add_annotation(
            x=mean_ret - i*std_ret, y=0,
            text=f"-{i}σ", showarrow=False, yshift=10, xshift=-15, font=dict(color=colors[i-1])
        )

    fig.update_layout(
        title=f"{ticker} - Return Distribution vs. Normal (Gaussian) Assumption",
        xaxis_title="Daily Return",
        yaxis_title="Probability Density",
        barmode='overlay',
        hovermode='x unified'
    )
    
    return fig

def plot_rolling_moments(returns_series: pd.Series, ticker: str, window: int = 126) -> go.Figure:
    """
    Plots 6-month (126 trading days) rolling skewness and kurtosis to visualize 
    how tail risk and asymmetry dynamically change over time.
    """
    clean_returns = returns_series.dropna()
    
    # Calculate rolling metrics (126 days = ~6 months of trading)
    rolling_skew = clean_returns.rolling(window=window).skew()
    rolling_kurt = clean_returns.rolling(window=window).kurt()
    
    # Create subplots sharing the X-axis (Timeline)
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
        subplot_titles=(f"Rolling Skewness (Asymmetry)", f"Rolling Excess Kurtosis (Tail Risk)")
    )
    
    # Trace 1: Skewness
    fig.add_trace(go.Scatter(
        x=rolling_skew.index, y=rolling_skew, 
        mode='lines', name='Skewness', line=dict(color='purple', width=2)
    ), row=1, col=1)
    
    # Zero line for Skewness (perfect symmetry)
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=1, col=1)
    
    # Trace 2: Kurtosis
    fig.add_trace(go.Scatter(
        x=rolling_kurt.index, y=rolling_kurt, 
        mode='lines', name='Kurtosis', line=dict(color='orange', width=2)
    ), row=2, col=1)
    
    # Zero line for Excess Kurtosis (Normal Distribution baseline)
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=2, col=1)
    
    fig.update_layout(
        height=600, 
        title_text=f"{ticker} - {window}-Day Rolling Distribution Moments", 
        showlegend=False,
        hovermode='x unified'
    )
    
    return fig