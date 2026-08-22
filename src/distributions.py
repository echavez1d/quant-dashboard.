import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
    # Create an evenly spaced array between min and max return for a smooth curve
    x_axis = np.linspace(x_min, x_max, 500)
    # Calculate Probability Density Function (PDF)
    y_axis = norm.pdf(x_axis, mean_ret, std_ret)

    fig.add_trace(go.Scatter(
        x=x_axis, y=y_axis,
        mode='lines',
        name='Normal Distribution',
        line=dict(color='firebrick', width=3)
    ))

    # 3. Add Vertical Lines for Standard Deviations (Risk Zones)
    colors = ['green', 'orange', 'red']
    for i in [1, 2, 3]:
        # Positive standard deviations
        fig.add_vline(x=mean_ret + i*std_ret, line_dash="dot", line_color=colors[i-1], opacity=0.8)
        # Negative standard deviations
        fig.add_vline(x=mean_ret - i*std_ret, line_dash="dot", line_color=colors[i-1], opacity=0.8)
        
        # Annotations for the negative (downside risk) standard deviations
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