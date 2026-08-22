import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# NEW IMPORT: gaussian_kde
from scipy.stats import norm, gaussian_kde

def plot_return_distribution(returns_series: pd.Series, ticker: str) -> go.Figure:
    """
    Creates an interactive histogram overlaid with both a Gaussian curve
    and a smoothed Kernel Density Estimation (KDE) line for direct comparison.
    """
    clean_returns = returns_series.dropna()
    mean_ret = clean_returns.mean()
    std_ret = clean_returns.std()

    fig = go.Figure()

    # 1. Empirical Histogram (Actual Data) - Now more transparent
    fig.add_trace(go.Histogram(
        x=clean_returns,
        histnorm='probability density',
        name='Actual Returns',
        marker_color='royalblue',
        opacity=0.4,  # Increased transparency to make curves clearer
        nbinsx=100
    ))

    # Calculate x-axis bounds (common for both curves)
    x_min, x_max = clean_returns.min(), clean_returns.max()
    x_axis = np.linspace(x_min, x_max, 500)

    # 2. Kernel Density Estimation (KDE) - The 'Messy Reality'
    # This creates a smoothed, continuous line of the empirical data.
    kde = gaussian_kde(clean_returns)
    kde_pdf = kde(x_axis)
    
    fig.add_trace(go.Scatter(
        x=x_axis, y=kde_pdf,
        mode='lines',
        name='KDE (Smoothed Reality)',
        line=dict(color='darkmagenta', width=4) # Distinct, thick line
    ))

    # 3. Theoretical Gaussian Curve - For Comparison
    y_axis_normal = norm.pdf(x_axis, mean_ret, std_ret)

    fig.add_trace(go.Scatter(
        x=x_axis, y=y_axis_normal,
        mode='lines',
        name='Normal (Gaussian) Assumption',
        # Dotted line to show it's just the ideal model
        line=dict(color='firebrick', width=3, dash='dash')
    ))

    # 4. Vertical Lines for Standard Deviations (unchanged)
    colors = ['green', 'orange', 'red']
    for i in [1, 2, 3]:
        fig.add_vline(x=mean_ret + i*std_ret, line_dash="dot", line_color=colors[i-1], opacity=0.8)
        fig.add_vline(x=mean_ret - i*std_ret, line_dash="dot", line_color=colors[i-1], opacity=0.8)
        
        fig.add_annotation(
            x=mean_ret - i*std_ret, y=0,
            text=f"-{i}σ", showarrow=False, yshift=10, xshift=-15, font=dict(color=colors[i-1])
        )

    fig.update_layout(
        title=f"{ticker} - 'Messy Reality' (KDE) vs. Gaussian Assumption",
        xaxis_title="Daily Return",
        yaxis_title="Probability Density",
        barmode='overlay',
        hovermode='x unified'
    )
    
    return fig

# (Rolling Moments function remains exactly the same below this line)
# def plot_rolling_moments(returns_series: pd.Series, ticker: str, window: int = 126) -> go.Figure:
#     ...