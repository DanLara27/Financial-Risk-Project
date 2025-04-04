import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import kurtosis, skew, shapiro, norm, t
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="Financial Risk Analysis - Nike (NKE)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #0066cc;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #004d99;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .section-divider {
        margin-top: 2rem;
        margin-bottom: 2rem;
        border-top: 1px solid #e0e0e0;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 0.15rem 0.3rem rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-left: 5px solid #0066cc;
        border-radius: 0.3rem;
        margin-bottom: 1rem;
    }
    .conclusion {
        background-color: #f0f7ff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #b8daff;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Title and Introduction
st.markdown("<h1 class='main-header'>Financial Risk Analysis: Nike (NKE)</h1>", unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
This application analyzes the financial risk of Nike (NKE) stock using various risk metrics including Value at Risk (VaR) 
and Expected Shortfall (ES). The analysis includes parametric, historical, and Monte Carlo approaches with different confidence levels.
</div>
""", unsafe_allow_html=True)

# Helper Functions
@st.cache_data
def get_stock_data(ticker="NKE", start_date="2010-01-01"):
    """Download historical stock data from Yahoo Finance"""
    try:
        df = yf.download(ticker, start=start_date)["Close"]
        return df
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        return None

@st.cache_data
def calculate_returns(prices):
    """Calculate daily returns from price data"""
    returns = prices.pct_change().dropna()
    return returns

# Main sidebar for controls
with st.sidebar:
    st.image("https://1000logos.net/wp-content/uploads/2021/11/Nike-Logo.png", width=200)
    st.header("Nike (NKE) Analysis")
    
    st.markdown("### Analysis Parameters")
    window_size = st.slider("Rolling Window Size (Days)", min_value=50, max_value=500, value=252, step=10)
    
    st.markdown("### Confidence Levels")
    st.info("Preset levels: 95%, 97.5%, 99%")
    
    st.markdown("### About This Project")
    st.write("Financial risk metrics calculation using rolling windows methodology.")
    st.write("Developed for Quantitative Methods in Finance course.")

# Load data
with st.spinner("Loading Nike stock data..."):
    prices = get_stock_data("NKE")
    returns = calculate_returns(prices)

# Main analysis
if returns is not None:
    # Basic Statistics Section
    st.markdown("<h2 class='sub-header'>Basic Statistics</h2>", unsafe_allow_html=True)
    
    mean_return = returns.mean()
    std_dev = returns.std()
    kurt = kurtosis(returns)
    skewness = skew(returns)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean Daily Return", f"{mean_return:.4%}")
    with col2:
        st.metric("Standard Deviation", f"{std_dev:.4%}")
    with col3:
        st.metric("Excess Kurtosis", f"{kurt:.4f}")
    with col4:
        st.metric("Skewness", f"{skewness:.4f}")

    # Visualization of Returns
    st.markdown("<h2 class='sub-header'>Returns Analysis</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Daily returns chart
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(returns.index, returns, color='#0066cc', alpha=0.7)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax.set_title("Nike Daily Returns (2010-Present)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Daily Return")
        ax.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig)
    
    with col2:
        # Return distribution
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Histogram
        ax.hist(returns, bins=40, alpha=0.7, color='#0066cc', edgecolor='black')
        ax.axvline(mean_return, color='red', linestyle='dashed', linewidth=2, label=f"Mean: {mean_return:.4%}")
        
        # Add normal distribution curve for comparison
        x = np.linspace(min(returns), max(returns), 100)
        y = norm.pdf(x, mean_return, std_dev)
        y_scaled = y * (len(returns) * (max(returns) - min(returns)) / 40) # Scale to match histogram
        ax.plot(x, y_scaled, 'r-', alpha=0.5, label='Normal Distribution')
        
        ax.legend()
        ax.set_title("Distribution of Returns")
        ax.set_xlabel("Daily Return")
        ax.set_ylabel("Frequency")
        ax.grid(True, linestyle='--', alpha=0.3)
        st.pyplot(fig)

    # Normality Tests
    st.markdown("<h2 class='sub-header'>Normality Tests</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Shapiro-Wilk Test
        stat, p_value = shapiro(returns)
        st.markdown(f"**Shapiro-Wilk Test Results:**")
        st.write(f"Test Statistic: {stat:.4f}")
        st.write(f"P-value: {p_value:.8f}")
        
        if p_value < 0.05:
            st.error("The distribution is NOT normal (Reject H₀)")
        else:
            st.success("The distribution appears to be normal (Fail to reject H₀)")
    
    with col2:
        # Q-Q Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        stats.probplot(returns, dist="norm", plot=ax)
        ax.set_title("Q-Q Plot of Returns")
        ax.grid(True, linestyle='--', alpha=0.3)
        st.pyplot(fig)

    # Risk Metrics Section
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>Risk Metrics Analysis</h2>", unsafe_allow_html=True)

    # Function to calculate risk metrics
    def calculate_risk_metrics(returns, confidence_levels=[0.95, 0.975, 0.99]):
        """Calculate VaR and ES using different methods"""
        results = {}
        
        mean = returns.mean()
        std = returns.std()
        
        # Degrees of freedom for t-distribution (estimate)
        df = 6  # typical value for financial returns
        
        for conf in confidence_levels:
            alpha = 1 - conf
            
            # Parametric Normal
            var_norm = norm.ppf(alpha, mean, std)
            
            # Parametric t-student
            var_t = mean + std * t.ppf(alpha, df)
            
            # Historical
            var_hist = returns.quantile(alpha)
            
            # Monte Carlo
            n_sims = 100000
            sim_returns = np.random.normal(mean, std, n_sims)
            var_mc = np.percentile(sim_returns, alpha * 100)
            
            # Expected Shortfall (CVAR)
            es_norm = mean - std * norm.pdf(norm.ppf(alpha)) / alpha
            es_t = returns[returns <= var_hist].mean()
            es_hist = returns[returns <= var_hist].mean()
            
            results[conf] = {
                'VaR_Normal': var_norm,
                'VaR_t': var_t,
                'VaR_Hist': var_hist,
                'VaR_MC': var_mc,
                'ES_Normal': es_norm,
                'ES_Hist': es_hist
            }
            
        return results

    risk_metrics = calculate_risk_metrics(returns)

    # Create tables for each confidence level
    tabs = st.tabs(["95% Confidence", "97.5% Confidence", "99% Confidence"])
    
    for i, (conf, tab) in enumerate(zip([0.95, 0.975, 0.99], tabs)):
        with tab:
            metrics = risk_metrics[conf]
            
            # Table of risk metrics
            st.markdown(f"### Risk Metrics at {conf*100}% Confidence Level")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("VaR (Normal)", f"{metrics['VaR_Normal']:.4%}")
            with col2:
                st.metric("VaR (Student-t)", f"{metrics['VaR_t']:.4%}")
            with col3:
                st.metric("VaR (Historical)", f"{metrics['VaR_Hist']:.4%}")
            with col4:
                st.metric("VaR (Monte Carlo)", f"{metrics['VaR_MC']:.4%}")
                
            st.markdown("### Expected Shortfall (CVaR)")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("ES (Normal)", f"{metrics['ES_Normal']:.4%}")
            with col2:
                st.metric("ES (Historical)", f"{metrics['ES_Hist']:.4%}")
            
            # Distribution visualization with risk metrics
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Histogram
            n, bins, patches = ax.hist(returns, bins=50, color='blue', alpha=0.7, label='Returns')
            
            # Color the tail bins
            for bin_left, bin_right, patch in zip(bins, bins[1:], patches):
                if bin_left < metrics['VaR_Hist']:
                    patch.set_facecolor('red')
                    
            # Mark risk metrics
            ax.axvline(x=metrics['VaR_Normal'], color='orange', linestyle='--', 
                      label=f"VaR Normal {conf*100}%: {metrics['VaR_Normal']:.4%}")
            ax.axvline(x=metrics['VaR_Hist'], color='green', linestyle='--', 
                      label=f"VaR Historical {conf*100}%: {metrics['VaR_Hist']:.4%}")
            ax.axvline(x=metrics['VaR_MC'], color='purple', linestyle='--', 
                      label=f"VaR Monte Carlo {conf*100}%: {metrics['VaR_MC']:.4%}")
            ax.axvline(x=metrics['ES_Hist'], color='black', linestyle='-.', 
                      label=f"ES {conf*100}%: {metrics['ES_Hist']:.4%}")
            
            # Configuration
            ax.set_title(f"Return Distribution with VaR and ES at {conf*100}% Confidence")
            ax.set_xlabel("Daily Return")
            ax.set_ylabel("Frequency")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.3)
            
            st.pyplot(fig)

    # Rolling Window Analysis
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>Rolling Window Risk Analysis</h2>", unsafe_allow_html=True)
    
    st.write(f"Using a rolling window of {window_size} trading days")

    # Helper Functions for Rolling Analysis
    def rolling_var_param(returns, alpha=0.05, window=252):
        """Calculate parametric VaR with rolling window"""
        z_alpha = norm.ppf(alpha)  # Negative value for losses
        
        rolling_mean = returns.rolling(window=window).mean()
        rolling_std = returns.rolling(window=window).std()
        
        var = rolling_mean + z_alpha * rolling_std  # Negative value (expected loss)
        return var

    def rolling_es_param(returns, alpha=0.05, window=252):
        """Calculate parametric ES with rolling window"""
        z_alpha = norm.ppf(alpha)
        
        rolling_mean = returns.rolling(window=window).mean()
        rolling_std = returns.rolling(window=window).std()
        
        es = rolling_mean - (norm.pdf(z_alpha) / alpha) * rolling_std
        return es

    def rolling_var_hist(returns, alpha=0.05, window=252):
        """Calculate historical VaR with rolling window"""
        return returns.rolling(window=window).quantile(alpha)

    def rolling_es_hist(returns, alpha=0.05, window=252):
        """Calculate historical ES with rolling window"""
        var_series = rolling_var_hist(returns, alpha, window)
        
        # For each point, calculate ES based on values below VaR
        es_values = []
        
        for i in range(len(returns)):
            if i < window - 1:
                es_values.append(np.nan)
            else:
                window_returns = returns.iloc[i-window+1:i+1]
                var_value = var_series.iloc[i]
                
                extreme_returns = window_returns[window_returns <= var_value]
                if not extreme_returns.empty:
                    es_values.append(extreme_returns.mean())
                else:
                    es_values.append(np.nan)
        
        return pd.Series(es_values, index=returns.index)

    # Calculate Rolling Risk Metrics
    with st.spinner("Calculating rolling risk metrics..."):
        # 95% confidence (alpha = 0.05)
        var_param_95 = rolling_var_param(returns, alpha=0.05, window=window_size)
        es_param_95 = rolling_es_param(returns, alpha=0.05, window=window_size)
        var_hist_95 = rolling_var_hist(returns, alpha=0.05, window=window_size)
        es_hist_95 = rolling_es_hist(returns, alpha=0.05, window=window_size)
        
        # 99% confidence (alpha = 0.01)
        var_param_99 = rolling_var_param(returns, alpha=0.01, window=window_size)
        es_param_99 = rolling_es_param(returns, alpha=0.01, window=window_size)
        var_hist_99 = rolling_var_hist(returns, alpha=0.01, window=window_size)
        es_hist_99 = rolling_es_hist(returns, alpha=0.01, window=window_size)
        
        # Shift by 1 for predictions
        var_param_95_shifted = var_param_95.shift(1)
        es_param_95_shifted = es_param_95.shift(1)
        var_hist_95_shifted = var_hist_95.shift(1)
        es_hist_95_shifted = es_hist_95.shift(1)
        
        var_param_99_shifted = var_param_99.shift(1)
        es_param_99_shifted = es_param_99.shift(1)
        var_hist_99_shifted = var_hist_99.shift(1)
        es_hist_99_shifted = es_hist_99.shift(1)

    # Plotting the Rolling VaR and ES
    # First, let the user select which metrics to display
    col1, col2 = st.columns(2)
    with col1:
        show_var_95 = st.checkbox("Show VaR 95%", value=True)
        show_var_99 = st.checkbox("Show VaR 99%", value=True)
    with col2:
        show_es_95 = st.checkbox("Show ES 95%", value=True)
        show_es_99 = st.checkbox("Show ES 99%", value=False)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Always show returns
    ax.plot(returns.index, returns, label="Daily Returns", color="grey", alpha=0.4)
    
    # Conditionally add risk metrics
    if show_var_95:
        ax.plot(var_param_95_shifted.index, var_param_95_shifted, 
                label="VaR Parametric 95%", color="blue", linestyle="--")
        ax.plot(var_hist_95_shifted.index, var_hist_95_shifted, 
                label="VaR Historical 95%", color="green", linestyle="--")
    
    if show_var_99:
        ax.plot(var_param_99_shifted.index, var_param_99_shifted, 
                label="VaR Parametric 99%", color="purple", linestyle="--")
        ax.plot(var_hist_99_shifted.index, var_hist_99_shifted, 
                label="VaR Historical 99%", color="orange", linestyle="--")
    
    if show_es_95:
        ax.plot(es_param_95_shifted.index, es_param_95_shifted, 
                label="ES Parametric 95%", color="cyan", linestyle="-.")
        ax.plot(es_hist_95_shifted.index, es_hist_95_shifted, 
                label="ES Historical 95%", color="pink", linestyle="-.")
    
    if show_es_99:
        ax.plot(es_param_99_shifted.index, es_param_99_shifted, 
                label="ES Parametric 99%", color="brown", linestyle="-.")
        ax.plot(es_hist_99_shifted.index, es_hist_99_shifted, 
                label="ES Historical 99%", color="yellow", linestyle="-.")
    
    # Plot styling
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.legend(loc="lower left")
    ax.set_title("Nike (NKE) Daily Returns with Rolling VaR and ES")
    ax.set_xlabel("Date")
    ax.set_ylabel("Return")
    ax.grid(True, linestyle="--", alpha=0.5)
    
    # Show the plot
    st.pyplot(fig)

    # Calculate Efficiency (Backtest)
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>Risk Model Efficiency Analysis</h2>", unsafe_allow_html=True)
    
    # Function to calculate violations
    def calculate_violations(returns, risk_metrics):
        """Calculate number and percentage of violations"""
        violations = (returns < risk_metrics.shift(1)).sum()
        total_days = (~risk_metrics.shift(1).isna()).sum()
        violation_pct = (violations / total_days) * 100 if total_days > 0 else 0
        return violations, violation_pct

    # Calculate violations for each risk metric
    var_param_95_viol, var_param_95_pct = calculate_violations(returns, var_param_95)
    es_param_95_viol, es_param_95_pct = calculate_violations(returns, es_param_95)
    var_hist_95_viol, var_hist_95_pct = calculate_violations(returns, var_hist_95)
    es_hist_95_viol, es_hist_95_pct = calculate_violations(returns, es_hist_95)
    
    var_param_99_viol, var_param_99_pct = calculate_violations(returns, var_param_99)
    es_param_99_viol, es_param_99_pct = calculate_violations(returns, es_param_99)
    var_hist_99_viol, var_hist_99_pct = calculate_violations(returns, var_hist_99)
    es_hist_99_viol, es_hist_99_pct = calculate_violations(returns, es_hist_99)
    
    # Create a DataFrame for the results
    results_data = {
        "Risk Metric": [
            "VaR Parametric 95%", "ES Parametric 95%", "VaR Historical 95%", "ES Historical 95%",
            "VaR Parametric 99%", "ES Parametric 99%", "VaR Historical 99%", "ES Historical 99%"
        ],
        "Violations": [
            var_param_95_viol, es_param_95_viol, var_hist_95_viol, es_hist_95_viol,
            var_param_99_viol, es_param_99_viol, var_hist_99_viol, es_hist_99_viol
        ],
        "Violation Percentage (%)": [
            f"{var_param_95_pct:.2f}%", f"{es_param_95_pct:.2f}%", 
            f"{var_hist_95_pct:.2f}%", f"{es_hist_95_pct:.2f}%",
            f"{var_param_99_pct:.2f}%", f"{es_param_99_pct:.2f}%", 
            f"{var_hist_99_pct:.2f}%", f"{es_hist_99_pct:.2f}%"
        ],
        "Expected Violations": [
            "5.00%", "5.00%", "5.00%", "5.00%", 
            "1.00%", "1.00%", "1.00%", "1.00%"
        ],
        "Efficiency": [
            "Good" if abs(var_param_95_pct - 5.0) < 1.0 else "Needs Improvement",
            "Good" if abs(es_param_95_pct - 5.0) < 1.0 else "Needs Improvement",
            "Good" if abs(var_hist_95_pct - 5.0) < 1.0 else "Needs Improvement",
            "Good" if abs(es_hist_95_pct - 5.0) < 1.0 else "Needs Improvement",
            "Good" if abs(var_param_99_pct - 1.0) < 0.5 else "Needs Improvement",
            "Good" if abs(es_param_99_pct - 1.0) < 0.5 else "Needs Improvement",
            "Good" if abs(var_hist_99_pct - 1.0) < 0.5 else "Needs Improvement",
            "Good" if abs(es_hist_99_pct - 1.0) < 0.5 else "Needs Improvement"
        ]
    }
    
    results_df = pd.DataFrame(results_data)
    
    # Style the DataFrame
    styled_results = results_df.style.apply(lambda x: ['background-color: #d4edda' if val == 'Good' else 'background-color: #f8d7da' 
                                                  for val in x], axis=0, subset=['Efficiency'])
    
    st.dataframe(styled_results, hide_index=True)
    
    # Alternative approach with volatility-based VaR
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>Moving Volatility VaR Analysis</h2>", unsafe_allow_html=True)
    
    st.write("""
    The formula used for Volatility-based VaR is:
    
    VaR_(1-α) = q_α × σ^252_t
    
    Where:
    - q_α is the normal distribution percentile at significance level α
    - σ^252_t is the standard deviation from a rolling window of 252 returns
    """)
    
    # Calculate volatility-based VaR
    def volatility_var(returns, alpha=0.05, window=252):
        """Calculate VaR using rolling volatility and normal distribution assumption"""
        q_alpha = norm.ppf(alpha)  # This is negative for losses
        rolling_std = returns.rolling(window=window).std()
        var = q_alpha * rolling_std
        return var
    
    # Calculate for both 95% and 99% confidence
    vol_var_95 = volatility_var(returns, alpha=0.05, window=window_size)
    vol_var_99 = volatility_var(returns, alpha=0.01, window=window_size)
    
    # Shift for prediction
    vol_var_95_shifted = vol_var_95.shift(1)
    vol_var_99_shifted = vol_var_99.shift(1)
    
    # Calculate violations for volatility-based VaR
    vol_var_95_viol, vol_var_95_pct = calculate_violations(returns, vol_var_95)
    vol_var_99_viol, vol_var_99_pct = calculate_violations(returns, vol_var_99)
    
    # Plot the volatility-based VaR
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.plot(returns.index, returns, label="Daily Returns", color="grey", alpha=0.4)
    ax.plot(vol_var_95_shifted.index, vol_var_95_shifted, 
            label="Volatility VaR 95%", color="crimson", linestyle="-")
    ax.plot(vol_var_99_shifted.index, vol_var_99_shifted, 
            label="Volatility VaR 99%", color="darkred", linestyle="-")
    
    # For comparison, you can add the parametric VaR
    ax.plot(var_param_95_shifted.index, var_param_95_shifted, 
            label="Parametric VaR 95%", color="blue", linestyle="--", alpha=0.6)
    ax.plot(var_param_99_shifted.index, var_param_99_shifted, 
            label="Parametric VaR 99%", color="navy", linestyle="--", alpha=0.6)
    
    # Plot styling
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.legend(loc="lower left")
    ax.set_title("Moving Volatility VaR vs. Parametric VaR")
    ax.set_xlabel("Date")
    ax.set_ylabel("Return")
    ax.grid(True, linestyle="--", alpha=0.5)
    
    st.pyplot(fig)
    
    # Display the volatility VaR efficiency results
    st.subheader("Volatility-Based VaR Efficiency")
    
    vol_results_data = {
        "Risk Metric": ["Volatility VaR 95%", "Volatility VaR 99%"],
        "Violations": [vol_var_95_viol, vol_var_99_viol],
        "Violation Percentage (%)": [f"{vol_var_95_pct:.2f}%", f"{vol_var_99_pct:.2f}%"],
        "Expected Violations": ["5.00%", "1.00%"],
        "Efficiency": [
            "Good" if abs(vol_var_95_pct - 5.0) < 1.0 else "Needs Improvement",
            "Good" if abs(vol_var_99_pct - 1.0) < 0.5 else "Needs Improvement"
        ]
    }
    
    vol_results_df = pd.DataFrame(vol_results_data)
    
    # Style the DataFrame
    styled_vol_results = vol_results_df.style.apply(lambda x: ['background-color: #d4edda' if val == 'Good' else 'background-color: #f8d7da' 
                                                        for val in x], axis=0, subset=['Efficiency'])
    
    st.dataframe(styled_vol_results, hide_index=True)

    # Conclusions
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>Conclusions</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="conclusion">
        <p><strong>Distribution Analysis:</strong> Nike's return distribution exhibits characteristics typical of financial assets - 
        notably excess kurtosis (heavy tails) and slight skewness. The Shapiro-Wilk test confirms the non-normality of returns, 
        which is visually apparent in both the histogram and Q-Q plot.</p>
        
        <p><strong>Risk Metrics Performance:</strong> Our risk models generally provide reasonable estimates of potential losses. 
        The Historical and Parametric approaches show different strengths, with Historical VaR typically capturing market behavior 
        better during periods of high volatility.</p>
        
        <p><strong>Expected Shortfall (ES/CVaR):</strong> As expected, ES measures provide more conservative risk estimates by 
        capturing the average of extreme losses beyond VaR. This makes them particularly valuable during market stress periods.</p>
        
        <p><strong>Volatility-Based VaR:</strong> The moving volatility-based VaR demonstrates the importance of dynamic risk 
        assessment that adapts to changing market conditions. This approach shows particular value during periods of volatility 
        clustering.</p>
        
        <p>Overall, financial risk assessment requires multiple complementary approaches rather than relying on a single metric. 
        A combination of VaR and ES metrics, implemented with different methodologies and confidence levels, provides""")