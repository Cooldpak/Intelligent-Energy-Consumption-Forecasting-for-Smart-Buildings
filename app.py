import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as ob
import os
import joblib
from datetime import datetime

# Import project modules
from features import load_and_preprocess_data, resample_to_hourly, engineer_features, split_train_test_chronological
from models import train_and_save_classical_models
from lstm import train_lstm_model
from anomalies import run_anomaly_detection_pipeline
from optimizer import simulate_load_shifting, generate_advisories, RATES

# Page Config
st.set_page_config(
    page_title="Intelligent Energy Control Center",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium UI styling override using custom CSS (Glassmorphism + Dark Mode Accent)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main title styling */
    .main-title {
        font-weight: 800;
        background: linear-gradient(135deg, #FF6B6B 0%, #4D96FF 50%, #6BCB77 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        margin-bottom: 0.2rem;
    }
    
    /* Subtitle styling */
    .sub-title {
        font-weight: 300;
        color: #A5F1E9;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Glassmorphism Metric Card Container */
    .metric-container {
        display: flex;
        justify-content: space-between;
        gap: 15px;
        margin-bottom: 25px;
    }
    
    /* Individual card styling */
    .metric-card {
        flex: 1;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        transition: transform 0.3s ease;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border: 1px solid rgba(255, 255, 255, 0.25);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #4D96FF;
        margin-top: 5px;
    }
    
    .metric-label {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #8E92BC;
        font-weight: 600;
    }
    
    /* Custom status colors */
    .color-energy { color: #54B435; }
    .color-temp { color: #FFA559; }
    .color-alert { color: #FF6464; }
    .color-cost { color: #F7D060; }
    
</style>
""", unsafe_allow_html=True)

# Path definitions
DATA_PATH = "D:\\Smart_energy_predictor\\energydata_complete.csv"
SAVE_DIR = "D:\\Smart_energy_predictor\\data"

# App State Helper
def is_pipeline_trained():
    predictions_file = os.path.join(SAVE_DIR, 'model_predictions.csv')
    metadata_file = os.path.join(SAVE_DIR, 'model_metadata.joblib')
    anomaly_file = os.path.join(SAVE_DIR, 'anomaly_complete_results.csv')
    return os.path.exists(predictions_file) and os.path.exists(metadata_file) and os.path.exists(anomaly_file)

# Welcome Screen & Data pipeline trigger
if not os.path.exists(DATA_PATH):
    st.error(f"❌ Dataset not found at `{DATA_PATH}`. Please place the CSV file in your workspace directory.")
    st.stop()

# Header Area
st.markdown('<div class="main-title">INTELLIGENT ENERGY CONTROL CENTER</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Smart Building Forecasting, Anomaly Detection & Load-Shifting Optimization</div>', unsafe_allow_html=True)

# Sidebar configurations
st.sidebar.image("https://img.icons8.com/clouds/200/lightning-bolt.png", width=120)
st.sidebar.markdown("### ⚙️ Control panel")

building_type = st.sidebar.selectbox(
    "Select Smart Building Unit",
    ["Main Office Headquarters (Building A)", "Residential Complex (Block B)", "Community Health Clinic"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧬 Pipeline Execution")

if st.sidebar.button("🚀 Run Complete ML/DL Pipeline"):
    with st.spinner("Executing Data Science Pipeline..."):
        # Create progress indicators
        progress_bar = st.progress(0)
        
        # Step 1: Classical Models
        st.write("⏱️ 1. Training Linear Regression & XGBoost models...")
        classical_meta = train_and_save_classical_models(DATA_PATH)
        progress_bar.progress(35)
        
        # Step 2: PyTorch LSTM
        st.write("⏱️ 2. Training PyTorch LSTM Sequence model (approx. 20s)...")
        lstm_metrics = train_lstm_model(DATA_PATH, epochs=5)
        progress_bar.progress(75)
        
        # Step 3: Anomaly Detection
        st.write("⏱️ 3. Fitting Isolation Forest & residual statistical thresholds...")
        run_anomaly_detection_pipeline(DATA_PATH)
        progress_bar.progress(100)
        
        st.success("✅ Complete pipeline executed! Refreshing dashboard data...")
        st.rerun()

# Check pipeline status before rendering dashboard
if not is_pipeline_trained():
    st.warning("⚠️ **Models not trained yet.** Please click the **Run Complete ML/DL Pipeline** button in the sidebar to load the data, train the baseline/advanced forecasting models, and activate the anomaly engine.")
    
    # Render preview info about the dataset
    st.markdown("### 📊 Dataset Overview (Preview)")
    df_preview = pd.read_csv(DATA_PATH, nrows=5)
    st.dataframe(df_preview)
    st.info("💡 *Note: The pipeline aggregates this 10-minute sensor telemetry to hourly averages/sums, engineers features (degree days, calendar index, rolling/lag metrics), trains Linear Regression, XGBoost, and a PyTorch LSTM, and computes recommendations and anomaly alerts.*")
    st.stop()

# Load Dashboard Data
@st.cache_data
def load_cached_data():
    df_raw = load_and_preprocess_data(DATA_PATH)
    df_hourly = resample_to_hourly(df_raw)
    
    # Load model predictions
    preds_df = pd.read_csv(os.path.join(SAVE_DIR, 'model_predictions.csv'), index_col='date')
    preds_df.index = pd.to_datetime(preds_df.index)
    
    # Load anomaly data
    anomaly_df = pd.read_csv(os.path.join(SAVE_DIR, 'anomaly_complete_results.csv'), index_col='date')
    anomaly_df.index = pd.to_datetime(anomaly_df.index)
    
    # Load metadata
    metadata = joblib.load(os.path.join(SAVE_DIR, 'model_metadata.joblib'))
    
    return df_hourly, preds_df, anomaly_df, metadata

df_hourly, preds_df, anomaly_df, metadata = load_cached_data()

# -----------------
# 1. METRIC CARDS AREA
# -----------------
# Get latest telemetry index
latest_time = df_hourly.index[-1]
latest_load = df_hourly.loc[latest_time, 'Appliances']
latest_temp = df_hourly.loc[latest_time, 'T_out']
latest_humidity = df_hourly.loc[latest_time, 'RH_out']

# Get active anomalies in the last 48 hours
recent_anoms = anomaly_df.iloc[-48:]['is_anomaly_residual'].sum()

st.markdown(f"""
<div class="metric-container">
    <div class="metric-card">
        <div class="metric-label">Latest Measured Load</div>
        <div class="metric-value color-energy">{int(latest_load)} Wh</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Outdoor Temperature</div>
        <div class="metric-value color-temp">{latest_temp:.1f} °C</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Outdoor Humidity</div>
        <div class="metric-value color-cost">{latest_humidity:.1f} %</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Active Warnings (Last 48h)</div>
        <div class="metric-value color-alert">{recent_anoms} Alerts</div>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------
# TABS DIVISION
# -----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "⚡ Real-Time Monitoring & anomalies", 
    "📈 Data deep-dive & EDA", 
    "🧠 Forecasting arena", 
    "🍃 Cost optimizer & advisories"
])

# ==========================================
# TAB 1: REAL-TIME MONITORING & ANOMALIES
# ==========================================
with tab1:
    st.subheader("Building Telemetry & Anomaly Analysis")
    st.markdown("This section overlays statistical anomaly alerts on top of real energy data, identifying unusual peaks that need maintenance or attention.")
    
    # Filter display duration
    days_to_show = st.slider("Select telemetry historical window (days)", 3, 30, 7, key='telemetry_slider')
    slice_size = days_to_show * 24
    
    df_slice = anomaly_df.iloc[-slice_size:]
    
    # Plotly Line Chart for energy consumption
    fig_telemetry = ob.Figure()
    
    # Actual Load line
    fig_telemetry.add_trace(ob.Scatter(
        x=df_slice.index, 
        y=df_slice['Actual'],
        mode='lines',
        name='Energy Consumption (Wh)',
        line=dict(color='#4D96FF', width=2)
    ))
    
    # Highlight anomalies
    anoms_only = df_slice[df_slice['is_anomaly_residual'] == 1]
    fig_telemetry.add_trace(ob.Scatter(
        x=anoms_only.index,
        y=anoms_only['Actual'],
        mode='markers',
        name='Forecasting Surprises (Anomaly)',
        marker=dict(color='#FF6464', size=10, symbol='circle', line=dict(color='white', width=1.5)),
        hovertemplate='<b>Anomaly Detected</b><br>Time: %{x}<br>Actual Usage: %{y} Wh<extra></extra>'
    ))
    
    fig_telemetry.update_layout(
        template='plotly_dark',
        height=450,
        margin=dict(l=20, r=20, t=10, b=10),
        xaxis_title='Timestamp',
        yaxis_title='Consumption (Wh)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    st.plotly_chart(fig_telemetry, use_container_width=True)
    
    # Anomaly listing table
    st.markdown("### 🚨 Log of Active Energy Anomalies")
    recent_anom_log = anomaly_df[anomaly_df['is_anomaly_residual'] == 1].tail(10).copy()
    if len(recent_anom_log) > 0:
        recent_anom_log['residual_z'] = recent_anom_log['residual_z'].apply(lambda x: f"{x:.2f} σ")
        recent_anom_log['residual'] = recent_anom_log['residual'].apply(lambda x: f"{x:+.1f} Wh")
        
        # Display table with key columns
        st.table(recent_anom_log[['Actual', 'XGB_Pred', 'residual', 'residual_z']].rename(columns={
            'Actual': 'Actual Load (Wh)',
            'XGB_Pred': 'Expected Load (Wh)',
            'residual': 'Deviation',
            'residual_z': 'Severity (Z-Score)'
        }))
    else:
        st.success("No anomalies detected in the current history window.")

# ==========================================
# TAB 2: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
with tab2:
    st.subheader("Jupyter-style Interactive Exploratory Analysis")
    st.markdown("Examine the underlying seasonal patterns, time dynamics, and external weather correlations from the training historical dataset.")
    
    eda_mode = st.radio(
        "Choose Exploratory Analysis Visual",
        ["Hourly Diurnal Patterns", "Weekly Load Dynamics", "Outside Temperature vs Energy Load", "Time-series Seasonal Decomposition"],
        horizontal=True
    )
    
    # 1. Hourly Pattern
    if eda_mode == "Hourly Diurnal Patterns":
        st.markdown("#### Diurnal Cycles: Average Consumption by Hour of the Day")
        df_hourly['hour'] = df_hourly.index.hour
        hourly_grouped = df_hourly.groupby('hour')['Appliances'].agg(['mean', 'std']).reset_index()
        
        fig_hour = px.line(
            hourly_grouped, x='hour', y='mean', 
            labels={'mean': 'Average Usage (Wh)', 'hour': 'Hour of Day'},
            template='plotly_dark', height=400
        )
        # Add shading for variance (std)
        fig_hour.add_trace(ob.Scatter(
            x=pd.concat([hourly_grouped['hour'], hourly_grouped['hour'].iloc[::-1]]),
            y=pd.concat([hourly_grouped['mean'] + hourly_grouped['std'], (hourly_grouped['mean'] - hourly_grouped['std']).iloc[::-1]]),
            fill='toself',
            fillcolor='rgba(77, 150, 255, 0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Variance (Standard Deviation)',
            showlegend=True
        ))
        st.plotly_chart(fig_hour, use_container_width=True)
        st.info("💡 **Observation**: We observe clear dual peaks—one in the morning (around 8 AM) as building activities start, and a larger peak in the early evening (around 5 PM to 8 PM) representing domestic/commercial thermal load spikes.")

    # 2. Weekly Load Dynamics
    elif eda_mode == "Weekly Load Dynamics":
        st.markdown("#### Weekly Cycles: Comparing Workdays vs Weekends")
        df_hourly['day_of_week'] = df_hourly.index.day_name()
        df_hourly['day_num'] = df_hourly.index.dayofweek
        
        weekly_grouped = df_hourly.groupby(['day_num', 'day_of_week'])['Appliances'].mean().reset_index()
        
        fig_week = px.bar(
            weekly_grouped, x='day_of_week', y='Appliances',
            labels={'Appliances': 'Average Usage (Wh)', 'day_of_week': 'Day of Week'},
            color='Appliances', color_continuous_scale='Bluered',
            template='plotly_dark', height=400
        )
        st.plotly_chart(fig_week, use_container_width=True)
        st.info("💡 **Observation**: Workdays demonstrate slightly higher baseline loads, whereas weekends exhibit different peak shapes (shifting towards mid-day usage).")

    # 3. Temp vs Load correlation
    elif eda_mode == "Outside Temperature vs Energy Load":
        st.markdown("#### HVAC Impact: Outdoor Temperature vs Energy Load")
        
        fig_scatter = px.scatter(
            df_hourly.sample(n=1000, random_state=42), 
            x='T_out', y='Appliances',
            color='RH_out', color_continuous_scale='Viridis',
            labels={'T_out': 'Outdoor Temperature (°C)', 'Appliances': 'Appliances Energy (Wh)', 'RH_out': 'Humidity (%)'},
            opacity=0.6,
            template='plotly_dark', height=400
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.info("💡 **Observation**: Energy usage increases significantly at colder temperatures (increased heating requirements) and shows another upward tilt during hot humid hours (increased air conditioning loads). This justifies engineering the **Heating and Cooling Degree Hours (HDH/CDH)** features.")

    # 4. Seasonal Decomposition
    elif eda_mode == "Time-series Seasonal Decomposition":
        st.markdown("#### Statistical Decomposition (Trend, Seasonal, Residual)")
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        # Decompose 1 week of data to keep it clean and readable
        decomp_data = df_hourly['Appliances'].iloc[-168:] # 1 week = 168 hours
        decomposition = seasonal_decompose(decomp_data, model='additive', period=24)
        
        fig_decomp = ob.Figure()
        fig_decomp.add_trace(ob.Scatter(x=decomp_data.index, y=decomposition.observed, name='Observed (Actual)', line=dict(color='#4D96FF')))
        fig_decomp.add_trace(ob.Scatter(x=decomp_data.index, y=decomposition.trend, name='Trend', line=dict(color='#FFA559')))
        fig_decomp.add_trace(ob.Scatter(x=decomp_data.index, y=decomposition.seasonal, name='Daily Seasonal', line=dict(color='#54B435')))
        fig_decomp.add_trace(ob.Scatter(x=decomp_data.index, y=decomposition.resid, name='Noise (Residuals)', line=dict(color='grey', dash='dot')))
        
        fig_decomp.update_layout(
            template='plotly_dark',
            height=500,
            xaxis_title='Timestamp',
            yaxis_title='Decomposition scale (Wh)',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        st.plotly_chart(fig_decomp, use_container_width=True)
        st.info("💡 **Observation**: Decomposing energy consumption separates the underlying weekly trend and predictable daily circadian load curves from erratic, unexpected noise (which we use for anomaly alerts).")

# ==========================================
# TAB 3: FORECASTING ARENA
# ==========================================
with tab3:
    st.subheader("Model Leaderboard & Interactive Forecast Playground")
    st.markdown("Compare predictions from the baseline Linear Regression, intermediate XGBoost, and advanced PyTorch LSTM sequence neural network.")
    
    # Model Comparison Table
    st.markdown("### 🏆 Leaderboard: Model Evaluation Metrics")
    
    # Load metrics from metadata
    metrics_dict = metadata['metrics']
    metrics_df = pd.DataFrame(metrics_dict).T
    
    st.table(metrics_df.style.highlight_min(subset=['MAE', 'RMSE', 'MAPE'], color='#3c1b1b')
                           .highlight_max(subset=['R2'], color='#1b3c1b'))
    
    st.info("🏆 **XGBoost** and **LSTM** show significant improvements in R² and MAE compared to the Linear Regression baseline by effectively capturing non-linear weather interactions and historical sequence dependencies.")
    
    # Forecast plot customization
    st.markdown("### 🔍 Forecast Chart Overlay")
    models_to_plot = st.multiselect(
        "Select models to show on forecast chart",
        ['Linear Regression', 'XGBoost', 'LSTM (PyTorch)'],
        default=['XGBoost', 'LSTM (PyTorch)']
    )
    
    days_forecast = st.slider("Select forecast length (days)", 1, 7, 3, key='forecast_slider')
    forecast_points = days_forecast * 24
    
    df_forecast_slice = preds_df.iloc[-forecast_points:]
    
    fig_forecast = ob.Figure()
    
    # Actual Load
    fig_forecast.add_trace(ob.Scatter(
        x=df_forecast_slice.index, y=df_forecast_slice['Actual'],
        name='Actual Load', line=dict(color='#4D96FF', width=2.5)
    ))
    
    # Conditional overlays
    if 'Linear Regression' in models_to_plot:
        fig_forecast.add_trace(ob.Scatter(
            x=df_forecast_slice.index, y=df_forecast_slice['LR_Pred'],
            name='Linear Regression (Baseline)', line=dict(color='#FF6464', dash='dash')
        ))
    if 'XGBoost' in models_to_plot:
        fig_forecast.add_trace(ob.Scatter(
            x=df_forecast_slice.index, y=df_forecast_slice['XGB_Pred'],
            name='XGBoost Regressor', line=dict(color='#6BCB77', width=2)
        ))
    if 'LSTM (PyTorch)' in models_to_plot and 'LSTM_Pred' in df_forecast_slice.columns:
        fig_forecast.add_trace(ob.Scatter(
            x=df_forecast_slice.index, y=df_forecast_slice['LSTM_Pred'],
            name='LSTM Sequence NN', line=dict(color='#FFA559', width=2)
        ))
        
    fig_forecast.update_layout(
        template='plotly_dark',
        height=450,
        xaxis_title='Timestamp',
        yaxis_title='Consumption (Wh)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    # Dynamic Simulator Playground
    st.markdown("### 🎮 Interactive Simulation Playground")
    st.markdown("Tweak environmental factors dynamically to see how they impact predicted energy loads in real-time.")
    
    col1, col2 = st.columns(2)
    with col1:
        sim_temp = st.slider("Simulate outdoor temperature change (°C)", -10.0, 15.0, 0.0, step=1.0, help="Simulate a shift in outdoor temperatures to model heating/cooling demands.")
        sim_occupancy = st.select_slider("Simulate building occupancy level", options=["Empty (Unoccupied)", "Low", "Normal (Baseline)", "High (Maximum occupancy)"], value="Normal (Baseline)")
    
    # Predict impact based on coefficients/xgboost features
    # Since we can't retrain in real-time, we scale the baseline forecasts dynamically to simulate HVAC changes
    # Heating Degree Hours / Cooling Degree Hours scale
    base_avg_load = df_forecast_slice['XGB_Pred'].mean()
    
    # Occupancy multiplier
    occ_multiplier = {"Empty (Unoccupied)": 0.65, "Low": 0.85, "Normal (Baseline)": 1.0, "High (Maximum occupancy)": 1.15}
    
    # Temperature multiplier: for every degree colder than comfort baseline (18C), increase heating load by 4%
    # For every degree warmer, increase cooling AC load by 6%
    temp_effect = 0.0
    if sim_temp < 0:
        temp_effect = abs(sim_temp) * 0.035 # heating increases
    else:
        temp_effect = sim_temp * 0.055 # AC cooling increases
        
    final_multiplier = occ_multiplier[sim_occupancy] + temp_effect
    simulated_load = df_forecast_slice['XGB_Pred'] * final_multiplier
    
    with col2:
        st.markdown(f"#### 📊 Simulated Load Summary")
        st.metric(
            label="Simulated Avg Energy Consumption", 
            value=f"{int(base_avg_load * final_multiplier)} Wh",
            delta=f"{int((final_multiplier - 1.0) * 100):+d}% from baseline"
        )
        
        # Mini comparison chart
        sim_chart_df = pd.DataFrame({
            'Baseline Forecast': df_forecast_slice['XGB_Pred'],
            'Simulated Load': simulated_load
        }, index=df_forecast_slice.index)
        
        fig_sim = px.line(
            sim_chart_df, labels={'value': 'Consumption (Wh)', 'date': 'Timestamp'},
            color_discrete_map={'Baseline Forecast': '#8E92BC', 'Simulated Load': '#FFA559'},
            template='plotly_dark', height=250
        )
        fig_sim.update_layout(margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation='h'))
        st.plotly_chart(fig_sim, use_container_width=True)

# ==========================================
# TAB 4: COST OPTIMIZER & SMART ADVISORIES
# ==========================================
with tab4:
    st.subheader("Decision Support: Cost Shifting Optimizer & advisory Panel")
    st.markdown("Maximize building cost efficiency and reduce your carbon footprint by shifting peak electricity consumption to off-peak pricing windows.")
    
    # Set pricing table layout
    st.markdown("#### 💳 Current Building Utility Pricing Structure")
    col_t1, col_t2, col_t3 = st.columns(3)
    col_t1.metric("Peak Rate (2PM - 8PM)", f"${RATES['PEAK']:.2f} / kWh", "Highest Grid Stress", delta_color="inverse")
    col_t2.metric("Shoulder Rate (Daytime/Late Night)", f"${RATES['SHOULDER']:.2f} / kWh", "Moderate Load")
    col_t3.metric("Off-Peak Rate (10PM - 6AM)", f"${RATES['OFF_PEAK']:.2f} / kWh", "Cheapest & Cleanest", delta_color="normal")
    
    st.markdown("---")
    
    # Shift parameters
    shift_pct = st.slider("Select proportion of flexible load to shift (%)", 0, 50, 25, step=5) / 100.0
    
    # Run simulation
    # Using XGBoost predictions for optimization pipeline
    opt_results = simulate_load_shifting(df_hourly.iloc[-168:], target_col='Appliances', shift_pct=shift_pct)
    
    # Cost Metrics displays
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Baseline Energy Cost", f"${opt_results['baseline_cost']:.2f}")
    col_m2.metric("Optimized Energy Cost", f"${opt_results['optimized_cost']:.2f}", f"-{opt_results['savings_percentage']}%")
    col_m3.metric("Projected Cost Savings", f"${opt_results['cost_savings']:.2f}", "Net Savings", delta_color="normal")
    col_m4.metric("Carbon Footprint Avoided", f"{opt_results['co2_savings']:.2f} kg", "CO2 Reduced", delta_color="normal")
    
    # Chart comparison
    st.markdown("### 📊 Load Shifting Profile Comparison")
    st.markdown("The green area indicates shifted usage during cheaper off-peak hours, flattening the red peak hours.")
    
    df_comparison = opt_results['usage_comparison'].head(72) # Show 3 days
    
    fig_opt = ob.Figure()
    
    # Forecasted peak load
    fig_opt.add_trace(ob.Scatter(
        x=df_comparison.index, y=df_comparison['Forecasted'],
        fill='tozeroy', fillcolor='rgba(255, 100, 100, 0.1)',
        line=dict(color='#FF6464', width=2),
        name='Baseline Scheduled Load'
    ))
    
    # Optimized shifted load
    fig_opt.add_trace(ob.Scatter(
        x=df_comparison.index, y=df_comparison['Optimized'],
        fill='tozeroy', fillcolor='rgba(107, 203, 119, 0.15)',
        line=dict(color='#6BCB77', width=2),
        name='Optimized Load Profile'
    ))
    
    fig_opt.update_layout(
        template='plotly_dark',
        height=380,
        xaxis_title='Timestamp',
        yaxis_title='Consumption (Wh)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    st.plotly_chart(fig_opt, use_container_width=True)
    
    # Advisory advisories panel
    st.markdown("### 💡 Tomorrow's Smart Energy Advisories")
    advisories = generate_advisories(df_hourly.iloc[-168:], target_col='Appliances')
    
    for adv in advisories:
        if adv.get('severity') == 'High':
            st.warning(adv['message'])
        elif adv.get('severity') == 'Medium':
            st.info(adv['message'])
        else:
            st.success(adv['message'])
