import pandas as pd
import numpy as np

# Define tariff rates (USD per kWh)
RATES = {
    'PEAK': 0.24,       # 2:00 PM - 8:00 PM (14:00 - 20:00)
    'SHOULDER': 0.14,   # 6:00 AM - 2:00 PM & 8:00 PM - 10:00 PM
    'OFF_PEAK': 0.08    # 10:00 PM - 6:00 AM
}

# Define carbon intensity (kg CO2 per kWh)
CARBON_INTENSITY = {
    'PEAK': 0.60,       # Grid is stressed, peaker plants active
    'SHOULDER': 0.45,   
    'OFF_PEAK': 0.35    # Grid relies on wind/nuclear/hydro
}

def get_tariff_type(hour):
    """
    Classifies a given hour into tariff periods.
    """
    if 14 <= hour < 20:
        return 'PEAK'
    elif 22 <= hour or hour < 6:
        return 'OFF_PEAK'
    else:
        return 'SHOULDER'

def calculate_costs_and_emissions(hourly_usage_wh, hour_indicators):
    """
    Calculates cost and emissions for a series of hourly usage values (in Wh).
    Note: Usage in Wh needs to be divided by 1000 to get kWh.
    """
    costs = []
    emissions = []
    
    for usage_wh, hour in zip(hourly_usage_wh, hour_indicators):
        usage_kwh = usage_wh / 1000.0
        period = get_tariff_type(hour)
        
        costs.append(usage_kwh * RATES[period])
        emissions.append(usage_kwh * CARBON_INTENSITY[period])
        
    return sum(costs), sum(emissions)

def simulate_load_shifting(hourly_df, target_col='XGB_Pred', shift_pct=0.25):
    """
    Simulates shifting a percentage of peak load to off-peak hours.
    Inputs:
        hourly_df: DataFrame containing the forecasts and datetime index.
        target_col: The column containing the predicted energy usage in Wh.
        shift_pct: The proportion of peak load that can be shifted (e.g., 0.25 = 25%).
    Returns:
        A dictionary with metrics and a comparison series.
    """
    df_opt = hourly_df.copy()
    df_opt['hour'] = df_opt.index.hour
    df_opt['period'] = df_opt['hour'].apply(get_tariff_type)
    
    # Calculate baseline costs and emissions
    base_cost, base_co2 = calculate_costs_and_emissions(df_opt[target_col].values, df_opt['hour'].values)
    
    # Simulate load shifting:
    # 1. Calculate how much Wh is shifted out of peak hours
    peak_mask = df_opt['period'] == 'PEAK'
    peak_usage = df_opt.loc[peak_mask, target_col]
    total_peak_usage = peak_usage.sum()
    shifted_energy_total = total_peak_usage * shift_pct
    
    # 2. Subtract the shifted energy from Peak hours
    optimized_usage = df_opt[target_col].copy()
    optimized_usage.loc[peak_mask] = optimized_usage.loc[peak_mask] * (1.0 - shift_pct)
    
    # 3. Add the shifted energy to Off-Peak hours (distribute evenly among off-peak hours)
    off_peak_mask = df_opt['period'] == 'OFF_PEAK'
    num_off_peak = off_peak_mask.sum()
    
    if num_off_peak > 0:
        added_energy_per_hour = shifted_energy_total / num_off_peak
        optimized_usage.loc[off_peak_mask] = optimized_usage.loc[off_peak_mask] + added_energy_per_hour
        
    df_opt['Optimized_Usage'] = optimized_usage
    
    # Calculate optimized costs and emissions
    opt_cost, opt_co2 = calculate_costs_and_emissions(df_opt['Optimized_Usage'].values, df_opt['hour'].values)
    
    cost_savings = base_cost - opt_cost
    co2_savings = base_co2 - opt_co2
    
    return {
        'baseline_cost': round(base_cost, 2),
        'optimized_cost': round(opt_cost, 2),
        'cost_savings': round(cost_savings, 2),
        'savings_percentage': round((cost_savings / base_cost) * 100, 1) if base_cost > 0 else 0,
        'baseline_co2': round(base_co2, 2),
        'optimized_co2': round(opt_co2, 2),
        'co2_savings': round(co2_savings, 2),
        'shifted_kwh': round(shifted_energy_total / 1000.0, 2),
        'usage_comparison': pd.DataFrame({
            'Forecasted': df_opt[target_col],
            'Optimized': df_opt['Optimized_Usage']
        }, index=df_opt.index)
    }

def generate_advisories(df_forecast, target_col='XGB_Pred', threshold_wh=200):
    """
    Generates dynamic, human-readable recommendations based on forecasting.
    """
    advisories = []
    
    # Group by date to examine daily behavior
    df_forecast['date_only'] = df_forecast.index.date
    df_forecast['hour'] = df_forecast.index.hour
    
    # Find the day with the highest peak hour load
    peak_hours = df_forecast[df_forecast['hour'].apply(get_tariff_type) == 'PEAK']
    
    if len(peak_hours) == 0:
        return ["No peak hours detected in the forecast period."]
        
    # Find top 3 hours of highest forecasted peak load
    top_peaks = peak_hours.sort_values(by=target_col, ascending=False).head(3)
    
    for idx, row in top_peaks.iterrows():
        load_wh = row[target_col]
        hour = row['hour']
        date_str = idx.strftime('%A, %b %d')
        
        # Convert hour to AM/PM format
        ampm_hour = idx.strftime('%I:00 %p')
        
        if load_wh > threshold_wh:
            # Recommend shifting a portion of this load
            potential_saving_val = (load_wh * 0.40 / 1000.0) * (RATES['PEAK'] - RATES['OFF_PEAK'])
            co2_saving_val = (load_wh * 0.40 / 1000.0) * (CARBON_INTENSITY['PEAK'] - CARBON_INTENSITY['OFF_PEAK'])
            
            advisories.append({
                'date': date_str,
                'hour': hour,
                'message': f"On **{date_str}** at **{ampm_hour}**, energy demand is projected to spike to **{round(load_wh, 1)} Wh** (Peak Rate). Shifting heavy appliances (e.g. dishwasher, washing machine, dryer) to off-peak hours (after 10:00 PM) could save you **${potential_saving_val:.2f}** and reduce carbon emissions by **{co2_saving_val:.3f} kg CO2** for this hour alone.",
                'severity': 'High' if load_wh > 400 else 'Medium'
            })
            
    # Add a general tip
    advisories.append({
        'date': 'General Advice',
        'hour': None,
        'message': "💡 **Tip**: Program your smart thermostat to 'pre-cool' or 'pre-heat' your building during shoulder hours (e.g. 11:00 AM - 1:00 PM) when electricity rates are lower, and turn off climate units during peak pricing windows (2:00 PM - 8:00 PM).",
        'severity': 'Info'
    })
    
    return advisories

if __name__ == '__main__':
    # Test optimizer
    test_dates = pd.date_range(start='2016-05-01 00:00:00', end='2016-05-02 23:00:00', freq='H')
    # Mock forecast data
    mock_load = [100 + 300 * np.sin(2 * np.pi * h / 24)**2 for h in range(len(test_dates))]
    mock_df = pd.DataFrame({'XGB_Pred': mock_load}, index=test_dates)
    
    results = simulate_load_shifting(mock_df, shift_pct=0.30)
    print("Baseline Cost:", results['baseline_cost'])
    print("Optimized Cost:", results['optimized_cost'])
    print("Savings:", results['cost_savings'])
    
    advisories = generate_advisories(mock_df)
    for adv in advisories:
        print(adv['message'])
