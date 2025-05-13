import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
import os

# Set style for better visualization
plt.style.use('ggplot')
sns.set(font_scale=1.2)
plt.rcParams['figure.figsize'] = (14, 8)

# Create output directory for plots
output_dir = 'data/plots'
os.makedirs(output_dir, exist_ok=True)

print("Loading cleaned data...")
# Load the cleaned data (using the non-resampled version for detailed analysis)
df = pd.read_csv('data/bitcoin_data_cleaned_no_resample_original.csv')

# Convert datetime to proper datetime object and set as index
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.set_index('datetime')

# 1. Time Series Analysis of Fees
print("Analyzing fee time series...")

# Resample to hourly data for smoother visualization
fee_hourly = df['block_median_fee_rate'].resample('1H').mean().fillna(method='ffill')

# Create a time series plot
plt.figure(figsize=(16, 8))
plt.plot(fee_hourly.index, fee_hourly.values, linewidth=1.5)
plt.title('Bitcoin Block Median Fee Rate Over Time', fontsize=16)
plt.ylabel('Median Fee Rate (sat/vB)', fontsize=14)
plt.xlabel('Date', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/fee_time_series.png', dpi=300)
plt.close()

# 2. Fee Analysis by Hour of Day
print("Analyzing fees by hour of day...")
# Extract hour of day
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek

# Compute hourly statistics
hourly_stats = df.groupby('hour')['block_median_fee_rate'].agg(['mean', 'std']).reset_index()

# Plot mean fee by hour
plt.figure(figsize=(14, 7))
plt.bar(hourly_stats['hour'], hourly_stats['mean'], yerr=hourly_stats['std'], 
        alpha=0.7, capsize=7, color='cornflowerblue', edgecolor='black')
plt.title('Mean Block Median Fee Rate by Hour of Day', fontsize=16)
plt.ylabel('Mean Fee Rate (sat/vB)', fontsize=14)
plt.xlabel('Hour of Day (0-23)', fontsize=14)
plt.xticks(range(0, 24, 2))
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{output_dir}/fee_by_hour_mean.png', dpi=300)
plt.close()

# Plot std deviation by hour
plt.figure(figsize=(14, 7))
plt.bar(hourly_stats['hour'], hourly_stats['std'], alpha=0.7, color='lightcoral', edgecolor='black')
plt.title('Standard Deviation of Block Median Fee Rate by Hour of Day', fontsize=16)
plt.ylabel('Std Deviation of Fee Rate (sat/vB)', fontsize=14)
plt.xlabel('Hour of Day (0-23)', fontsize=14)
plt.xticks(range(0, 24, 2))
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{output_dir}/fee_by_hour_std.png', dpi=300)
plt.close()

# 3. Fee Analysis by Day of Week
print("Analyzing fees by day of week...")
# Compute daily statistics
daily_stats = df.groupby('day_of_week')['block_median_fee_rate'].agg(['mean', 'std']).reset_index()

# Map day of week numbers to names
day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
             4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
daily_stats['day_name'] = daily_stats['day_of_week'].map(day_names)

# Plot mean fee by day of week
plt.figure(figsize=(14, 7))
plt.bar(daily_stats['day_name'], daily_stats['mean'], yerr=daily_stats['std'], 
        alpha=0.7, capsize=7, color='mediumseagreen', edgecolor='black')
plt.title('Mean Block Median Fee Rate by Day of Week', fontsize=16)
plt.ylabel('Mean Fee Rate (sat/vB)', fontsize=14)
plt.xlabel('Day of Week', fontsize=14)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{output_dir}/fee_by_day_mean.png', dpi=300)
plt.close()

# Plot std deviation by day of week
plt.figure(figsize=(14, 7))
plt.bar(daily_stats['day_name'], daily_stats['std'], 
        alpha=0.7, color='darkorange', edgecolor='black')
plt.title('Standard Deviation of Block Median Fee Rate by Day of Week', fontsize=16)
plt.ylabel('Std Deviation of Fee Rate (sat/vB)', fontsize=14)
plt.xlabel('Day of Week', fontsize=14)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{output_dir}/fee_by_day_std.png', dpi=300)
plt.close()

# 4. Heatmap of Day of Week × Hour of Day
print("Creating heatmap for day of week × hour of day...")
# Create pivot table
pivot_data = df.pivot_table(
    values='block_median_fee_rate', 
    index='day_of_week',
    columns='hour', 
    aggfunc='mean'
)

# Create day of week names for y-axis
pivot_data.index = [day_names[day] for day in pivot_data.index]

# Create heatmap
plt.figure(figsize=(16, 10))
sns.heatmap(pivot_data, cmap='viridis', annot=True, fmt='.2f', linewidths=0.5)
plt.title('Mean Block Median Fee Rate by Day of Week and Hour of Day', fontsize=16)
plt.ylabel('Day of Week', fontsize=14)
plt.xlabel('Hour of Day (0-23)', fontsize=14)
plt.tight_layout()
plt.savefig(f'{output_dir}/fee_heatmap.png', dpi=300)
plt.close()

# Create a second version of the heatmap without annotations for clearer color patterns
plt.figure(figsize=(16, 10))
sns.heatmap(pivot_data, cmap='viridis', linewidths=0.5)
plt.title('Mean Block Median Fee Rate by Day of Week and Hour of Day (No Annotations)', fontsize=16)
plt.ylabel('Day of Week', fontsize=14)
plt.xlabel('Hour of Day (0-23)', fontsize=14)
plt.tight_layout()
plt.savefig(f'{output_dir}/fee_heatmap_no_annot.png', dpi=300)
plt.close()

print(f"EDA completed. All plots saved to {output_dir} directory.") 