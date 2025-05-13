import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
from scipy import stats

# Path to the data file using absolute paths
input_file = 'data/bitcoin_data_2024-11-03_to_2025-05-02.csv'
output_file = 'data/bitcoin_data_cleaned.csv'

# Read the data
print("Loading data...")
df = pd.read_csv(input_file)

# 1. Remove unnecessary columns
print("Removing unnecessary columns...")
columns_to_drop = ['block_version', 'difficulty', 'hash_rate', 
                  'prev6_block_median_fee_rate', 'mempool_fee_histogram']
df = df.drop(columns=columns_to_drop, errors='ignore')

# 2. Convert timestamp to datetime and set as index
print("Converting timestamp to datetime...")
df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
df = df.set_index('datetime')
df = df.sort_index()  # Ensure data is sorted by time

# 3. Handle missing values
print("Handling missing values...")
# For important columns, first forward fill
important_columns = [
    'block_median_fee_rate', 'mempool_size_bytes', 'mempool_tx_count',
    'mempool_min_fee_rate', 'mempool_median_fee_rate', 'mempool_avg_fee_rate'
]

# Fill NaN with forward fill first
df[important_columns] = df[important_columns].ffill()

# Then use backward fill for any remaining NaNs
df[important_columns] = df[important_columns].bfill()

# For other numerical columns, fill with the mean
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
other_numeric_cols = [col for col in numeric_cols if col not in important_columns]
for col in other_numeric_cols:
    df[col] = df[col].fillna(df[col].mean())

# 4. Resample to hourly intervals
print("Resampling to hourly intervals...")
# Define aggregation methods for different columns
agg_dict = {}
for col in df.columns:
    if col in ['block_height']:
        agg_dict[col] = 'max'  # Take the max block height in each hour
    elif col in ['block_median_fee_rate', 'mempool_median_fee_rate']:
        agg_dict[col] = 'median'
    elif col in ['block_interval_seconds']:
        agg_dict[col] = 'mean'
    else:
        agg_dict[col] = 'mean'  # Default to mean for other columns

# Resample to hourly data
df_hourly = df.resample('1H').agg(agg_dict)

# Interpolate missing values after resampling
df_hourly = df_hourly.interpolate(method='linear')

# 5. Handle outliers
print("Handling outliers...")
# Function to cap outliers using IQR method
def cap_outliers(series, lower_quantile=0.25, upper_quantile=0.75, iqr_multiplier=1.5):
    q1 = series.quantile(lower_quantile)
    q3 = series.quantile(upper_quantile)
    iqr = q3 - q1
    lower_bound = q1 - (iqr_multiplier * iqr)
    upper_bound = q3 + (iqr_multiplier * iqr)
    return series.clip(lower=lower_bound, upper=upper_bound)

# Columns to check for outliers
fee_columns = ['block_median_fee_rate', 'mempool_min_fee_rate', 
               'mempool_median_fee_rate', 'mempool_avg_fee_rate']

# Cap outliers
for col in fee_columns:
    df_hourly[col] = cap_outliers(df_hourly[col])

# 6. Normalize numerical variables
print("Normalizing numerical variables...")
# Columns to normalize
columns_to_normalize = [
    'block_median_fee_rate',
    'mempool_size_bytes', 'mempool_tx_count', 'mempool_min_fee_rate',
    'mempool_median_fee_rate', 'mempool_avg_fee_rate',
    'block_weight', 'block_tx_count', 'block_interval_seconds',
    'btc_price_usd', 'prev_block_median_fee_rate'
]

# Create a copy of the original data before normalization
df_original = df_hourly.copy()

# Apply StandardScaler to selected columns
scaler = StandardScaler()
df_hourly[columns_to_normalize] = scaler.fit_transform(df_hourly[columns_to_normalize])

# Save the cleaned data
print("Saving cleaned data...")
# Create directory if it doesn't exist
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Save both normalized and non-normalized versions
df_hourly.to_csv(output_file)
df_original.to_csv(output_file.replace('.csv', '_original.csv'))

print(f"Data cleaning completed. Files saved to: {output_file} and {output_file.replace('.csv', '_original.csv')}")

# Generate basic statistics
print("\nData statistics after cleaning:")
print(f"Total records: {len(df_hourly)}")
print(f"Date range: {df_hourly.index.min()} to {df_hourly.index.max()}")
print(f"Missing values: {df_hourly.isna().sum().sum()}") 