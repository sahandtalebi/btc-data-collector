import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
import os
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_and_prepare_data(file_path='data/bitcoin_data_cleaned_no_resample_original.csv'):
    """
    Load and prepare data for Prophet model with regressors
    
    Args:
        file_path: Path to the cleaned data file
        
    Returns:
        prophet_df: DataFrame prepared for Prophet
        train_df: Training portion (5 months)
        test_df: Testing portion (1 month)
    """
    print("Loading data...")
    # Load the data
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Get the dataset's time range for splitting
    start_date = df['datetime'].min()
    end_date = df['datetime'].max()
    
    print(f"Data spans from {start_date} to {end_date}")
    
    # Calculate the split point (5 months for training, 1 month for testing)
    total_days = (end_date - start_date).days
    train_days = int(total_days * (5/6))  # 5/6 of the data for training
    split_date = start_date + timedelta(days=train_days)
    
    print(f"Splitting data at {split_date} (5 months training, 1 month testing)")
    
    # Split the data
    train_df = df[df['datetime'] < split_date].copy()
    test_df = df[df['datetime'] >= split_date].copy()
    
    print(f"Training data: {len(train_df)} samples")
    print(f"Testing data: {len(test_df)} samples")
    
    # Check for missing values in training data
    print(f"Missing values in training data (top 5):")
    print(train_df.isnull().sum().head())
    
    # Prepare data for Prophet (both train and future dataframes)
    print("Preparing data for Prophet with regressors...")
    
    # Define regressors to use
    regressors = ['btc_price_usd', 'mempool_size_bytes', 'mempool_tx_count', 'block_interval_seconds']
    
    # Create the Prophet dataframe for training
    prophet_train_df = pd.DataFrame()
    prophet_train_df['ds'] = train_df['datetime']
    prophet_train_df['y'] = train_df['block_median_fee_rate']
    
    # Add only available regressors and handle missing values
    available_regressors = []
    for regressor in regressors:
        if regressor in train_df.columns:
            prophet_train_df[regressor] = train_df[regressor]
            available_regressors.append(regressor)
    
    # Create the future dataframe including test period
    prophet_future_df = pd.DataFrame()
    prophet_future_df['ds'] = pd.concat([train_df['datetime'], test_df['datetime']])
    
    # Add regressor values for the entire period
    for regressor in available_regressors:
        prophet_future_df[regressor] = pd.concat([train_df[regressor], test_df[regressor]])
    
    # Print data stats to verify
    print(f"\nShape of prophet_train_df: {prophet_train_df.shape}")
    print(f"Missing values in prophet_train_df: {prophet_train_df.isnull().sum().sum()}")
    
    print(f"Shape of prophet_future_df: {prophet_future_df.shape}")
    print(f"Missing values in prophet_future_df: {prophet_future_df.isnull().sum().sum()}")
    
    # Print the first few rows to check
    print("\nFirst few rows of prophet_train_df:")
    print(prophet_train_df.head())
    
    return prophet_train_df, prophet_future_df, train_df, test_df, available_regressors

def run_prophet_with_regressors():
    """
    Run Prophet model with multiple regressors and evaluate on test data
    """
    # Create output directory
    output_dir = 'models/prophet/output_regressors'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and prepare data
    prophet_train_df, prophet_future_df, train_df, test_df, regressors = load_and_prepare_data()
    
    # Initialize the model with seasonality settings
    print("Initializing Prophet model with regressors...")
    model = Prophet(
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10,
        daily_seasonality=True,
        weekly_seasonality=True
    )
    
    # Add regressors to the model
    for regressor in regressors:
        print(f"Adding regressor: {regressor}")
        model.add_regressor(regressor)
    
    # Fit the model on training data
    print("Fitting Prophet model...")
    model.fit(prophet_train_df)
    
    # Predict for the entire future dataframe (includes test period)
    print("Generating predictions...")
    forecast = model.predict(prophet_future_df)
    
    # Plot components
    print("Plotting components...")
    fig1 = model.plot_components(forecast)
    fig1.savefig(f"{output_dir}/components.png", dpi=300, bbox_inches='tight')
    
    # Plot forecast
    fig2 = model.plot(forecast)
    fig2.savefig(f"{output_dir}/forecast.png", dpi=300, bbox_inches='tight')
    
    # Save forecast to CSV
    forecast.to_csv(f"{output_dir}/forecast.csv")
    
    # Evaluate on test data
    test_dates = test_df['datetime']
    forecast_test = forecast[forecast['ds'].isin(test_dates)]
    
    # Extract actual and predicted values
    y_true = test_df['block_median_fee_rate'].values
    y_pred = forecast_test['yhat'].values
    
    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Calculate MAPE (Mean Absolute Percentage Error) with handling division by zero
    # Use a small epsilon to avoid division by zero
    epsilon = 1e-10
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    
    # Save metrics
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape
    }
    
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f"{output_dir}/test_metrics.csv", index=False)
    
    # Print metrics
    print("\nTest Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot test predictions
    plt.figure(figsize=(15, 8))
    
    # Convert to proper datetime format for plotting
    train_dates = train_df['datetime']
    test_dates = test_df['datetime']
    forecast_dates = forecast_test['ds']
    
    # Plot training data
    plt.plot(train_dates, train_df['block_median_fee_rate'], 
             color='blue', label='Training Data')
    
    # Plot testing data
    plt.plot(test_dates, test_df['block_median_fee_rate'], 
             color='green', label='Testing Data (Actual)')
    
    # Plot predictions
    plt.plot(forecast_dates, forecast_test['yhat'], 
             color='red', linestyle='--', label='Prophet Forecast')
    
    # Plot prediction intervals
    plt.fill_between(forecast_dates, 
                     forecast_test['yhat_lower'], 
                     forecast_test['yhat_upper'], 
                     color='red', alpha=0.2, label='95% Confidence Interval')
    
    # Add labels and title
    plt.title('Bitcoin Fee Rate Forecast vs Actual (Test Period)')
    plt.xlabel('Date')
    plt.ylabel('Block Median Fee Rate (sat/vB)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig(f"{output_dir}/test_prediction.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Results saved to {output_dir}")
    
    # Try to run cross-validation
    try:
        print("Running cross-validation...")
        # Use prophet's cross_validation function correctly
        cv_results = cross_validation(
            model=model,
            horizon='24 hours',
            initial='30 days',
            period='7 days'
        )
        
        cv_metrics = performance_metrics(cv_results)
        cv_metrics.to_csv(f"{output_dir}/cv_metrics.csv")
        
        # Plot cross-validation metrics
        plt.figure(figsize=(16, 8))
        ax = plot_cross_validation_metric(cv_results, metric='rmse')
        plt.savefig(f"{output_dir}/cv_rmse.png", dpi=300, bbox_inches='tight')
        
        plt.figure(figsize=(16, 8))
        ax = plot_cross_validation_metric(cv_results, metric='mae')
        plt.savefig(f"{output_dir}/cv_mae.png", dpi=300, bbox_inches='tight')
        
        print("Cross-validation completed successfully.")
    except Exception as e:
        print(f"Error during cross-validation: {e}")
        print("Continuing with the rest of the process...")
    
    return forecast, metrics

if __name__ == "__main__":
    try:
        forecast, metrics = run_prophet_with_regressors()
        print("Prophet model with regressors completed!")
    except Exception as e:
        print(f"Error running Prophet model: {e}")
        import traceback
        traceback.print_exc()
