import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
import os

class BitcoinFeeProphetModel:
    def __init__(self, 
                 changepoint_prior_scale=0.05, 
                 seasonality_prior_scale=10, 
                 output_dir='models/prophet/output'):
        """
        Initialize Prophet model for Bitcoin fee prediction
        
        Args:
            changepoint_prior_scale: Controls flexibility of trend
            seasonality_prior_scale: Controls flexibility of seasonality
            output_dir: Directory to save outputs
        """
        self.model = Prophet(
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            daily_seasonality=True,
            weekly_seasonality=True
        )
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def prepare_data(self, df, target_col='block_median_fee_rate'):
        """
        Prepare data for Prophet model
        
        Args:
            df: DataFrame with datetime index and target column
            target_col: Column name of target variable (fees)
            
        Returns:
            DataFrame with 'ds' (dates) and 'y' (target values) columns
        """
        # Prophet requires columns named 'ds' and 'y'
        prophet_df = pd.DataFrame()
        prophet_df['ds'] = df.index
        prophet_df['y'] = df[target_col]
        return prophet_df
    
    def fit(self, df):
        """
        Fit Prophet model
        
        Args:
            df: DataFrame with 'ds' and 'y' columns
        """
        print("Fitting Prophet model...")
        self.model.fit(df)
        
    def predict(self, periods=24, freq='H'):
        """
        Generate predictions
        
        Args:
            periods: Number of periods to predict
            freq: Frequency of predictions ('H' for hourly)
            
        Returns:
            DataFrame with predictions
        """
        print(f"Predicting {periods} periods ahead...")
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        forecast = self.model.predict(future)
        return forecast
    
    def evaluate(self, df, initial='30 days', period='7 days', horizon='24 hours'):
        """
        Evaluate model with cross-validation
        
        Args:
            df: DataFrame with 'ds' and 'y' columns
            initial: Initial training period
            period: Period between cutoffs
            horizon: Forecast horizon
            
        Returns:
            DataFrame with cross-validation metrics
        """
        print("Performing cross-validation...")
        cv_results = cross_validation(
            self.model, 
            df, 
            initial=initial, 
            period=period, 
            horizon=horizon
        )
        
        cv_metrics = performance_metrics(cv_results)
        
        # Save metrics
        cv_metrics.to_csv(f"{self.output_dir}/cv_metrics.csv")
        
        return cv_results, cv_metrics
    
    def plot_components(self, forecast):
        """
        Plot forecast components
        
        Args:
            forecast: Forecast DataFrame from predict()
        """
        print("Plotting forecast components...")
        fig1 = self.model.plot_components(forecast)
        fig1.savefig(f"{self.output_dir}/components.png", dpi=300, bbox_inches='tight')
        
        # Plot forecast
        fig2 = self.model.plot(forecast)
        fig2.savefig(f"{self.output_dir}/forecast.png", dpi=300, bbox_inches='tight')
        
    def plot_cv_metrics(self, cv_results):
        """
        Plot cross-validation metrics
        
        Args:
            cv_results: Results from cross_validation()
        """
        print("Plotting cross-validation metrics...")
        fig = plt.figure(figsize=(16, 8))
        ax = plot_cross_validation_metric(cv_results, metric='rmse')
        fig.savefig(f"{self.output_dir}/cv_rmse.png", dpi=300, bbox_inches='tight')
        
        fig = plt.figure(figsize=(16, 8))
        ax = plot_cross_validation_metric(cv_results, metric='mae')
        fig.savefig(f"{self.output_dir}/cv_mae.png", dpi=300, bbox_inches='tight')
        
    def run_pipeline(self, df, target_col='block_median_fee_rate', prediction_periods=48):
        """
        Run the entire Prophet modeling pipeline
        
        Args:
            df: DataFrame with datetime index and target column
            target_col: Column name to predict
            prediction_periods: Number of periods to predict
            
        Returns:
            forecast: Forecast DataFrame
            cv_metrics: Cross-validation metrics
        """
        # Prepare data
        prophet_df = self.prepare_data(df, target_col)
        
        # Fit model
        self.fit(prophet_df)
        
        # Make predictions
        forecast = self.predict(periods=prediction_periods)
        
        # Plot components
        self.plot_components(forecast)
        
        # Evaluate
        cv_results, cv_metrics = self.evaluate(prophet_df)
        
        # Plot CV metrics
        self.plot_cv_metrics(cv_results)
        
        return forecast, cv_metrics

if __name__ == "__main__":
    # Load cleaned data
    df = pd.read_csv('data/bitcoin_data_cleaned_no_resample_original.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    
    # Resample to hourly if needed
    df_hourly = df['block_median_fee_rate'].resample('1H').mean().fillna(method='ffill')
    df_hourly = pd.DataFrame(df_hourly)
    
    # Initialize and run Prophet model
    prophet_model = BitcoinFeeProphetModel(
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10
    )
    
    forecast, cv_metrics = prophet_model.run_pipeline(df_hourly)
    
    print("Prophet model training completed!")
    print(f"RMSE: {cv_metrics['rmse'].mean():.4f}")
    print(f"MAE: {cv_metrics['mae'].mean():.4f}") 