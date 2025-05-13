import pandas as pd
import argparse
import os
import json
from datetime import datetime

# Import model classes from their respective modules
import sys
sys.path.append('.')  # Add current directory to path

from models.prophet.prophet_model import BitcoinFeeProphetModel
from models.deepar.deepar_model import BitcoinFeeDeepARModel
from models.deep_learning.nbeats_model import BitcoinFeeNBEATSModel
from models.matrix_profile.matrix_profile_model import BitcoinFeeMatrixProfile

def load_data(file_path, resample=True):
    """
    Load and prepare data for modeling
    
    Args:
        file_path: Path to data file
        resample: Whether to resample to hourly frequency
        
    Returns:
        DataFrame with datetime index
    """
    print(f"Loading data from {file_path}...")
    
    # Load data
    df = pd.read_csv(file_path)
    
    # Convert datetime and set as index
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    
    # Resample to hourly if requested
    if resample:
        print("Resampling to hourly frequency...")
        df_hourly = df['block_median_fee_rate'].resample('1H').mean().fillna(method='ffill')
        df_hourly = pd.DataFrame(df_hourly)
        return df_hourly
    
    return df

def run_prophet_model(df, prediction_periods=48):
    """
    Run Prophet model
    
    Args:
        df: DataFrame with datetime index
        prediction_periods: Number of periods to predict
        
    Returns:
        Dictionary with model results
    """
    print("\n=== Running Prophet Model ===")
    
    # Initialize Prophet model
    prophet_model = BitcoinFeeProphetModel(
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10,
        output_dir='models/prophet/output'
    )
    
    # Run full pipeline
    forecast, cv_metrics = prophet_model.run_pipeline(
        df, 
        target_col='block_median_fee_rate',
        prediction_periods=prediction_periods
    )
    
    print(f"Prophet model completed with RMSE: {cv_metrics['rmse'].mean():.4f}")
    
    return {
        'model_type': 'prophet',
        'output_dir': prophet_model.output_dir,
        'metrics': {
            'rmse': float(cv_metrics['rmse'].mean()),
            'mae': float(cv_metrics['mae'].mean())
        }
    }

def run_deepar_model(df):
    """
    Run DeepAR model
    
    Args:
        df: DataFrame with datetime index
        
    Returns:
        Dictionary with model results
    """
    print("\n=== Running DeepAR Model ===")
    
    # Initialize DeepAR model
    deepar_model = BitcoinFeeDeepARModel(
        context_length=168,  # 1 week
        prediction_length=24,  # 1 day
        hidden_size=40,
        num_layers=2,
        dropout=0.1,
        learning_rate=0.001,
        batch_size=32,
        num_epochs=30,  # Reduced for demonstration
        output_dir='models/deepar/output'
    )
    
    # Run full pipeline
    metrics = deepar_model.run_pipeline(df, target_col='block_median_fee_rate')
    
    print(f"DeepAR model completed with RMSE: {metrics['rmse']:.4f}")
    
    return {
        'model_type': 'deepar',
        'output_dir': deepar_model.output_dir,
        'metrics': metrics
    }

def run_nbeats_model(df):
    """
    Run N-BEATS model
    
    Args:
        df: DataFrame with datetime index
        
    Returns:
        Dictionary with model results
    """
    print("\n=== Running N-BEATS Model ===")
    
    # Initialize N-BEATS model
    nbeats_model = BitcoinFeeNBEATSModel(
        context_length=168,  # 1 week
        prediction_length=24,  # 1 day
        hidden_size=128,
        stacks=2,
        blocks_per_stack=3,
        learning_rate=0.001,
        batch_size=32,
        num_epochs=20,  # Reduced for demonstration
        output_dir='models/deep_learning/nbeats_output'
    )
    
    # Run full pipeline
    metrics = nbeats_model.run_pipeline(df, target_col='block_median_fee_rate')
    
    print(f"N-BEATS model completed with RMSE: {metrics['rmse']:.4f}")
    
    return {
        'model_type': 'nbeats',
        'output_dir': nbeats_model.output_dir,
        'metrics': metrics
    }

def run_matrix_profile_analysis(df):
    """
    Run Matrix Profile analysis
    
    Args:
        df: DataFrame with datetime index
        
    Returns:
        Dictionary with analysis results
    """
    print("\n=== Running Matrix Profile Analysis ===")
    
    # Initialize Matrix Profile analyzer
    mp_analyzer = BitcoinFeeMatrixProfile(
        window_size=24,  # 24 hours = daily patterns
        output_dir='models/matrix_profile/output'
    )
    
    # Run full pipeline
    results = mp_analyzer.run_pipeline(
        df,
        target_col='block_median_fee_rate',
        n_motifs=3,
        n_discords=3
    )
    
    print(f"Matrix Profile analysis completed with {len(results['motifs'])} motifs identified")
    
    return {
        'model_type': 'matrix_profile',
        'output_dir': mp_analyzer.output_dir,
        'results': {
            'n_motifs': len(results['motifs']),
            'n_discords': len(results['discords'])
        }
    }

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Bitcoin fee prediction models')
    
    parser.add_argument('--data', type=str, default='data/bitcoin_data_cleaned_no_resample_original.csv',
                        help='Path to cleaned data file')
    
    parser.add_argument('--resample', action='store_true', default=True,
                        help='Resample data to hourly frequency')
    
    parser.add_argument('--models', type=str, nargs='+', 
                        choices=['prophet', 'deepar', 'nbeats', 'matrix_profile', 'all'],
                        default=['all'],
                        help='Which models to run')
    
    args = parser.parse_args()
    
    # Load data
    df = load_data(args.data, resample=args.resample)
    
    # Determine which models to run
    models_to_run = args.models
    if 'all' in models_to_run:
        models_to_run = ['prophet', 'deepar', 'nbeats', 'matrix_profile']
    
    results = []
    
    # Run selected models
    if 'prophet' in models_to_run:
        try:
            prophet_results = run_prophet_model(df)
            results.append(prophet_results)
        except Exception as e:
            print(f"Error running Prophet model: {e}")
    
    if 'deepar' in models_to_run:
        try:
            deepar_results = run_deepar_model(df)
            results.append(deepar_results)
        except Exception as e:
            print(f"Error running DeepAR model: {e}")
    
    if 'nbeats' in models_to_run:
        try:
            nbeats_results = run_nbeats_model(df)
            results.append(nbeats_results)
        except Exception as e:
            print(f"Error running N-BEATS model: {e}")
    
    if 'matrix_profile' in models_to_run:
        try:
            mp_results = run_matrix_profile_analysis(df)
            results.append(mp_results)
        except Exception as e:
            print(f"Error running Matrix Profile analysis: {e}")
    
    # Save overall results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    overall_results = {
        'timestamp': timestamp,
        'data_file': args.data,
        'resampled': args.resample,
        'models_run': models_to_run,
        'results': results
    }
    
    with open(f'models/results_{timestamp}.json', 'w') as f:
        json.dump(overall_results, f, indent=2)
    
    print(f"\nAll models completed. Results saved to models/results_{timestamp}.json")

if __name__ == "__main__":
    main() 