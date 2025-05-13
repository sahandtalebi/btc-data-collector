import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import os
from datetime import datetime
import json

class TimeSeriesDataset(Dataset):
    def __init__(self, data, context_length, prediction_length):
        """
        Dataset for DeepAR model
        
        Args:
            data: Time series data (numpy array)
            context_length: Length of context window
            prediction_length: Length of prediction window
        """
        self.data = data
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.n_samples = len(data) - context_length - prediction_length + 1
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        start_idx = idx
        end_idx = start_idx + self.context_length
        target_end_idx = end_idx + self.prediction_length
        
        context = self.data[start_idx:end_idx]
        target = self.data[end_idx:target_end_idx]
        
        return {'context': torch.FloatTensor(context),
                'target': torch.FloatTensor(target)}

class DeepARModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=40, num_layers=2, dropout=0.1):
        """
        DeepAR model based on LSTM
        
        Args:
            input_size: Dimension of input features
            hidden_size: Size of hidden state in LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(DeepARModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # Outputs mean and std for Gaussian distribution
        self.mean_layer = nn.Linear(hidden_size, 1)
        self.std_layer = nn.Linear(hidden_size, 1)
        self.softplus = nn.Softplus()
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_size]
        lstm_out, _ = self.lstm(x)
        
        # Get outputs for the last time step for forecasting
        means = self.mean_layer(lstm_out)
        stds = self.softplus(self.std_layer(lstm_out))
        
        return means, stds
    
    def loss_fn(self, means, stds, targets):
        """
        Gaussian Negative Log Likelihood loss
        
        Args:
            means: Predicted means
            stds: Predicted standard deviations
            targets: True values
            
        Returns:
            Negative log likelihood loss
        """
        # Gaussian NLL loss: -log(p(y|μ,σ))
        dist = torch.distributions.Normal(means, stds)
        log_probs = dist.log_prob(targets)
        loss = -log_probs.mean()
        return loss
    
    def sample_forecast(self, context, prediction_length, num_samples=100):
        """
        Generate samples from predictive distribution
        
        Args:
            context: Context window data
            prediction_length: Number of steps to predict
            num_samples: Number of samples to draw
            
        Returns:
            Samples from predictive distribution
        """
        self.eval()
        with torch.no_grad():
            # Initialize forecasts with samples
            forecasts = torch.zeros(num_samples, prediction_length)
            
            # Ensure context is a tensor with batch dimension
            if not isinstance(context, torch.Tensor):
                context = torch.FloatTensor(context)
            
            if len(context.shape) == 1:
                context = context.unsqueeze(0).unsqueeze(-1)  # Add batch and feature dimensions
            elif len(context.shape) == 2:
                context = context.unsqueeze(-1)  # Add feature dimension
            
            # For each prediction step
            current_input = context
            
            for t in range(prediction_length):
                # Get distribution parameters
                means, stds = self(current_input)
                
                # Get parameters for the next time step
                mean_t = means[:, -1, 0]
                std_t = stds[:, -1, 0]
                
                # Sample from distribution
                dist = torch.distributions.Normal(mean_t, std_t)
                samples = dist.sample((num_samples,))
                forecasts[:, t] = samples
                
                # Update input for next step
                next_input = samples.unsqueeze(1).unsqueeze(-1)  # [num_samples, 1, 1]
                
                # Concatenate with all but the first element of the current input
                # to maintain the context length
                new_input = []
                for i in range(num_samples):
                    sample_context = current_input[0, 1:, :].clone()  # Remove oldest element
                    sample_context = torch.cat([sample_context, next_input[i]], dim=0)  # Add new prediction
                    new_input.append(sample_context.unsqueeze(0))
                
                current_input = torch.cat(new_input, dim=0)
            
            return forecasts

class BitcoinFeeDeepARModel:
    def __init__(self, 
                 context_length=168, 
                 prediction_length=24, 
                 hidden_size=40,
                 num_layers=2,
                 dropout=0.1,
                 learning_rate=0.001,
                 batch_size=32,
                 num_epochs=50,
                 output_dir='models/deepar/output'):
        """
        Initialize DeepAR model for Bitcoin fee prediction
        
        Args:
            context_length: Length of context window (168 = 1 week of hourly data)
            prediction_length: Length of prediction window (24 = 1 day ahead)
            hidden_size: Size of hidden state in LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            learning_rate: Learning rate for optimization
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            output_dir: Directory to save outputs
        """
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.output_dir = output_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize model
        self.model = DeepARModel(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
    
    def prepare_data(self, df, target_col='block_median_fee_rate', train_ratio=0.8):
        """
        Prepare data for DeepAR model
        
        Args:
            df: DataFrame with datetime index and target column
            target_col: Column name of target variable (fees)
            train_ratio: Ratio of data to use for training
            
        Returns:
            train_loader: DataLoader for training
            val_loader: DataLoader for validation
            scaler: Scaler for data normalization
        """
        # Extract target variable
        if isinstance(df, pd.DataFrame):
            if target_col in df.columns:
                series = df[target_col].values
            else:
                series = df.iloc[:, 0].values
        else:
            series = df  # Assume numpy array or similar
        
        # Normalize data (simple min-max scaling)
        self.data_min = series.min()
        self.data_max = series.max()
        series_norm = (series - self.data_min) / (self.data_max - self.data_min)
        
        # Save scaling parameters
        scaling_params = {
            'data_min': float(self.data_min),
            'data_max': float(self.data_max)
        }
        with open(f"{self.output_dir}/scaling_params.json", 'w') as f:
            json.dump(scaling_params, f)
        
        # Split into training and validation sets
        train_size = int(len(series_norm) * train_ratio)
        train_data = series_norm[:train_size]
        val_data = series_norm[train_size - self.context_length:]  # Overlap for context
        
        # Create datasets
        train_dataset = TimeSeriesDataset(
            train_data, 
            self.context_length, 
            self.prediction_length
        )
        
        val_dataset = TimeSeriesDataset(
            val_data, 
            self.context_length, 
            self.prediction_length
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False
        )
        
        return train_loader, val_loader
    
    def train(self, train_loader, val_loader):
        """
        Train DeepAR model
        
        Args:
            train_loader: DataLoader for training
            val_loader: DataLoader for validation
            
        Returns:
            train_losses: List of training losses
            val_losses: List of validation losses
        """
        print(f"Training DeepAR model on {self.device}...")
        
        train_losses = []
        val_losses = []
        
        for epoch in range(self.num_epochs):
            # Training
            self.model.train()
            epoch_train_loss = 0
            
            for batch in train_loader:
                context = batch['context'].unsqueeze(-1).to(self.device)  # Add feature dimension
                target = batch['target'].unsqueeze(-1).to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                means, stds = self.model(context)
                
                # Compute loss and backpropagate
                loss = self.model.loss_fn(means, stds, target)
                loss.backward()
                self.optimizer.step()
                
                epoch_train_loss += loss.item()
            
            epoch_train_loss /= len(train_loader)
            train_losses.append(epoch_train_loss)
            
            # Validation
            self.model.eval()
            epoch_val_loss = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    context = batch['context'].unsqueeze(-1).to(self.device)
                    target = batch['target'].unsqueeze(-1).to(self.device)
                    
                    means, stds = self.model(context)
                    loss = self.model.loss_fn(means, stds, target)
                    
                    epoch_val_loss += loss.item()
            
            epoch_val_loss /= len(val_loader)
            val_losses.append(epoch_val_loss)
            
            print(f"Epoch {epoch+1}/{self.num_epochs}, "
                  f"Train Loss: {epoch_train_loss:.4f}, "
                  f"Val Loss: {epoch_val_loss:.4f}")
        
        # Save model
        torch.save(self.model.state_dict(), f"{self.output_dir}/deepar_model.pth")
        
        # Plot losses
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('DeepAR Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{self.output_dir}/training_loss.png", dpi=300, bbox_inches='tight')
        
        return train_losses, val_losses
    
    def predict(self, context_data, prediction_length=None, num_samples=100):
        """
        Generate probabilistic forecasts
        
        Args:
            context_data: Context window data
            prediction_length: Length of prediction window (default: self.prediction_length)
            num_samples: Number of samples to draw
            
        Returns:
            forecasts: Samples from predictive distribution
        """
        if prediction_length is None:
            prediction_length = self.prediction_length
            
        self.model.eval()
        
        # Normalize context data
        context_norm = (context_data - self.data_min) / (self.data_max - self.data_min)
        
        # Generate samples
        with torch.no_grad():
            forecast_norm = self.model.sample_forecast(
                context_norm, 
                prediction_length, 
                num_samples
            ).numpy()
        
        # Denormalize forecasts
        forecasts = forecast_norm * (self.data_max - self.data_min) + self.data_min
        
        return forecasts
    
    def evaluate(self, test_data, context_length=None, prediction_length=None, num_samples=100):
        """
        Evaluate model on test data
        
        Args:
            test_data: Test data
            context_length: Length of context window (default: self.context_length)
            prediction_length: Length of prediction window (default: self.prediction_length)
            num_samples: Number of samples to draw
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        if context_length is None:
            context_length = self.context_length
        
        if prediction_length is None:
            prediction_length = self.prediction_length
        
        # Use last 'context_length' points for prediction
        context = test_data[-context_length - prediction_length:-prediction_length]
        actuals = test_data[-prediction_length:]
        
        # Generate forecasts
        forecasts = self.predict(context, prediction_length, num_samples)
        
        # Calculate metrics
        point_forecasts = forecasts.mean(axis=0)
        mse = np.mean((point_forecasts - actuals)**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(point_forecasts - actuals))
        
        # Calculate quantiles for evaluation
        q10 = np.percentile(forecasts, 10, axis=0)
        q50 = np.percentile(forecasts, 50, axis=0)  # Median
        q90 = np.percentile(forecasts, 90, axis=0)
        
        # Check quantile coverage
        coverage_90 = np.mean((actuals >= q10) & (actuals <= q90))
        
        # Plot forecast with uncertainty intervals
        plt.figure(figsize=(12, 6))
        x = np.arange(len(context) + len(actuals))
        plt.plot(x[:len(context)], context, 'b-', label='Context')
        plt.plot(x[len(context):], actuals, 'k-', label='Actual')
        plt.plot(x[len(context):], point_forecasts, 'r-', label='Mean Forecast')
        plt.fill_between(x[len(context):], q10, q90, color='r', alpha=0.2, label='80% Prediction Interval')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.title('DeepAR Probabilistic Forecast')
        plt.savefig(f"{self.output_dir}/forecast_evaluation.png", dpi=300, bbox_inches='tight')
        
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'coverage_90': coverage_90
        }
        
        # Save metrics
        with open(f"{self.output_dir}/evaluation_metrics.json", 'w') as f:
            json.dump({k: float(v) for k, v in metrics.items()}, f)
        
        return metrics
    
    def run_pipeline(self, df, target_col='block_median_fee_rate'):
        """
        Run the entire DeepAR modeling pipeline
        
        Args:
            df: DataFrame with datetime index and target column
            target_col: Column name to predict
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        # Prepare data
        train_loader, val_loader = self.prepare_data(df, target_col)
        
        # Train model
        self.train(train_loader, val_loader)
        
        # Evaluate on the last segment of data
        if isinstance(df, pd.DataFrame):
            test_data = df[target_col].values
        else:
            test_data = df
        
        metrics = self.evaluate(test_data)
        
        return metrics

if __name__ == "__main__":
    # Load cleaned data
    df = pd.read_csv('data/bitcoin_data_cleaned_no_resample_original.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    
    # Resample to hourly for simplicity
    df_hourly = df['block_median_fee_rate'].resample('1H').mean().fillna(method='ffill')
    df_hourly = pd.DataFrame(df_hourly)
    
    # Initialize and run DeepAR model
    deepar_model = BitcoinFeeDeepARModel(
        context_length=168,  # 1 week
        prediction_length=24,  # 1 day
        hidden_size=40,
        num_layers=2,
        dropout=0.1,
        learning_rate=0.001,
        batch_size=32,
        num_epochs=50  # Reduced for demonstration
    )
    
    metrics = deepar_model.run_pipeline(df_hourly)
    
    print("DeepAR model training completed!")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"90% Coverage: {metrics['coverage_90']:.4f}") 