import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

class TimeSeriesDataset(Dataset):
    def __init__(self, data, context_length, prediction_length):
        """
        Dataset for time series forecasting
        
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

class NBEATSBlock(nn.Module):
    def __init__(self, input_size, theta_size, hidden_size, stack_type='generic'):
        """
        N-BEATS block
        
        Args:
            input_size: Size of input sequence
            theta_size: Size of theta output (input_size for backcast, prediction_length for forecast)
            hidden_size: Size of hidden layers
            stack_type: Type of stack ('generic', 'trend', or 'seasonality')
        """
        super(NBEATSBlock, self).__init__()
        
        self.input_size = input_size
        self.theta_size = theta_size
        self.hidden_size = hidden_size
        self.stack_type = stack_type
        
        # Fully connected stack
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        
        # Output layers
        self.theta_b = nn.Linear(hidden_size, theta_size)  # Backcast
        self.theta_f = nn.Linear(hidden_size, theta_size)  # Forecast
        
        # Activation function
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Forward pass through fully connected layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        
        # Extract backcast and forecast parameters
        theta_b = self.theta_b(x)
        theta_f = self.theta_f(x)
        
        # Apply basis function
        if self.stack_type == 'generic':
            # Generic stack: theta is directly used
            backcast = theta_b
            forecast = theta_f
        else:
            # For interpretable stacks, we would implement trend and seasonality basis functions
            # Simplified implementation for demonstration
            backcast = theta_b
            forecast = theta_f
        
        return backcast, forecast

class NBEATS(nn.Module):
    def __init__(self, input_size, prediction_length, hidden_size=128, stacks=2, blocks_per_stack=3):
        """
        N-BEATS model
        
        Args:
            input_size: Size of input sequence
            prediction_length: Length of prediction window
            hidden_size: Size of hidden layers
            stacks: Number of stacks
            blocks_per_stack: Number of blocks per stack
        """
        super(NBEATS, self).__init__()
        
        self.input_size = input_size
        self.prediction_length = prediction_length
        self.hidden_size = hidden_size
        self.stacks = stacks
        self.blocks_per_stack = blocks_per_stack
        
        # Create stacks and blocks
        self.blocks = nn.ModuleList()
        
        for i in range(stacks):
            for j in range(blocks_per_stack):
                # First half stacks are generic, second half are interpretable
                if i < stacks // 2:
                    block_type = 'generic'
                else:
                    block_type = 'trend' if j % 2 == 0 else 'seasonality'
                
                block = NBEATSBlock(
                    input_size=input_size,
                    theta_size=max(input_size, prediction_length),  # For simplicity
                    hidden_size=hidden_size,
                    stack_type=block_type
                )
                
                self.blocks.append(block)
    
    def forward(self, x):
        # Initial backcast and forecast
        residuals = x.clone()
        forecast = torch.zeros(x.size(0), self.prediction_length, device=x.device)
        
        # Apply each block sequentially
        for block in self.blocks:
            # Pass residuals through block
            backcast, block_forecast = block(residuals)
            
            # Update residuals and forecast
            residuals = residuals - backcast
            forecast = forecast + block_forecast[:, :self.prediction_length]
        
        return forecast

class BitcoinFeeNBEATSModel:
    def __init__(self, 
                 context_length=168,
                 prediction_length=24,
                 hidden_size=128,
                 stacks=2,
                 blocks_per_stack=3, 
                 learning_rate=0.001,
                 batch_size=32,
                 num_epochs=50,
                 output_dir='models/deep_learning/nbeats_output'):
        """
        Initialize N-BEATS model for Bitcoin fee prediction
        
        Args:
            context_length: Length of context window (168 = 1 week of hourly data)
            prediction_length: Length of prediction window (24 = 1 day ahead)
            hidden_size: Size of hidden layers
            stacks: Number of stacks
            blocks_per_stack: Number of blocks per stack
            learning_rate: Learning rate for optimization
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            output_dir: Directory to save outputs
        """
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.hidden_size = hidden_size
        self.stacks = stacks
        self.blocks_per_stack = blocks_per_stack
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.output_dir = output_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize model
        self.model = NBEATS(
            input_size=context_length,
            prediction_length=prediction_length,
            hidden_size=hidden_size,
            stacks=stacks,
            blocks_per_stack=blocks_per_stack
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Loss function
        self.loss_fn = nn.MSELoss()
    
    def prepare_data(self, df, target_col='block_median_fee_rate', train_ratio=0.8):
        """
        Prepare data for N-BEATS model
        
        Args:
            df: DataFrame with datetime index and target column
            target_col: Column name of target variable (fees)
            train_ratio: Ratio of data to use for training
            
        Returns:
            train_loader: DataLoader for training
            val_loader: DataLoader for validation
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
        Train N-BEATS model
        
        Args:
            train_loader: DataLoader for training
            val_loader: DataLoader for validation
            
        Returns:
            train_losses: List of training losses
            val_losses: List of validation losses
        """
        print(f"Training N-BEATS model on {self.device}...")
        
        train_losses = []
        val_losses = []
        
        for epoch in range(self.num_epochs):
            # Training
            self.model.train()
            epoch_train_loss = 0
            
            for batch in train_loader:
                context = batch['context'].to(self.device)
                target = batch['target'].to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                forecast = self.model(context)
                
                # Compute loss and backpropagate
                loss = self.loss_fn(forecast, target)
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
                    context = batch['context'].to(self.device)
                    target = batch['target'].to(self.device)
                    
                    forecast = self.model(context)
                    loss = self.loss_fn(forecast, target)
                    
                    epoch_val_loss += loss.item()
            
            epoch_val_loss /= len(val_loader)
            val_losses.append(epoch_val_loss)
            
            print(f"Epoch {epoch+1}/{self.num_epochs}, "
                  f"Train Loss: {epoch_train_loss:.4f}, "
                  f"Val Loss: {epoch_val_loss:.4f}")
        
        # Save model
        torch.save(self.model.state_dict(), f"{self.output_dir}/nbeats_model.pth")
        
        # Plot losses
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('N-BEATS Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{self.output_dir}/training_loss.png", dpi=300, bbox_inches='tight')
        
        return train_losses, val_losses
    
    def predict(self, context):
        """
        Generate forecasts
        
        Args:
            context: Context window data
            
        Returns:
            forecasts: Predicted values
        """
        self.model.eval()
        
        # Normalize context data
        context_norm = (context - self.data_min) / (self.data_max - self.data_min)
        
        # Convert to tensor
        if not isinstance(context_norm, torch.Tensor):
            context_tensor = torch.FloatTensor(context_norm).unsqueeze(0).to(self.device)
        else:
            context_tensor = context_norm.unsqueeze(0).to(self.device)
        
        # Generate forecast
        with torch.no_grad():
            forecast_norm = self.model(context_tensor).cpu().numpy()[0]
        
        # Denormalize forecast
        forecast = forecast_norm * (self.data_max - self.data_min) + self.data_min
        
        return forecast
    
    def evaluate(self, test_data):
        """
        Evaluate model on test data
        
        Args:
            test_data: Test data
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        # Use last 'context_length' points for prediction
        context = test_data[-self.context_length - self.prediction_length:-self.prediction_length]
        actuals = test_data[-self.prediction_length:]
        
        # Generate forecast
        forecast = self.predict(context)
        
        # Calculate metrics
        mse = np.mean((forecast - actuals)**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(forecast - actuals))
        
        # Plot forecast
        plt.figure(figsize=(12, 6))
        x = np.arange(len(context) + len(actuals))
        plt.plot(x[:len(context)], context, 'b-', label='Context')
        plt.plot(x[len(context):], actuals, 'k-', label='Actual')
        plt.plot(x[len(context):], forecast, 'r-', label='Forecast')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.title('N-BEATS Forecast')
        plt.savefig(f"{self.output_dir}/forecast_evaluation.png", dpi=300, bbox_inches='tight')
        
        metrics = {
            'rmse': rmse,
            'mae': mae
        }
        
        # Save metrics
        with open(f"{self.output_dir}/evaluation_metrics.json", 'w') as f:
            json.dump({k: float(v) for k, v in metrics.items()}, f)
        
        return metrics
    
    def run_pipeline(self, df, target_col='block_median_fee_rate'):
        """
        Run the entire N-BEATS modeling pipeline
        
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
    
    # Initialize and run N-BEATS model
    nbeats_model = BitcoinFeeNBEATSModel(
        context_length=168,  # 1 week
        prediction_length=24,  # 1 day
        hidden_size=128,
        stacks=2,
        blocks_per_stack=3,
        learning_rate=0.001,
        batch_size=32,
        num_epochs=30  # Reduced for demonstration
    )
    
    metrics = nbeats_model.run_pipeline(df_hourly)
    
    print("N-BEATS model training completed!")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}") 