import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import stumpy
from datetime import datetime, timedelta

class BitcoinFeeMultivariateMatrixProfile:
    def __init__(self, 
                 window_size=24,  # 24 hours = daily patterns
                 output_dir='models/matrix_profile/output_multivariate'):
        """
        Initialize Multivariate Matrix Profile for Bitcoin fee analysis
        
        Args:
            window_size: Size of sliding window for pattern mining (24 = daily patterns)
            output_dir: Directory to save outputs
        """
        self.window_size = window_size
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def compute_matrix_profile(self, df, target_cols):
        """
        Compute Multivariate Matrix Profile for time series
        
        Args:
            df: DataFrame with datetime index and target columns
            target_cols: List of column names to include in analysis
            
        Returns:
            mp: Matrix Profile
            mp_indices: Matrix Profile indices
        """
        print(f"Computing Multivariate Matrix Profile with window size {self.window_size} for {len(target_cols)} variables...")
        
        # Extract target variables
        multivariate_series = []
        for col in target_cols:
            if col in df.columns:
                # Normalize the data to make all variables comparable
                series = df[col].values
                series = (series - np.mean(series)) / np.std(series)
                multivariate_series.append(series)
        
        # Convert to required format for mstump (m x n where m is number of dimensions and n is length)
        multivariate_series = np.array(multivariate_series)
        
        # Compute Multivariate Matrix Profile
        mp = stumpy.mstump(multivariate_series, self.window_size)
        
        # Extract profile and indices
        mp_profile = mp[0][:, 0]  # distance profile (first element is the matrix profile)
        mp_indices = mp[1][:, 0]  # matrix profile indices (second element is the profile indices)
        
        # Save raw matrix profile
        np.save(f"{self.output_dir}/multivariate_matrix_profile.npy", mp_profile)
        np.save(f"{self.output_dir}/multivariate_matrix_profile_indices.npy", mp_indices)
        
        # Plot Matrix Profile
        plt.figure(figsize=(15, 10))
        
        plt.subplot(len(target_cols) + 1, 1, 1)
        plt.plot(mp_profile)
        plt.title('Multivariate Matrix Profile')
        plt.ylabel('Distance')
        plt.grid(True, alpha=0.3)
        
        # Plot each time series
        for i, col in enumerate(target_cols):
            plt.subplot(len(target_cols) + 1, 1, i + 2)
            plt.plot(multivariate_series[i])
            plt.title(f'Time Series: {col}')
            plt.ylabel('Normalized Value')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/multivariate_matrix_profile.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        return mp, mp_profile, mp_indices
    
    def find_motifs(self, df, mp, mp_indices, n_motifs=3, target_cols=None):
        """
        Find motifs (recurring patterns) in multivariate time series
        
        Args:
            df: DataFrame with datetime index
            mp: Matrix Profile from mstump
            mp_indices: Matrix Profile indices
            n_motifs: Number of motifs to find
            target_cols: List of column names to include in analysis
            
        Returns:
            motifs: List of motif indices
        """
        print(f"Finding top {n_motifs} multivariate motifs...")
        
        # Extract time series for plotting
        timestamps = df.index
        multivariate_series = []
        for col in target_cols:
            if col in df.columns:
                # Normalize the data for consistent visualization
                series = df[col].values
                series = (series - np.mean(series)) / np.std(series)
                multivariate_series.append(series)
        
        multivariate_series = np.array(multivariate_series)
        
        # Get the matrix profile values (1st dimension)
        mp_profile = mp[0][:, 0]
        
        # Find motif indices using the matrix profile
        motif_indices = []
        exclusion_zone = self.window_size // 2
        
        # Get the top-n motifs
        for i in range(n_motifs):
            # Find the lowest matrix profile value that's not been found yet
            if i == 0:
                # First motif is the lowest matrix profile value
                motif_idx = np.argmin(mp_profile)
                motif_neighbor_idx = int(mp_indices[motif_idx])
            else:
                # Make a copy of the profile to avoid modifying the original
                mp_profile_copy = mp_profile.copy()
                
                # Set already found motifs to inf
                for idx, _ in motif_indices:
                    # Exclude a zone around this motif
                    start_idx = max(0, idx - exclusion_zone)
                    end_idx = min(len(mp_profile_copy), idx + exclusion_zone)
                    mp_profile_copy[start_idx:end_idx] = np.inf
                
                # Find the next motif
                motif_idx = np.argmin(mp_profile_copy)
                motif_neighbor_idx = int(mp_indices[motif_idx])
            
            # Store motif and its nearest neighbor
            motif_indices.append((motif_idx, motif_neighbor_idx))
        
        # Plot motifs for each dimension
        motif_data = []
        
        for i, (idx1, idx2) in enumerate(motif_indices):
            # Create figure for this motif
            fig, axs = plt.subplots(len(target_cols), 1, figsize=(15, 4*len(target_cols)))
            
            if len(target_cols) == 1:
                axs = [axs]  # Make it iterable for single variable case
            
            # Plot each dimension
            for dim, (col, ax) in enumerate(zip(target_cols, axs)):
                # Plot the motif and its match
                ax.plot(range(self.window_size), 
                        multivariate_series[dim, idx1:idx1 + self.window_size], 
                        label=f'Motif {i+1}', color='blue', linewidth=2)
                ax.plot(range(self.window_size), 
                        multivariate_series[dim, idx2:idx2 + self.window_size], 
                        label=f'Match {i+1}', color='red', linewidth=2, linestyle='--')
                ax.set_title(f'{col} - Motif {i+1}')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/motif_{i+1}_detail.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Store motif data
            try:
                motif1_start = timestamps[idx1]
                motif1_end = timestamps[idx1 + self.window_size - 1]
                motif2_start = timestamps[idx2]
                motif2_end = timestamps[idx2 + self.window_size - 1]
                
                motif_data.append({
                    'motif_id': i + 1,
                    'occurrence1': {
                        'start_index': int(idx1),
                        'end_index': int(idx1 + self.window_size - 1),
                        'start_time': str(motif1_start),
                        'end_time': str(motif1_end)
                    },
                    'occurrence2': {
                        'start_index': int(idx2),
                        'end_index': int(idx2 + self.window_size - 1),
                        'start_time': str(motif2_start),
                        'end_time': str(motif2_end)
                    }
                })
            except Exception as e:
                print(f"Error saving timestamp data: {e}")
                # Fallback if timestamps are not available
                motif_data.append({
                    'motif_id': i + 1,
                    'occurrence1': {
                        'start_index': int(idx1),
                        'end_index': int(idx1 + self.window_size - 1)
                    },
                    'occurrence2': {
                        'start_index': int(idx2),
                        'end_index': int(idx2 + self.window_size - 1)
                    }
                })
        
        # Save motif data
        with open(f"{self.output_dir}/motifs.json", 'w') as f:
            json.dump(motif_data, f, indent=2)
        
        # Also create an overview figure showing all series with all motifs
        plt.figure(figsize=(20, 4*len(target_cols)))
        
        for dim, col in enumerate(target_cols):
            plt.subplot(len(target_cols), 1, dim+1)
            
            # Plot the full time series
            plt.plot(multivariate_series[dim], alpha=0.5, color='gray')
            
            # Plot each motif with different colors
            colors = ['red', 'blue', 'green', 'purple', 'orange']
            
            for i, (idx1, idx2) in enumerate(motif_indices):
                color = colors[i % len(colors)]
                
                # Plot motif and its nearest neighbor
                plt.plot(range(idx1, idx1 + self.window_size), 
                        multivariate_series[dim, idx1:idx1 + self.window_size], 
                        color=color, linewidth=2)
                
                plt.plot(range(idx2, idx2 + self.window_size), 
                        multivariate_series[dim, idx2:idx2 + self.window_size], 
                        color=color, linewidth=2, linestyle='--')
            
            plt.title(f'{col} with {n_motifs} Motifs')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/all_motifs_overview.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        return motif_indices, motif_data
    
    def find_discords(self, mp_profile, n_discords=3):
        """
        Find discords (anomalies) in multivariate time series
        
        Args:
            mp_profile: Matrix Profile from 1st dimension
            n_discords: Number of discords to find
            
        Returns:
            discords: List of discord indices
        """
        print(f"Finding top {n_discords} multivariate discords (anomalies)...")
        
        # Copy the profile to avoid modifying the original
        mp_copy = mp_profile.copy()
        
        # Find discord indices (highest matrix profile values)
        discord_indices = []
        exclusion_zone = self.window_size // 2
        
        for i in range(n_discords):
            # Find index of highest value in matrix profile
            discord_idx = np.argmax(mp_copy)
            discord_indices.append(discord_idx)
            
            # Exclude a zone around this discord for next iteration
            exclusion_start = max(0, discord_idx - exclusion_zone)
            exclusion_end = min(len(mp_copy), discord_idx + exclusion_zone)
            mp_copy[exclusion_start:exclusion_end] = -np.inf
        
        # Save discord indices
        with open(f"{self.output_dir}/discords.json", 'w') as f:
            json.dump([int(idx) for idx in discord_indices], f)
        
        return discord_indices
    
    def visualize_discords(self, df, discord_indices, target_cols):
        """
        Visualize discords (anomalies) in multivariate time series
        
        Args:
            df: DataFrame with datetime index
            discord_indices: List of discord indices
            target_cols: List of column names to visualize
        """
        print("Visualizing multivariate discords...")
        
        # Extract timestamps and time series
        timestamps = df.index
        multivariate_series = []
        for col in target_cols:
            if col in df.columns:
                # Normalize the data for consistent visualization
                series = df[col].values
                series = (series - np.mean(series)) / np.std(series)
                multivariate_series.append(series)
        
        multivariate_series = np.array(multivariate_series)
        
        # Plot discords for each dimension
        discord_data = []
        
        for i, idx in enumerate(discord_indices):
            # Create figure for this discord
            fig, axs = plt.subplots(len(target_cols), 1, figsize=(12, 4*len(target_cols)))
            
            if len(target_cols) == 1:
                axs = [axs]  # Make it iterable for single variable case
            
            # Plot each dimension
            for dim, (col, ax) in enumerate(zip(target_cols, axs)):
                # Plot the discord
                ax.plot(range(self.window_size), 
                        multivariate_series[dim, idx:idx + self.window_size], 
                        color='red', linewidth=2)
                ax.set_title(f'{col} - Discord {i+1}')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/discord_{i+1}_detail.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Store discord data
            try:
                discord_start = timestamps[idx]
                discord_end = timestamps[idx + self.window_size - 1]
                
                discord_data.append({
                    'discord_id': i + 1,
                    'start_index': int(idx),
                    'end_index': int(idx + self.window_size - 1),
                    'start_time': str(discord_start),
                    'end_time': str(discord_end)
                })
            except Exception as e:
                print(f"Error saving timestamp data: {e}")
                # Fallback if timestamps are not available
                discord_data.append({
                    'discord_id': i + 1,
                    'start_index': int(idx),
                    'end_index': int(idx + self.window_size - 1)
                })
        
        # Save discord data
        with open(f"{self.output_dir}/discord_details.json", 'w') as f:
            json.dump(discord_data, f, indent=2)
        
        # Also create an overview figure showing all series with all discords
        plt.figure(figsize=(20, 4*len(target_cols)))
        
        for dim, col in enumerate(target_cols):
            plt.subplot(len(target_cols), 1, dim+1)
            
            # Plot the full time series
            plt.plot(multivariate_series[dim], alpha=0.5, color='gray')
            
            # Plot each discord
            colors = ['red', 'blue', 'green', 'purple', 'orange']
            
            for i, idx in enumerate(discord_indices):
                color = colors[i % len(colors)]
                
                # Plot discord
                plt.plot(range(idx, idx + self.window_size), 
                        multivariate_series[dim, idx:idx + self.window_size], 
                        color=color, linewidth=2)
            
            plt.title(f'{col} with Discords')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/all_discords_overview.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_pipeline(self, df, target_cols, n_motifs=3, n_discords=3):
        """
        Run the entire Multivariate Matrix Profile analysis pipeline
        
        Args:
            df: DataFrame with datetime index
            target_cols: List of column names to analyze
            n_motifs: Number of motifs to find
            n_discords: Number of discords to find
            
        Returns:
            results: Dictionary with analysis results
        """
        # Compute Multivariate Matrix Profile
        mp, mp_profile, mp_indices = self.compute_matrix_profile(df, target_cols)
        
        # Find motifs (recurring patterns)
        motif_indices, motif_data = self.find_motifs(df, mp, mp_indices, n_motifs, target_cols)
        
        # Find discords (anomalies)
        discord_indices = self.find_discords(mp_profile, n_discords)
        
        # Visualize discords
        self.visualize_discords(df, discord_indices, target_cols)
        
        results = {
            'window_size': self.window_size,
            'variables': target_cols,
            'motifs': motif_indices,
            'discords': discord_indices
        }
        
        # Save results summary
        with open(f"{self.output_dir}/results_summary.json", 'w') as f:
            json.dump({
                'window_size': self.window_size,
                'n_variables': len(target_cols),
                'variables': target_cols,
                'n_motifs_found': len(motif_indices),
                'n_discords_found': len(discord_indices)
            }, f, indent=2)
        
        return results

if __name__ == "__main__":
    # Load cleaned data
    df = pd.read_csv('data/bitcoin_data_cleaned_no_resample_original.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    
    # Resample to hourly for simplicity
    df_hourly = df.resample('1H').mean().fillna(method='ffill')
    
    # Select target columns for multivariate analysis
    target_cols = [
        'block_median_fee_rate',    # Primary target (fee)
        'mempool_size_bytes',       # Size of mempool in bytes
        'btc_price_usd',            # BTC price in USD
        'mempool_tx_count'          # Number of transactions in mempool
    ]
    
    # Initialize and run Matrix Profile analysis
    mp_analyzer = BitcoinFeeMultivariateMatrixProfile(
        window_size=24,  # 24 hours = daily patterns
        output_dir='models/matrix_profile/output_multivariate'
    )
    
    results = mp_analyzer.run_pipeline(
        df_hourly,
        target_cols=target_cols,
        n_motifs=3,
        n_discords=3
    )
    
    print("Multivariate Matrix Profile analysis completed!")
    print(f"Found {len(results['motifs'])} motifs and {len(results['discords'])} discords")
    print(f"Check {mp_analyzer.output_dir} for detailed results") 