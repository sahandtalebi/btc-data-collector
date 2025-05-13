import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import stumpy
from datetime import datetime, timedelta

class BitcoinFeeMatrixProfile:
    def __init__(self, 
                 window_size=24,  # 24 hours = daily patterns
                 output_dir='models/matrix_profile/output'):
        """
        Initialize Matrix Profile for Bitcoin fee analysis
        
        Args:
            window_size: Size of sliding window for pattern mining (24 = daily patterns)
            output_dir: Directory to save outputs
        """
        self.window_size = window_size
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def compute_matrix_profile(self, df, target_col='block_median_fee_rate'):
        """
        Compute Matrix Profile for time series
        
        Args:
            df: DataFrame with datetime index and target column
            target_col: Column name of target variable (fees)
            
        Returns:
            mp: Matrix Profile
            mp_indices: Matrix Profile indices
        """
        print(f"Computing Matrix Profile with window size {self.window_size}...")
        
        # Extract target variable
        if isinstance(df, pd.DataFrame):
            if target_col in df.columns:
                series = df[target_col].values
            else:
                series = df.iloc[:, 0].values
        else:
            series = df  # Assume numpy array or similar
        
        # Compute Matrix Profile
        mp = stumpy.stump(series, self.window_size)
        
        # Extract profile and indices
        mp_profile = mp[:, 0]  # distance profile
        mp_indices = mp[:, 1]  # matrix profile indices
        
        # Save raw matrix profile
        np.save(f"{self.output_dir}/matrix_profile.npy", mp_profile)
        np.save(f"{self.output_dir}/matrix_profile_indices.npy", mp_indices)
        
        # Plot Matrix Profile
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 1, 1)
        plt.plot(series)
        plt.title('Original Time Series')
        plt.ylabel(target_col)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.plot(mp_profile)
        plt.title('Matrix Profile')
        plt.xlabel('Index')
        plt.ylabel('Distance')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/matrix_profile.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        return mp_profile, mp_indices
    
    def find_motifs(self, df, mp_indices, n_motifs=3, target_col='block_median_fee_rate'):
        """
        Find motifs (recurring patterns) in time series
        
        Args:
            df: DataFrame with datetime index and target column
            mp_indices: Matrix Profile indices
            n_motifs: Number of motifs to find
            target_col: Column name of target variable
            
        Returns:
            motifs: List of motif indices
        """
        print(f"Finding top {n_motifs} motifs...")
        
        # Extract time series
        if isinstance(df, pd.DataFrame):
            if target_col in df.columns:
                series = df[target_col].values
            else:
                series = df.iloc[:, 0].values
            timestamps = df.index
        else:
            series = df
            timestamps = np.arange(len(series))
        
        # Find motif indices using the matrix profile
        motif_indices = []
        exclusion_zone = self.window_size // 2
        
        # Get the top-n motifs
        for i in range(n_motifs):
            # Find the lowest matrix profile value that's not been found yet
            # and isn't in the exclusion zone of an already discovered motif
            if i == 0:
                # First motif is the lowest matrix profile value
                motif_idx = np.argmin(mp_indices)
                motif_neighbor_idx = int(mp_indices[motif_idx])
            else:
                # For subsequent motifs, we exclude previously found motifs
                for idx in range(len(mp_indices)):
                    if idx not in excluded_indices:
                        motif_idx = idx
                        motif_neighbor_idx = int(mp_indices[idx])
                        break
            
            # Store motif and its nearest neighbor
            motif_indices.append((motif_idx, motif_neighbor_idx))
            
            # Exclude zones around this motif for the next iteration
            excluded_indices = set()
            for idx in motif_indices[-1]:
                for j in range(max(0, idx - exclusion_zone), min(len(mp_indices), idx + exclusion_zone)):
                    excluded_indices.add(j)
        
        # Plot motifs
        plt.figure(figsize=(15, 10))
        
        # Plot the full time series
        plt.plot(series, alpha=0.5, color='gray')
        
        # Plot each motif with different colors
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        
        motif_data = []
        
        for i, (idx1, idx2) in enumerate(motif_indices):
            color = colors[i % len(colors)]
            
            # Plot motif and its nearest neighbor
            plt.plot(range(idx1, idx1 + self.window_size), 
                    series[idx1:idx1 + self.window_size], 
                    color=color, linewidth=2)
            
            plt.plot(range(idx2, idx2 + self.window_size), 
                    series[idx2:idx2 + self.window_size], 
                    color=color, linewidth=2, linestyle='--')
            
            # Store motif data (datetime ranges if available)
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
            except:
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
        
        plt.title(f'Top {n_motifs} Motifs (Window Size = {self.window_size})')
        plt.xlabel('Index')
        plt.ylabel(target_col)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/motifs.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save motif data
        with open(f"{self.output_dir}/motifs.json", 'w') as f:
            json.dump(motif_data, f, indent=2)
        
        # Plot each motif pair separately for detailed view
        for i, (idx1, idx2) in enumerate(motif_indices):
            plt.figure(figsize=(12, 6))
            
            plt.plot(range(self.window_size), 
                    series[idx1:idx1 + self.window_size], 
                    color='blue', linewidth=2, label=f'Motif {i+1}')
            
            plt.plot(range(self.window_size), 
                    series[idx2:idx2 + self.window_size], 
                    color='red', linewidth=2, linestyle='--', label=f'Match {i+1}')
            
            plt.title(f'Motif {i+1} Detail (Window Size = {self.window_size})')
            plt.xlabel('Steps within Pattern')
            plt.ylabel(target_col)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/motif_{i+1}_detail.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        return motif_indices, motif_data
    
    def find_discords(self, mp_profile, n_discords=3):
        """
        Find discords (anomalies) in time series
        
        Args:
            mp_profile: Matrix Profile
            n_discords: Number of discords to find
            
        Returns:
            discords: List of discord indices
        """
        print(f"Finding top {n_discords} discords (anomalies)...")
        
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
    
    def visualize_discords(self, df, discord_indices, target_col='block_median_fee_rate'):
        """
        Visualize discords (anomalies) in time series
        
        Args:
            df: DataFrame with datetime index and target column
            discord_indices: List of discord indices
            target_col: Column name of target variable
        """
        print("Visualizing discords...")
        
        # Extract time series
        if isinstance(df, pd.DataFrame):
            if target_col in df.columns:
                series = df[target_col].values
            else:
                series = df.iloc[:, 0].values
            timestamps = df.index
        else:
            series = df
            timestamps = np.arange(len(series))
        
        # Plot discords
        plt.figure(figsize=(15, 10))
        
        # Plot the full time series
        plt.plot(series, alpha=0.5, color='gray')
        
        # Plot each discord with different colors
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        
        discord_data = []
        
        for i, idx in enumerate(discord_indices):
            color = colors[i % len(colors)]
            
            # Plot discord
            plt.plot(range(idx, idx + self.window_size), 
                    series[idx:idx + self.window_size], 
                    color=color, linewidth=2)
            
            # Store discord data (datetime ranges if available)
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
            except:
                # Fallback if timestamps are not available
                discord_data.append({
                    'discord_id': i + 1,
                    'start_index': int(idx),
                    'end_index': int(idx + self.window_size - 1)
                })
        
        plt.title(f'Top Discords (Anomalies) (Window Size = {self.window_size})')
        plt.xlabel('Index')
        plt.ylabel(target_col)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/discords.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save discord data
        with open(f"{self.output_dir}/discord_details.json", 'w') as f:
            json.dump(discord_data, f, indent=2)
        
        # Plot each discord separately for detailed view
        for i, idx in enumerate(discord_indices):
            plt.figure(figsize=(12, 6))
            
            # Plot the discord
            plt.plot(range(self.window_size), 
                    series[idx:idx + self.window_size], 
                    color='red', linewidth=2)
            
            plt.title(f'Discord {i+1} Detail (Window Size = {self.window_size})')
            plt.xlabel('Steps within Pattern')
            plt.ylabel(target_col)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/discord_{i+1}_detail.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def generate_fee_recommendations(self, df, motif_data, target_col='block_median_fee_rate'):
        """
        Generate fee recommendations based on motifs
        
        Args:
            df: DataFrame with datetime index and target column
            motif_data: Motif details
            target_col: Column name of target variable
            
        Returns:
            recommendations: Dictionary with fee recommendations
        """
        print("Generating fee recommendations based on patterns...")
        
        # Extract time series
        if isinstance(df, pd.DataFrame):
            if target_col in df.columns:
                series = df[target_col].values
            else:
                series = df.iloc[:, 0].values
            timestamps = df.index
        else:
            series = df
        
        # Analyze motifs to identify low-fee patterns
        motif_avg_fees = []
        
        for motif in motif_data:
            motif_id = motif['motif_id']
            idx1 = motif['occurrence1']['start_index']
            idx2 = motif['occurrence2']['start_index']
            
            # Extract the fees for both occurrences
            fees1 = series[idx1:idx1 + self.window_size]
            fees2 = series[idx2:idx2 + self.window_size]
            
            # Calculate average fee for each occurrence
            avg_fee1 = np.mean(fees1)
            avg_fee2 = np.mean(fees2)
            
            # Calculate overall average and standard deviation
            motif_avg = (avg_fee1 + avg_fee2) / 2
            motif_std = np.std(np.concatenate([fees1, fees2]))
            
            # Store for comparison
            motif_avg_fees.append({
                'motif_id': motif_id,
                'avg_fee': motif_avg,
                'std_fee': motif_std,
                'min_fee': min(np.min(fees1), np.min(fees2)),
                'occurrence1': {'start_index': idx1, 'avg_fee': avg_fee1},
                'occurrence2': {'start_index': idx2, 'avg_fee': avg_fee2}
            })
        
        # Sort motifs by average fee (lowest first)
        motif_avg_fees.sort(key=lambda x: x['avg_fee'])
        
        # Generate recommendations based on the patterns
        recommendations = {
            'lowest_fee_pattern': {
                'motif_id': motif_avg_fees[0]['motif_id'],
                'avg_fee': float(motif_avg_fees[0]['avg_fee']),
                'std_fee': float(motif_avg_fees[0]['std_fee']),
                'pattern_relative_hours': list(range(self.window_size)),
                'pattern_description': f"This pattern shows recurring periods of lower fees. Consider scheduling transactions during similar patterns."
            },
            'highest_fee_pattern': {
                'motif_id': motif_avg_fees[-1]['motif_id'],
                'avg_fee': float(motif_avg_fees[-1]['avg_fee']),
                'std_fee': float(motif_avg_fees[-1]['std_fee']),
                'pattern_relative_hours': list(range(self.window_size)),
                'pattern_description': f"This pattern shows recurring periods of higher fees. Avoid scheduling transactions during similar patterns if possible."
            },
            'optimal_transaction_recommendation': "Based on the identified patterns, transactions are likely to have lower fees if timed according to the lowest fee pattern."
        }
        
        # Save recommendations
        with open(f"{self.output_dir}/fee_recommendations.json", 'w') as f:
            json.dump(recommendations, f, indent=2)
        
        return recommendations
    
    def run_pipeline(self, df, target_col='block_median_fee_rate', n_motifs=3, n_discords=3):
        """
        Run the entire Matrix Profile analysis pipeline
        
        Args:
            df: DataFrame with datetime index and target column
            target_col: Column name to analyze
            n_motifs: Number of motifs to find
            n_discords: Number of discords to find
            
        Returns:
            results: Dictionary with analysis results
        """
        # Compute Matrix Profile
        mp_profile, mp_indices = self.compute_matrix_profile(df, target_col)
        
        # Find motifs (recurring patterns)
        motif_indices, motif_data = self.find_motifs(df, mp_indices, n_motifs, target_col)
        
        # Find discords (anomalies)
        discord_indices = self.find_discords(mp_profile, n_discords)
        
        # Visualize discords
        self.visualize_discords(df, discord_indices, target_col)
        
        # Generate fee recommendations
        recommendations = self.generate_fee_recommendations(df, motif_data, target_col)
        
        results = {
            'window_size': self.window_size,
            'motifs': motif_indices,
            'discords': discord_indices,
            'recommendations': recommendations
        }
        
        # Save results summary
        with open(f"{self.output_dir}/results_summary.json", 'w') as f:
            json.dump({
                'window_size': self.window_size,
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
    df_hourly = df['block_median_fee_rate'].resample('1H').mean().fillna(method='ffill')
    df_hourly = pd.DataFrame(df_hourly)
    
    # Initialize and run Matrix Profile analysis
    mp_analyzer = BitcoinFeeMatrixProfile(
        window_size=24,  # 24 hours = daily patterns
        output_dir='models/matrix_profile/output'
    )
    
    results = mp_analyzer.run_pipeline(
        df_hourly,
        target_col='block_median_fee_rate',
        n_motifs=3,
        n_discords=3
    )
    
    print("Matrix Profile analysis completed!")
    print(f"Found {len(results['motifs'])} motifs and {len(results['discords'])} discords")
    print(f"Check {mp_analyzer.output_dir} for detailed results") 