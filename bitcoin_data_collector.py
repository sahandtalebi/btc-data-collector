#!/usr/bin/env python3
import requests
import pandas as pd
import time
import datetime
import os
from datetime import datetime, timedelta
import json
import argparse

class BitcoinDataCollector:
    def __init__(self):
        # Mempool.space API base URL
        self.mempool_api_base = "https://mempool.space/api"
        
        # CoinGecko API base URL
        self.coingecko_api_base = "https://api.coingecko.com/api/v3"
        
        # Create dataframe to store collected data
        self.df = pd.DataFrame(columns=[
            'timestamp', 'block_height', 'median_fee_rate', 'mempool_size_bytes',
            'mempool_tx_count', 'mempool_min_fee', 'mempool_median_fee',
            'block_weight', 'block_tx_count', 'block_interval_seconds',
            'difficulty', 'hash_rate', 'btc_price_usd', 'block_version'
        ])
    
    def get_block_by_height(self, height):
        """Fetch block information by height"""
        try:
            url = f"{self.mempool_api_base}/block-height/{height}"
            response = requests.get(url)
            if response.status_code == 200:
                block_hash = response.text
                return self.get_block_by_hash(block_hash)
            else:
                print(f"Error fetching block height {height}: {response.status_code}")
                return None
        except Exception as e:
            print(f"Exception when fetching block height {height}: {e}")
            return None
    
    def get_block_by_hash(self, block_hash):
        """Fetch block information by hash"""
        try:
            url = f"{self.mempool_api_base}/block/{block_hash}"
            response = requests.get(url)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error fetching block {block_hash}: {response.status_code}")
                return None
        except Exception as e:
            print(f"Exception when fetching block {block_hash}: {e}")
            return None
    
    def get_blocks_by_date_range(self, start_date, end_date):
        """Get list of blocks within a date range"""
        try:
            start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
            end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
            
            # Get current tip height
            url = f"{self.mempool_api_base}/blocks/tip/height"
            response = requests.get(url)
            if response.status_code != 200:
                print(f"Error fetching tip height: {response.status_code}")
                return []
            
            current_height = int(response.text)
            
            # Collect blocks
            blocks = []
            height = current_height
            
            # Start from the tip and go backwards in time
            while True:
                print(f"Processing block at height {height}")
                block = self.get_block_by_height(height)
                if block is None:
                    height -= 1
                    continue
                
                block_timestamp = block.get('timestamp', 0)
                
                # If block timestamp is earlier than start date, stop
                if block_timestamp < start_timestamp:
                    break
                
                # If block timestamp is within our desired range, add it
                if start_timestamp <= block_timestamp <= end_timestamp:
                    blocks.append(block)
                
                # If we've gone beyond the end date, we're done
                if block_timestamp > end_timestamp:
                    pass
                
                height -= 1
                time.sleep(0.1)  # Add delay to avoid rate limits
            
            return sorted(blocks, key=lambda x: x.get('height', 0))
        except Exception as e:
            print(f"Exception when fetching blocks by date range: {e}")
            return []
    
    def get_mempool_stats(self, timestamp):
        """Get mempool statistics close to the given timestamp"""
        try:
            url = f"{self.mempool_api_base}/mempool"
            response = requests.get(url)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error fetching mempool stats: {response.status_code}")
                return None
        except Exception as e:
            print(f"Exception when fetching mempool stats: {e}")
            return None
    
    def get_fee_estimates(self):
        """Get fee estimates from mempool.space API"""
        try:
            url = f"{self.mempool_api_base}/v1/fees/recommended"
            response = requests.get(url)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error fetching fee estimates: {response.status_code}")
                return None
        except Exception as e:
            print(f"Exception when fetching fee estimates: {e}")
            return None
    
    def get_difficulty(self):
        """Get current network difficulty"""
        try:
            url = f"{self.mempool_api_base}/v1/difficulty-adjustment"
            response = requests.get(url)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error fetching difficulty: {response.status_code}")
                return None
        except Exception as e:
            print(f"Exception when fetching difficulty: {e}")
            return None
    
    def get_bitcoin_price_history(self, start_date, end_date):
        """Get Bitcoin price history from CoinGecko API"""
        try:
            # Convert dates to UNIX timestamps (milliseconds)
            start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
            end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
            
            url = f"{self.coingecko_api_base}/coins/bitcoin/market_chart/range"
            params = {
                "vs_currency": "usd",
                "from": start_timestamp // 1000,  # Convert to seconds
                "to": end_timestamp // 1000,      # Convert to seconds
            }
            
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                # Create dictionary mapping timestamps to prices
                price_dict = {}
                for price_data in data.get('prices', []):
                    timestamp, price = price_data
                    # Convert milliseconds to seconds
                    price_dict[timestamp // 1000] = price
                return price_dict
            else:
                print(f"Error fetching Bitcoin price history: {response.status_code}, {response.text}")
                return {}
        except Exception as e:
            print(f"Exception when fetching Bitcoin price history: {e}")
            return {}
    
    def find_closest_price(self, price_dict, target_timestamp):
        """Find the closest price to the target timestamp"""
        if not price_dict:
            return None
        
        closest_timestamp = min(price_dict.keys(), key=lambda x: abs(x - target_timestamp))
        return price_dict.get(closest_timestamp)
    
    def collect_data(self, start_date, end_date):
        """Collect Bitcoin data for the specified date range"""
        # Get blocks in the date range
        blocks = self.get_blocks_by_date_range(start_date, end_date)
        if not blocks:
            print("No blocks found in the specified date range")
            return False
        
        # Get Bitcoin price history
        price_history = self.get_bitcoin_price_history(start_date, end_date)
        if not price_history:
            print("Failed to fetch Bitcoin price history")
            return False
        
        # Process each block
        previous_timestamp = None
        for block in blocks:
            try:
                height = block.get('height')
                timestamp = block.get('timestamp')
                
                # Calculate block interval (except for the first block)
                block_interval = None
                if previous_timestamp is not None:
                    block_interval = timestamp - previous_timestamp
                previous_timestamp = timestamp
                
                # Get mempool stats close to this block's time
                mempool_stats = self.get_mempool_stats(timestamp)
                
                # Get fee estimates
                fee_estimates = self.get_fee_estimates()
                
                # Get difficulty data
                difficulty_data = self.get_difficulty()
                
                # Find closest Bitcoin price
                btc_price = self.find_closest_price(price_history, timestamp)
                
                # Extract the data we need
                record = {
                    'timestamp': timestamp,
                    'block_height': height,
                    'median_fee_rate': block.get('extras', {}).get('medianFee'),
                    'mempool_size_bytes': mempool_stats.get('vsize') if mempool_stats else None,
                    'mempool_tx_count': mempool_stats.get('count') if mempool_stats else None,
                    'mempool_min_fee': fee_estimates.get('minimumFee') if fee_estimates else None,
                    'mempool_median_fee': fee_estimates.get('halfHourFee') if fee_estimates else None,
                    'block_weight': block.get('weight'),
                    'block_tx_count': block.get('tx_count'),
                    'block_interval_seconds': block_interval,
                    'difficulty': difficulty_data.get('current') if difficulty_data else None,
                    'hash_rate': difficulty_data.get('currentHashrate') if difficulty_data else None,
                    'btc_price_usd': btc_price,
                    'block_version': block.get('version')
                }
                
                # Add to dataframe
                self.df = pd.concat([self.df, pd.DataFrame([record])], ignore_index=True)
                
                print(f"Processed block {height} at {datetime.fromtimestamp(timestamp)}")
                
                # Add a delay to avoid rate limits
                time.sleep(1)
                
            except Exception as e:
                print(f"Error processing block {block.get('height')}: {e}")
                continue
        
        return True
    
    def save_to_csv(self, filename="bitcoin_data.csv"):
        """Save the collected data to a CSV file"""
        try:
            self.df.to_csv(filename, index=False)
            print(f"Data saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving data to CSV: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Collect Bitcoin network and price data.')
    parser.add_argument('--start_date', type=str, required=True, help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end_date', type=str, required=True, help='End date in YYYY-MM-DD format')
    parser.add_argument('--output', type=str, default='bitcoin_data.csv', help='Output CSV filename')
    
    args = parser.parse_args()
    
    # Validate dates
    try:
        datetime.strptime(args.start_date, "%Y-%m-%d")
        datetime.strptime(args.end_date, "%Y-%m-%d")
    except ValueError:
        print("Invalid date format. Please use YYYY-MM-DD")
        return
    
    # Check that start_date is before end_date
    if args.start_date > args.end_date:
        print("Start date must be before end date")
        return
    
    collector = BitcoinDataCollector()
    print(f"Collecting data from {args.start_date} to {args.end_date}...")
    
    if collector.collect_data(args.start_date, args.end_date):
        collector.save_to_csv(args.output)
        print("Data collection completed successfully.")
    else:
        print("Data collection failed.")

if __name__ == "__main__":
    main() 