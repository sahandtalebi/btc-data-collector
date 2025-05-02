#!/usr/bin/env python3
# Main Bitcoin Data Collector Class

import requests
import pandas as pd
import time
import datetime
import os
import json
import argparse
import statistics
from datetime import datetime, timedelta
import numpy as np

class BitcoinDataCollector:
    def __init__(self, csv_output_path="bitcoin_data.csv"):
        """Initialize Bitcoin Data Collector"""
        # Mempool.space API base URL
        self.mempool_api = "https://mempool.space/api"
        
        # CoinGecko API base URL
        self.coingecko_api = "https://api.coingecko.com/api/v3"
        
        # Output file path
        self.csv_output_path = csv_output_path
        
        # Create dataframe to store collected data
        self.df = pd.DataFrame(columns=[
            'timestamp', 'block_height', 'block_median_fee_rate', 'mempool_size_bytes',
            'mempool_tx_count', 'mempool_min_fee_rate', 'mempool_median_fee_rate',
            'block_weight', 'block_tx_count', 'block_interval_seconds',
            'difficulty', 'hash_rate', 'btc_price_usd', 'block_version',
            'mempool_avg_fee_rate', 'mempool_fee_rate_stddev',
            'mempool_fee_histogram', 'time_of_day', 'day_of_week',
            # Lagged features
            'prev_block_median_fee_rate', 'prev6_block_median_fee_rate',
            'prev_mempool_size_bytes', 'prev_mempool_tx_count',
            'prev_btc_price_usd'
        ])
        
        # Cache for block data
        self.block_cache = {}
        
        print(f"Bitcoin Data Collector initialized. Output will be saved to {csv_output_path}")
    
    def get_block_height_by_date(self, target_date):
        """Estimate block height close to specified date"""
        try:
            # Convert date to timestamp
            if isinstance(target_date, str):
                target_timestamp = int(datetime.strptime(target_date, "%Y-%m-%d").timestamp())
            else:
                target_timestamp = int(target_date.timestamp())
            
            # Get current block height
            current_height_url = f"{self.mempool_api}/blocks/tip/height"
            response = requests.get(current_height_url)
            if response.status_code != 200:
                print(f"Error fetching current block height: {response.status_code}")
                return None
            
            current_height = int(response.text)
            
            # Get current block
            current_block = self.get_block_by_height(current_height)
            if not current_block:
                return None
            
            current_timestamp = current_block.get('timestamp')
            
            # Calculate approximate number of blocks between target date and current date
            # Average block time: 10 minutes (600 seconds)
            time_diff = current_timestamp - target_timestamp
            block_diff = time_diff // 600  # Divide by average block time (10 minutes = 600 seconds)
            
            estimated_height = max(0, current_height - block_diff)
            
            # Refine estimate with a few iterations
            for _ in range(3):  # Maximum 3 attempts to get closer
                block = self.get_block_by_height(estimated_height)
                if not block:
                    break
                
                block_timestamp = block.get('timestamp')
                time_diff = block_timestamp - target_timestamp
                
                # If close enough, exit
                if abs(time_diff) < 3600:  # One hour
                    return estimated_height
                
                # Adjust estimate
                block_diff = time_diff // 600
                estimated_height = max(0, estimated_height - block_diff)
            
            return estimated_height
        except Exception as e:
            print(f"Error in get_block_height_by_date: {e}")
            return None
    
    def get_block_by_height(self, height):
        """Get block information by height"""
        try:
            # Check cache
            if height in self.block_cache:
                return self.block_cache[height]
            
            # Get block hash
            hash_url = f"{self.mempool_api}/block-height/{height}"
            response = requests.get(hash_url)
            if response.status_code != 200:
                print(f"Error fetching block hash for height {height}: {response.status_code}")
                return None
            
            block_hash = response.text
            
            # Get block data
            block_url = f"{self.mempool_api}/block/{block_hash}"
            response = requests.get(block_url)
            if response.status_code != 200:
                print(f"Error fetching block {block_hash}: {response.status_code}")
                return None
            
            block_data = response.json()
            
            # Save to cache
            self.block_cache[height] = block_data
            
            return block_data
        except Exception as e:
            print(f"Error in get_block_by_height({height}): {e}")
            return None
    
    def get_mempool_stats(self):
        """Get mempool statistics"""
        try:
            url = f"{self.mempool_api}/mempool"
            response = requests.get(url)
            if response.status_code != 200:
                print(f"Error fetching mempool stats: {response.status_code}")
                return None
            
            return response.json()
        except Exception as e:
            print(f"Error in get_mempool_stats: {e}")
            return None
    
    def get_mempool_blocks(self):
        """Get predicted mempool blocks (fee histogram)"""
        try:
            url = f"{self.mempool_api}/v1/fees/mempool-blocks"
            response = requests.get(url)
            if response.status_code != 200:
                print(f"Error fetching mempool blocks: {response.status_code}")
                return None
            
            return response.json()
        except Exception as e:
            print(f"Error in get_mempool_blocks: {e}")
            return None
    
    def get_fee_estimates(self):
        """Get fee estimates"""
        try:
            url = f"{self.mempool_api}/v1/fees/recommended"
            response = requests.get(url)
            if response.status_code != 200:
                print(f"Error fetching fee estimates: {response.status_code}")
                return None
            
            return response.json()
        except Exception as e:
            print(f"Error in get_fee_estimates: {e}")
            return None
    
    def get_difficulty(self):
        """Get network difficulty"""
        try:
            url = f"{self.mempool_api}/v1/difficulty-adjustment"
            response = requests.get(url)
            if response.status_code != 200:
                print(f"Error fetching difficulty: {response.status_code}")
                return None
            
            return response.json()
        except Exception as e:
            print(f"Error in get_difficulty: {e}")
            return None
    
    def get_block_transactions(self, block_hash):
        """Get transactions of a block (for calculating block median fee rate)"""
        try:
            url = f"{self.mempool_api}/block/{block_hash}/txids"
            response = requests.get(url)
            if response.status_code != 200:
                print(f"Error fetching block transactions: {response.status_code}")
                return None
            
            tx_ids = response.json()
            
            # Since the number of transactions can be large, we only sample a portion
            # Maximum 100 transactions for estimating median fee rate
            sample_size = min(100, len(tx_ids))
            sampled_tx_ids = tx_ids[:sample_size]  # First transactions usually have higher fee rates
            
            tx_fee_rates = []
            for tx_id in sampled_tx_ids:
                tx_data = self.get_transaction(tx_id)
                if tx_data and 'fee' in tx_data and 'weight' in tx_data:
                    fee = tx_data['fee']
                    weight = tx_data['weight']
                    vsize = weight / 4  # Convert weight to virtual size
                    fee_rate = fee / vsize if vsize > 0 else 0
                    tx_fee_rates.append(fee_rate)
                
                # Small delay to avoid rate limiting
                time.sleep(0.1)
            
            if tx_fee_rates:
                return tx_fee_rates
            
            return None
        except Exception as e:
            print(f"Error in get_block_transactions: {e}")
            return None
    
    def get_transaction(self, tx_id):
        """Get transaction details"""
        try:
            url = f"{self.mempool_api}/tx/{tx_id}"
            response = requests.get(url)
            if response.status_code != 200:
                print(f"Error fetching transaction {tx_id}: {response.status_code}")
                return None
            
            return response.json()
        except Exception as e:
            print(f"Error in get_transaction: {e}")
            return None
    
    def get_bitcoin_price(self, timestamp=None):
        """Get Bitcoin price from CoinGecko"""
        try:
            if timestamp:
                # Convert timestamp to date
                date = datetime.fromtimestamp(timestamp).strftime('%d-%m-%Y')
                
                # Get price on specific date
                url = f"{self.coingecko_api}/coins/bitcoin/history?date={date}"
                response = requests.get(url)
                if response.status_code != 200:
                    print(f"Error fetching historical Bitcoin price: {response.status_code}")
                    return None
                
                data = response.json()
                return data.get('market_data', {}).get('current_price', {}).get('usd')
            else:
                # Get current price
                url = f"{self.coingecko_api}/simple/price?ids=bitcoin&vs_currencies=usd"
                response = requests.get(url)
                if response.status_code != 200:
                    print(f"Error fetching current Bitcoin price: {response.status_code}")
                    return None
                
                data = response.json()
                return data.get('bitcoin', {}).get('usd')
        except Exception as e:
            print(f"Error in get_bitcoin_price: {e}")
            return None
    
    def calculate_block_median_fee_rate(self, block_hash):
        """Calculate block median fee rate"""
        tx_fee_rates = self.get_block_transactions(block_hash)
        if tx_fee_rates and len(tx_fee_rates) > 0:
            return statistics.median(tx_fee_rates)
        return None
    
    def calculate_mempool_fee_stats(self, mempool_blocks):
        """Calculate mempool fee statistics"""
        if not mempool_blocks or not isinstance(mempool_blocks, list) or len(mempool_blocks) == 0:
            return None, None, None
        
        # Extract fee rates from mempool blocks
        fee_rates = []
        histogram = {}
        
        for block in mempool_blocks:
            fees = block.get('feeRange', [])
            if len(fees) >= 2:
                min_fee, max_fee = fees[0], fees[-1]
                count = block.get('nTx', 0)
                
                # Add fee rates to list (weighted by transaction count)
                avg_fee = (min_fee + max_fee) / 2
                fee_rates.extend([avg_fee] * count)
                
                # Add to histogram
                fee_bucket = round(avg_fee)
                if fee_bucket in histogram:
                    histogram[fee_bucket] += count
                else:
                    histogram[fee_bucket] = count
        
        if fee_rates:
            avg_fee_rate = sum(fee_rates) / len(fee_rates)
            fee_rate_stddev = statistics.stdev(fee_rates) if len(fee_rates) > 1 else 0
            return avg_fee_rate, fee_rate_stddev, histogram
        
        return None, None, None
    
    def collect_block_data(self, height):
        """Collect data for a specific block"""
        try:
            print(f"Collecting data for block at height {height}")
            
            # Get block information
            block = self.get_block_by_height(height)
            if not block:
                print(f"Failed to retrieve block at height {height}")
                return None
            
            # Block timestamp
            timestamp = block.get('timestamp')
            dt = datetime.fromtimestamp(timestamp) if timestamp else datetime.now()
            
            # Get previous block information (for calculating time interval)
            prev_block = self.get_block_by_height(height - 1)
            prev_timestamp = prev_block.get('timestamp') if prev_block else None
            
            # Calculate block interval
            block_interval = timestamp - prev_timestamp if prev_timestamp else None
            
            # Get mempool stats
            mempool_stats = self.get_mempool_stats()
            
            # Get fee estimates
            fee_estimates = self.get_fee_estimates()
            
            # Get mempool blocks (for fee histogram)
            mempool_blocks = self.get_mempool_blocks()
            
            # Calculate mempool fee stats
            mempool_avg_fee_rate, mempool_fee_rate_stddev, fee_histogram = self.calculate_mempool_fee_stats(mempool_blocks)
            
            # Get network difficulty
            difficulty_data = self.get_difficulty()
            
            # Get Bitcoin price
            btc_price = self.get_bitcoin_price(timestamp)
            
            # Calculate block median fee rate
            block_median_fee_rate = self.calculate_block_median_fee_rate(block.get('id'))
            
            # Get previous block data for lagged features
            prev_block_data = self.df[self.df['block_height'] == height - 1].to_dict('records')[0] if not self.df.empty and any(self.df['block_height'] == height - 1) else None
            prev6_block_data = self.df[self.df['block_height'] == height - 6].to_dict('records')[0] if not self.df.empty and any(self.df['block_height'] == height - 6) else None
            
            # Create record
            record = {
                'timestamp': timestamp,
                'block_height': height,
                'block_median_fee_rate': block_median_fee_rate,
                'mempool_size_bytes': mempool_stats.get('vsize') if mempool_stats else None,
                'mempool_tx_count': mempool_stats.get('count') if mempool_stats else None,
                'mempool_min_fee_rate': fee_estimates.get('minimumFee') if fee_estimates else None,
                'mempool_median_fee_rate': fee_estimates.get('halfHourFee') if fee_estimates else None,
                'block_weight': block.get('weight'),
                'block_tx_count': block.get('tx_count'),
                'block_interval_seconds': block_interval,
                'difficulty': difficulty_data.get('current') if difficulty_data else None,
                'hash_rate': difficulty_data.get('currentHashrate') if difficulty_data else None,
                'btc_price_usd': btc_price,
                'block_version': block.get('version'),
                'mempool_avg_fee_rate': mempool_avg_fee_rate,
                'mempool_fee_rate_stddev': mempool_fee_rate_stddev,
                'mempool_fee_histogram': json.dumps(fee_histogram) if fee_histogram else None,
                'time_of_day': dt.hour,
                'day_of_week': dt.weekday(),
                # Lagged features
                'prev_block_median_fee_rate': prev_block_data.get('block_median_fee_rate') if prev_block_data else None,
                'prev6_block_median_fee_rate': prev6_block_data.get('block_median_fee_rate') if prev6_block_data else None,
                'prev_mempool_size_bytes': prev_block_data.get('mempool_size_bytes') if prev_block_data else None,
                'prev_mempool_tx_count': prev_block_data.get('mempool_tx_count') if prev_block_data else None,
                'prev_btc_price_usd': prev_block_data.get('btc_price_usd') if prev_block_data else None
            }
            
            return record
        except Exception as e:
            print(f"Error collecting data for block {height}: {e}")
            return None
    
    def collect_data_for_blocks(self, start_height, end_height, save_interval=10):
        """Collect data for a range of blocks"""
        if start_height > end_height:
            print("Start height must be less than or equal to end height")
            return False
        
        total_blocks = end_height - start_height + 1
        print(f"Starting data collection for {total_blocks} blocks from {start_height} to {end_height}")
        
        try:
            for height in range(start_height, end_height + 1):
                record = self.collect_block_data(height)
                if record:
                    # Add to dataframe
                    self.df = pd.concat([self.df, pd.DataFrame([record])], ignore_index=True)
                    print(f"Successfully collected data for block {height}")
                else:
                    print(f"Failed to collect data for block {height}")
                
                # Periodic saving
                if (height - start_height + 1) % save_interval == 0:
                    self.save_to_csv()
                    print(f"Progress: {height - start_height + 1}/{total_blocks} blocks processed")
                
                # Delay to avoid rate limiting
                time.sleep(1)
            
            # Final save
            self.save_to_csv()
            print(f"Data collection completed for blocks {start_height} to {end_height}")
            return True
        except Exception as e:
            print(f"Error during data collection: {e}")
            # Save data collected so far
            self.save_to_csv()
            return False
    
    def collect_data_by_date_range(self, start_date, end_date, save_interval=10):
        """Collect data for a date range"""
        try:
            # Convert dates to block heights
            start_height = self.get_block_height_by_date(start_date)
            if not start_height:
                print(f"Failed to determine block height for start date {start_date}")
                return False
            
            end_height = self.get_block_height_by_date(end_date)
            if not end_height:
                print(f"Failed to determine block height for end date {end_date}")
                return False
            
            # Collect data
            return self.collect_data_for_blocks(start_height, end_height, save_interval)
        except Exception as e:
            print(f"Error in collect_data_by_date_range: {e}")
            return False
    
    def save_to_csv(self):
        """Save data to CSV file"""
        try:
            if not self.df.empty:
                # Sort by block height
                self.df = self.df.sort_values(by='block_height').reset_index(drop=True)
                
                # Save to CSV
                self.df.to_csv(self.csv_output_path, index=False)
                print(f"Data saved to {self.csv_output_path} ({len(self.df)} records)")
                return True
            else:
                print("No data to save")
                return False
        except Exception as e:
            print(f"Error saving data to CSV: {e}")
            return False
    
    def load_from_csv(self):
        """Load data from CSV file"""
        try:
            if os.path.exists(self.csv_output_path):
                self.df = pd.read_csv(self.csv_output_path)
                print(f"Loaded {len(self.df)} records from {self.csv_output_path}")
                return True
            else:
                print(f"CSV file {self.csv_output_path} does not exist")
                return False
        except Exception as e:
            print(f"Error loading data from CSV: {e}")
            return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collect Bitcoin network and price data.')
    parser.add_argument('--mode', choices=['blocks', 'dates'], default='dates', help='Collection mode: by block heights or by dates')
    parser.add_argument('--start_height', type=int, help='Start block height (for blocks mode)')
    parser.add_argument('--end_height', type=int, help='End block height (for blocks mode)')
    parser.add_argument('--start_date', type=str, help='Start date in YYYY-MM-DD format (for dates mode)')
    parser.add_argument('--end_date', type=str, help='End date in YYYY-MM-DD format (for dates mode)')
    parser.add_argument('--output', type=str, default='bitcoin_data.csv', help='Output CSV filename')
    parser.add_argument('--save_interval', type=int, default=10, help='How often to save progress (number of blocks)')
    parser.add_argument('--continue_from_file', action='store_true', help='Continue collection from existing file')
    
    args = parser.parse_args()
    
    # Create data collector
    collector = BitcoinDataCollector(csv_output_path=args.output)
    
    # Load existing data if needed
    if args.continue_from_file:
        if not collector.load_from_csv():
            print("Failed to load existing data. Starting fresh collection.")
    
    if args.mode == 'blocks':
        # Check input parameters
        if args.start_height is None or args.end_height is None:
            print("Error: start_height and end_height are required for blocks mode")
            exit(1)
        
        if args.start_height > args.end_height:
            print("Error: start_height must be less than or equal to end_height")
            exit(1)
        
        # Collect data by block height
        print(f"Collecting data for blocks from {args.start_height} to {args.end_height}")
        collector.collect_data_for_blocks(args.start_height, args.end_height, args.save_interval)
    else:  # mode == 'dates'
        # Check input parameters
        if args.start_date is None or args.end_date is None:
            print("Error: start_date and end_date are required for dates mode")
            exit(1)
        
        # Validate date format
        try:
            datetime.strptime(args.start_date, "%Y-%m-%d")
            datetime.strptime(args.end_date, "%Y-%m-%d")
        except ValueError:
            print("Error: dates must be in YYYY-MM-DD format")
            exit(1)
        
        if args.start_date > args.end_date:
            print("Error: start_date must be before end_date")
            exit(1)
        
        # Collect data by date range
        print(f"Collecting data for date range from {args.start_date} to {args.end_date}")
        collector.collect_data_by_date_range(args.start_date, args.end_date, args.save_interval) 