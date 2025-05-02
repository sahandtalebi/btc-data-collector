#!/usr/bin/env python3
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
        """مقداردهی اولیه کلاس جمع‌آوری داده بیت‌کوین"""
        # Mempool.space API base URL
        self.mempool_api = "https://mempool.space/api"
        
        # CoinGecko API base URL
        self.coingecko_api = "https://api.coingecko.com/api/v3"
        
        # مسیر فایل خروجی
        self.csv_output_path = csv_output_path
        
        # Create dataframe to store collected data
        self.df = pd.DataFrame(columns=[
            'timestamp', 'block_height', 'block_median_fee_rate', 'mempool_size_bytes',
            'mempool_tx_count', 'mempool_min_fee_rate', 'mempool_median_fee_rate',
            'block_weight', 'block_tx_count', 'block_interval_seconds',
            'difficulty', 'hash_rate', 'btc_price_usd', 'block_version',
            'mempool_avg_fee_rate', 'mempool_fee_rate_stddev',
            'mempool_fee_histogram', 'time_of_day', 'day_of_week',
            # ویژگی‌های تاخیری
            'prev_block_median_fee_rate', 'prev6_block_median_fee_rate',
            'prev_mempool_size_bytes', 'prev_mempool_tx_count',
            'prev_btc_price_usd'
        ])
        
        # کش برای ذخیره موقت دیتای بلاک‌ها
        self.block_cache = {}
        
        print(f"Bitcoin Data Collector initialized. Output will be saved to {csv_output_path}")
    
    def get_block_height_by_date(self, target_date):
        """تخمین ارتفاع بلاک نزدیک به تاریخ مشخص شده"""
        try:
            # تبدیل تاریخ به timestamp
            if isinstance(target_date, str):
                target_timestamp = int(datetime.strptime(target_date, "%Y-%m-%d").timestamp())
            else:
                target_timestamp = int(target_date.timestamp())
            
            # دریافت ارتفاع بلاک فعلی
            current_height_url = f"{self.mempool_api}/blocks/tip/height"
            response = requests.get(current_height_url)
            if response.status_code != 200:
                print(f"Error fetching current block height: {response.status_code}")
                return None
            
            current_height = int(response.text)
            
            # دریافت بلاک فعلی
            current_block = self.get_block_by_height(current_height)
            if not current_block:
                return None
            
            current_timestamp = current_block.get('timestamp')
            
            # محاسبه تقریبی تعداد بلاک‌ها بین تاریخ هدف و تاریخ فعلی
            # میانگین زمان بلاک: 10 دقیقه (600 ثانیه)
            time_diff = current_timestamp - target_timestamp
            block_diff = time_diff // 600  # تقسیم بر میانگین زمان بلاک (10 دقیقه = 600 ثانیه)
            
            estimated_height = max(0, current_height - block_diff)
            
            # تصحیح تخمین با چند بار تکرار
            for _ in range(3):  # حداکثر 3 بار تلاش برای نزدیک‌تر شدن
                block = self.get_block_by_height(estimated_height)
                if not block:
                    break
                
                block_timestamp = block.get('timestamp')
                time_diff = block_timestamp - target_timestamp
                
                # اگر به اندازه کافی نزدیک هستیم، خروج
                if abs(time_diff) < 3600:  # یک ساعت
                    return estimated_height
                
                # تصحیح تخمین
                block_diff = time_diff // 600
                estimated_height = max(0, estimated_height - block_diff)
            
            return estimated_height
        except Exception as e:
            print(f"Error in get_block_height_by_date: {e}")
            return None
    
    def get_block_by_height(self, height):
        """دریافت اطلاعات بلاک با ارتفاع مشخص"""
        try:
            # بررسی کش
            if height in self.block_cache:
                return self.block_cache[height]
            
            # دریافت هش بلاک
            hash_url = f"{self.mempool_api}/block-height/{height}"
            response = requests.get(hash_url)
            if response.status_code != 200:
                print(f"Error fetching block hash for height {height}: {response.status_code}")
                return None
            
            block_hash = response.text
            
            # دریافت اطلاعات بلاک
            block_url = f"{self.mempool_api}/block/{block_hash}"
            response = requests.get(block_url)
            if response.status_code != 200:
                print(f"Error fetching block {block_hash}: {response.status_code}")
                return None
            
            block_data = response.json()
            
            # ذخیره در کش
            self.block_cache[height] = block_data
            
            return block_data
        except Exception as e:
            print(f"Error in get_block_by_height({height}): {e}")
            return None
    
    def get_mempool_stats(self):
        """دریافت آمار ممپول"""
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
        """دریافت بلاک‌های پیش‌بینی شده ممپول (هیستوگرام کارمزد)"""
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
        """دریافت تخمین کارمزدها"""
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
        """دریافت سختی شبکه"""
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
        """دریافت تراکنش‌های یک بلاک (برای محاسبه نرخ کارمزد میانه بلاک)"""
        try:
            url = f"{self.mempool_api}/block/{block_hash}/txids"
            response = requests.get(url)
            if response.status_code != 200:
                print(f"Error fetching block transactions: {response.status_code}")
                return None
            
            tx_ids = response.json()
            
            # از آنجایی که تعداد تراکنش‌ها می‌تواند زیاد باشد، فقط یک نمونه انتخاب می‌کنیم
            # حداکثر 100 تراکنش برای تخمین نرخ کارمزد میانه
            sample_size = min(100, len(tx_ids))
            sampled_tx_ids = tx_ids[:sample_size]  # تراکنش‌های اول معمولاً کارمزد بالاتری دارند
            
            tx_fee_rates = []
            for tx_id in sampled_tx_ids:
                tx_data = self.get_transaction(tx_id)
                if tx_data and 'fee' in tx_data and 'weight' in tx_data:
                    fee = tx_data['fee']
                    weight = tx_data['weight']
                    vsize = weight / 4  # تبدیل وزن به سایز مجازی
                    fee_rate = fee / vsize if vsize > 0 else 0
                    tx_fee_rates.append(fee_rate)
                
                # تأخیر کوتاه برای جلوگیری از محدودیت نرخ
                time.sleep(0.1)
            
            if tx_fee_rates:
                return tx_fee_rates
            
            return None
        except Exception as e:
            print(f"Error in get_block_transactions: {e}")
            return None
    
    def get_transaction(self, tx_id):
        """دریافت اطلاعات تراکنش"""
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
        """دریافت قیمت بیت‌کوین از CoinGecko"""
        try:
            if timestamp:
                # تبدیل timestamp به تاریخ
                date = datetime.fromtimestamp(timestamp).strftime('%d-%m-%Y')
                
                # دریافت قیمت در تاریخ خاص
                url = f"{self.coingecko_api}/coins/bitcoin/history?date={date}"
                response = requests.get(url)
                if response.status_code != 200:
                    print(f"Error fetching historical Bitcoin price: {response.status_code}")
                    return None
                
                data = response.json()
                return data.get('market_data', {}).get('current_price', {}).get('usd')
            else:
                # دریافت قیمت فعلی
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
        """محاسبه نرخ کارمزد میانه بلاک"""
        tx_fee_rates = self.get_block_transactions(block_hash)
        if tx_fee_rates and len(tx_fee_rates) > 0:
            return statistics.median(tx_fee_rates)
        return None
    
    def calculate_mempool_fee_stats(self, mempool_blocks):
        """محاسبه آمار کارمزد ممپول"""
        if not mempool_blocks or not isinstance(mempool_blocks, list) or len(mempool_blocks) == 0:
            return None, None, None
        
        # استخراج نرخ‌های کارمزد از بلاک‌های ممپول
        fee_rates = []
        histogram = {}
        
        for block in mempool_blocks:
            fees = block.get('feeRange', [])
            if len(fees) >= 2:
                min_fee, max_fee = fees[0], fees[-1]
                count = block.get('nTx', 0)
                
                # افزودن نرخ‌های کارمزد به لیست (با وزن تعداد تراکنش‌ها)
                avg_fee = (min_fee + max_fee) / 2
                fee_rates.extend([avg_fee] * count)
                
                # افزودن به هیستوگرام
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
        """جمع‌آوری داده‌های مربوط به یک بلاک خاص"""
        try:
            print(f"Collecting data for block at height {height}")
            
            # دریافت اطلاعات بلاک
            block = self.get_block_by_height(height)
            if not block:
                print(f"Failed to retrieve block at height {height}")
                return None
            
            # زمان بلاک
            timestamp = block.get('timestamp')
            dt = datetime.fromtimestamp(timestamp) if timestamp else datetime.now()
            
            # دریافت اطلاعات بلاک قبلی (برای محاسبه فاصله زمانی)
            prev_block = self.get_block_by_height(height - 1)
            prev_timestamp = prev_block.get('timestamp') if prev_block else None
            
            # محاسبه فاصله زمانی بلاک
            block_interval = timestamp - prev_timestamp if prev_timestamp else None
            
            # دریافت آمار ممپول
            mempool_stats = self.get_mempool_stats()
            
            # دریافت تخمین کارمزدها
            fee_estimates = self.get_fee_estimates()
            
            # دریافت بلاک‌های ممپول (برای هیستوگرام کارمزد)
            mempool_blocks = self.get_mempool_blocks()
            
            # محاسبه آمار کارمزد ممپول
            mempool_avg_fee_rate, mempool_fee_rate_stddev, fee_histogram = self.calculate_mempool_fee_stats(mempool_blocks)
            
            # دریافت سختی شبکه
            difficulty_data = self.get_difficulty()
            
            # دریافت قیمت بیت‌کوین
            btc_price = self.get_bitcoin_price(timestamp)
            
            # محاسبه نرخ کارمزد میانه بلاک
            block_median_fee_rate = self.calculate_block_median_fee_rate(block.get('id'))
            
            # دریافت داده‌های بلاک قبلی برای ویژگی‌های تاخیری
            prev_block_data = self.df[self.df['block_height'] == height - 1].to_dict('records')[0] if not self.df.empty and any(self.df['block_height'] == height - 1) else None
            prev6_block_data = self.df[self.df['block_height'] == height - 6].to_dict('records')[0] if not self.df.empty and any(self.df['block_height'] == height - 6) else None
            
            # ایجاد رکورد
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
                # ویژگی‌های تاخیری
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
        """جمع‌آوری داده‌ها برای محدوده‌ای از بلاک‌ها"""
        if start_height > end_height:
            print("Start height must be less than or equal to end height")
            return False
        
        total_blocks = end_height - start_height + 1
        print(f"Starting data collection for {total_blocks} blocks from {start_height} to {end_height}")
        
        try:
            for height in range(start_height, end_height + 1):
                record = self.collect_block_data(height)
                if record:
                    # اضافه کردن به دیتافریم
                    self.df = pd.concat([self.df, pd.DataFrame([record])], ignore_index=True)
                    print(f"Successfully collected data for block {height}")
                else:
                    print(f"Failed to collect data for block {height}")
                
                # ذخیره دوره‌ای
                if (height - start_height + 1) % save_interval == 0:
                    self.save_to_csv()
                    print(f"Progress: {height - start_height + 1}/{total_blocks} blocks processed")
                
                # تأخیر برای جلوگیری از محدودیت نرخ
                time.sleep(1)
            
            # ذخیره نهایی
            self.save_to_csv()
            print(f"Data collection completed for blocks {start_height} to {end_height}")
            return True
        except Exception as e:
            print(f"Error during data collection: {e}")
            # ذخیره داده‌های جمع‌آوری شده تا این لحظه
            self.save_to_csv()
            return False
    
    def collect_data_by_date_range(self, start_date, end_date, save_interval=10):
        """جمع‌آوری داده‌ها برای محدوده زمانی"""
        try:
            # تبدیل تاریخ‌ها به ارتفاع بلاک
            start_height = self.get_block_height_by_date(start_date)
            if not start_height:
                print(f"Failed to determine block height for start date {start_date}")
                return False
            
            end_height = self.get_block_height_by_date(end_date)
            if not end_height:
                print(f"Failed to determine block height for end date {end_date}")
                return False
            
            # جمع‌آوری داده‌ها
            return self.collect_data_for_blocks(start_height, end_height, save_interval)
        except Exception as e:
            print(f"Error in collect_data_by_date_range: {e}")
            return False
    
    def save_to_csv(self):
        """ذخیره داده‌ها به فایل CSV"""
        try:
            if not self.df.empty:
                # مرتب‌سازی براساس ارتفاع بلاک
                self.df = self.df.sort_values(by='block_height').reset_index(drop=True)
                
                # ذخیره به CSV
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
        """بارگیری داده‌ها از فایل CSV"""
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

def main():
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
    
    # ایجاد جمع‌آوری‌کننده داده
    collector = BitcoinDataCollector(csv_output_path=args.output)
    
    # بارگیری داده‌های موجود در صورت نیاز
    if args.continue_from_file:
        if not collector.load_from_csv():
            print("Failed to load existing data. Starting fresh collection.")
    
    if args.mode == 'blocks':
        # بررسی پارامترهای ورودی
        if args.start_height is None or args.end_height is None:
            print("Error: start_height and end_height are required for blocks mode")
            return
        
        if args.start_height > args.end_height:
            print("Error: start_height must be less than or equal to end_height")
            return
        
        # جمع‌آوری داده‌ها براساس ارتفاع بلاک
        print(f"Collecting data for blocks from {args.start_height} to {args.end_height}")
        collector.collect_data_for_blocks(args.start_height, args.end_height, args.save_interval)
    else:  # mode == 'dates'
        # بررسی پارامترهای ورودی
        if args.start_date is None or args.end_date is None:
            print("Error: start_date and end_date are required for dates mode")
            return
        
        # اعتبارسنجی فرمت تاریخ
        try:
            datetime.strptime(args.start_date, "%Y-%m-%d")
            datetime.strptime(args.end_date, "%Y-%m-%d")
        except ValueError:
            print("Error: dates must be in YYYY-MM-DD format")
            return
        
        if args.start_date > args.end_date:
            print("Error: start_date must be before end_date")
            return
        
        # جمع‌آوری داده‌ها براساس محدوده تاریخ
        print(f"Collecting data for date range from {args.start_date} to {args.end_date}")
        collector.collect_data_by_date_range(args.start_date, args.end_date, args.save_interval)

if __name__ == "__main__":
    main() 