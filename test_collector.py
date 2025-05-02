#!/usr/bin/env python3
"""
اسکریپت تست برای کلاس BitcoinDataCollector
"""
import os
import pandas as pd
from bitcoin_data_collector_v2 import BitcoinDataCollector

def test_single_block():
    """تست جمع‌آوری داده برای یک بلاک خاص"""
    collector = BitcoinDataCollector(csv_output_path="test_single_block.csv")
    
    # انتخاب یک بلاک جدید
    # دریافت بلاک فعلی
    current_height_url = f"{collector.mempool_api}/blocks/tip/height"
    import requests
    response = requests.get(current_height_url)
    if response.status_code == 200:
        current_height = int(response.text)
        # انتخاب بلاک چند تا عقب‌تر برای اطمینان از اینکه کاملاً تأیید شده است
        test_block_height = current_height - 10
        
        print(f"Testing data collection for block at height {test_block_height}")
        
        # جمع‌آوری داده
        block_data = collector.collect_block_data(test_block_height)
        
        if block_data:
            print("\nCollected data:")
            for key, value in block_data.items():
                print(f"{key}: {value}")
            
            # ذخیره به CSV
            collector.df = pd.concat([collector.df, pd.DataFrame([block_data])], ignore_index=True)
            collector.save_to_csv()
            
            print(f"\nData saved to {collector.csv_output_path}")
            return True
        else:
            print("Failed to collect data for the test block")
            return False
    else:
        print(f"Error fetching current block height: {response.status_code}")
        return False

def test_date_range():
    """تست جمع‌آوری داده برای محدوده زمانی کوتاه"""
    collector = BitcoinDataCollector(csv_output_path="test_date_range.csv")
    
    # جمع‌آوری داده‌ها برای یک روز اخیر
    from datetime import datetime, timedelta
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    print(f"Testing data collection for date range from {start_date} to {end_date}")
    
    result = collector.collect_data_by_date_range(start_date, end_date, save_interval=5)
    
    if result:
        print(f"Successfully collected data for date range. {len(collector.df)} records saved to {collector.csv_output_path}")
        return True
    else:
        print("Failed to collect data for the date range")
        return False

def test_height_range():
    """تست جمع‌آوری داده برای محدوده بلاک کوتاه"""
    collector = BitcoinDataCollector(csv_output_path="test_height_range.csv")
    
    # دریافت بلاک فعلی
    import requests
    current_height_url = f"{collector.mempool_api}/blocks/tip/height"
    response = requests.get(current_height_url)
    if response.status_code == 200:
        current_height = int(response.text)
        
        # جمع‌آوری داده‌ها برای 5 بلاک اخیر
        end_height = current_height - 5
        start_height = current_height - 10
        
        print(f"Testing data collection for block range from {start_height} to {end_height}")
        
        result = collector.collect_data_for_blocks(start_height, end_height, save_interval=2)
        
        if result:
            print(f"Successfully collected data for block range. {len(collector.df)} records saved to {collector.csv_output_path}")
            return True
        else:
            print("Failed to collect data for the block range")
            return False
    else:
        print(f"Error fetching current block height: {response.status_code}")
        return False

def cleanup():
    """پاک کردن فایل‌های تست"""
    test_files = ["test_single_block.csv", "test_date_range.csv", "test_height_range.csv"]
    
    for file in test_files:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"Removed test file: {file}")
            except Exception as e:
                print(f"Error removing {file}: {e}")

if __name__ == "__main__":
    print("=== Testing Bitcoin Data Collector ===")
    
    try:
        print("\n--- Test 1: Single Block ---")
        test_single_block()
        
        print("\n--- Test 2: Height Range ---")
        test_height_range()
        
        print("\n--- Test 3: Date Range ---")
        test_date_range()
        
        print("\nAll tests completed.")
    except Exception as e:
        print(f"Error during tests: {e}")
    finally:
        # پرسیدن از کاربر برای پاک کردن فایل‌های تست
        response = input("\nClean up test files? (y/n): ")
        if response.lower() == 'y':
            cleanup()
            print("Test files cleaned up.")
        else:
            print("Test files kept for review.") 