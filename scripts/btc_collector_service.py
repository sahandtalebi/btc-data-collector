#!/usr/bin/env python3
import os
import time
import sys
import logging
from datetime import datetime, timedelta

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import BitcoinDataCollector from the main package
from src.collector.bitcoin_data_collector import BitcoinDataCollector

# Settings
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
SAVE_INTERVAL = 5  # Save every 5 blocks

# Create necessary directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def setup_logging():
    """Set up logging configuration"""
    log_file = os.path.join(LOG_DIR, f'collector_{datetime.now().strftime("%Y%m%d")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger('btc_collector')

def collect_last_six_months():
    """Collect data for the last six months"""
    logger = setup_logging()
    
    # Calculate start and end dates (6 months ago to today)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)  # Approximately 6 months
    
    # Convert to required string format
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    
    logger.info(f"Starting data collection for date range: {start_date_str} to {end_date_str}")
    
    # Create output filename
    output_file = os.path.join(DATA_DIR, f'bitcoin_data_{start_date_str}_to_{end_date_str}.csv')
    
    # Create collector instance
    collector = BitcoinDataCollector(csv_output_path=output_file)
    
    # Check for existing file and continue collection if needed
    continue_from_file = os.path.exists(output_file)
    if continue_from_file:
        logger.info(f"Found existing data file. Will continue from last collected point.")
        collector.load_from_csv()
    
    # Start data collection
    success = collector.collect_data_by_date_range(start_date_str, end_date_str, save_interval=SAVE_INTERVAL)
    
    if success:
        logger.info("Data collection completed successfully!")
    else:
        logger.error("Data collection encountered errors. Check the logs for details.")
    
    return success

def main():
    """Main function for running the service"""
    logger = setup_logging()
    logger.info("Bitcoin Data Collector service starting...")
    
    while True:
        try:
            # Collect data for the last six months
            collect_last_six_months()
            
            # After completing a full collection cycle, wait 24 hours before running again
            logger.info("Collection cycle completed. Sleeping for 24 hours before next cycle.")
            time.sleep(24 * 3600)  # 24 hours
            
        except KeyboardInterrupt:
            logger.info("Service stopped by user.")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            logger.info("Restarting collection in 10 minutes...")
            time.sleep(600)  # Wait 10 minutes and try again

if __name__ == "__main__":
    main() 