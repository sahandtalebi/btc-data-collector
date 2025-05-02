#!/usr/bin/env python3
import os
import time
from datetime import datetime, timedelta
from bitcoin_data_collector_v2 import BitcoinDataCollector

# تنظیمات
DATA_DIR = 'data'
LOG_DIR = 'logs'
SAVE_INTERVAL = 5  # ذخیره هر 5 بلاک یکبار

# ایجاد دایرکتوری‌های مورد نیاز
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def setup_logging():
    """راه‌اندازی لاگ گیری"""
    import logging
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
    """جمع‌آوری داده‌های ۶ ماه اخیر"""
    logger = setup_logging()
    
    # محاسبه تاریخ‌های شروع و پایان (۶ ماه قبل تا امروز)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)  # تقریباً ۶ ماه
    
    # تبدیل به فرمت استرینگ مورد نیاز
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    
    logger.info(f"Starting data collection for date range: {start_date_str} to {end_date_str}")
    
    # ایجاد نام فایل خروجی
    output_file = os.path.join(DATA_DIR, f'bitcoin_data_{start_date_str}_to_{end_date_str}.csv')
    
    # ایجاد نمونه جمع‌آوری‌کننده
    collector = BitcoinDataCollector(csv_output_path=output_file)
    
    # بررسی وجود فایل قبلی و ادامه جمع‌آوری در صورت نیاز
    continue_from_file = os.path.exists(output_file)
    if continue_from_file:
        logger.info(f"Found existing data file. Will continue from last collected point.")
        collector.load_from_csv()
    
    # شروع جمع‌آوری داده‌ها
    success = collector.collect_data_by_date_range(start_date_str, end_date_str, save_interval=SAVE_INTERVAL)
    
    if success:
        logger.info("Data collection completed successfully!")
    else:
        logger.error("Data collection encountered errors. Check the logs for details.")
    
    return success

def main():
    """تابع اصلی برای اجرای سرویس"""
    logger = setup_logging()
    logger.info("Bitcoin Data Collector service starting...")
    
    while True:
        try:
            # جمع‌آوری داده‌های ۶ ماه اخیر
            collect_last_six_months()
            
            # پس از اتمام جمع‌آوری کامل، ۲۴ ساعت صبر می‌کنیم تا دوباره اجرا شود
            logger.info("Collection cycle completed. Sleeping for 24 hours before next cycle.")
            time.sleep(24 * 3600)  # 24 ساعت
            
        except KeyboardInterrupt:
            logger.info("Service stopped by user.")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            logger.info("Restarting collection in 10 minutes...")
            time.sleep(600)  # 10 دقیقه صبر می‌کنیم و دوباره سعی می‌کنیم

if __name__ == "__main__":
    main() 