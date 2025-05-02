#!/usr/bin/env python3
import requests
import json
import time
import pandas as pd
from datetime import datetime, timedelta
import os

class BitcoinDataTest:
    def __init__(self):
        # API های اصلی که براساس تست‌های اولیه عملکرد خوبی دارند
        self.mempool_api = "https://mempool.space/api"
        self.blockstream_api = "https://blockstream.info/api"
        self.coingecko_api = "https://api.coingecko.com/api/v3"
        
        # ایجاد دیکشنری برای ذخیره نتایج تست
        self.test_results = {
            "mempool_space": {},
            "blockstream": {},
            "coingecko": {}
        }
        
        # ایجاد دیتافریم نمونه
        self.sample_df = pd.DataFrame(columns=[
            'timestamp', 'block_height', 'block_median_fee_rate', 'mempool_size_bytes',
            'mempool_tx_count', 'mempool_min_fee_rate', 'mempool_median_fee_rate',
            'block_weight', 'block_tx_count', 'block_interval_seconds',
            'difficulty', 'hash_rate', 'btc_price_usd', 'block_version',
            'mempool_avg_fee_rate', 'mempool_fee_rate_stddev',
            'mempool_fee_histogram', 'time_of_day', 'day_of_week'
        ])
    
    def test_mempool_space(self):
        """تست API های mempool.space"""
        api = self.mempool_api
        results = self.test_results["mempool_space"]
        
        # تست دسترسی به اطلاعات بلاک
        try:
            # دریافت ارتفاع آخرین بلاک
            tip_url = f"{api}/blocks/tip/height"
            response = requests.get(tip_url)
            if response.status_code == 200:
                block_height = int(response.text)
                results["block_tip"] = {"success": True, "data": block_height}
                
                # دریافت هش بلاک با ارتفاع مشخص
                height_url = f"{api}/block-height/{block_height}"
                response = requests.get(height_url)
                if response.status_code == 200:
                    block_hash = response.text
                    results["block_hash"] = {"success": True, "data": block_hash}
                    
                    # دریافت اطلاعات کامل بلاک
                    block_url = f"{api}/block/{block_hash}"
                    response = requests.get(block_url)
                    if response.status_code == 200:
                        block_data = response.json()
                        results["block_data"] = {"success": True, "data": block_data}
                        print(f"Successfully retrieved block data for height {block_height}")
                        
                        # بررسی فیلدهای مهم در داده‌های بلاک
                        fields = ['height', 'timestamp', 'tx_count', 'weight', 'version']
                        missing_fields = [f for f in fields if f not in block_data]
                        if missing_fields:
                            print(f"Warning: Missing fields in block data: {missing_fields}")
                        else:
                            print("All required block fields are present")
                    else:
                        results["block_data"] = {"success": False, "error": f"Status code: {response.status_code}"}
                else:
                    results["block_hash"] = {"success": False, "error": f"Status code: {response.status_code}"}
            else:
                results["block_tip"] = {"success": False, "error": f"Status code: {response.status_code}"}
        except Exception as e:
            print(f"Error testing mempool.space block data: {e}")
            results["block_data"] = {"success": False, "error": str(e)}
        
        # تست دسترسی به اطلاعات ممپول
        try:
            # دریافت آمار کلی ممپول
            mempool_url = f"{api}/mempool"
            response = requests.get(mempool_url)
            if response.status_code == 200:
                mempool_data = response.json()
                results["mempool_stats"] = {"success": True, "data": mempool_data}
                print("Successfully retrieved mempool statistics")
                
                # بررسی فیلدهای مهم در آمار ممپول
                fields = ['count', 'vsize']
                missing_fields = [f for f in fields if f not in mempool_data]
                if missing_fields:
                    print(f"Warning: Missing fields in mempool stats: {missing_fields}")
                else:
                    print("All required mempool stats fields are present")
            else:
                results["mempool_stats"] = {"success": False, "error": f"Status code: {response.status_code}"}
        except Exception as e:
            print(f"Error testing mempool.space mempool stats: {e}")
            results["mempool_stats"] = {"success": False, "error": str(e)}
        
        # تست دسترسی به تخمین کارمزدها
        try:
            fee_url = f"{api}/v1/fees/recommended"
            response = requests.get(fee_url)
            if response.status_code == 200:
                fee_data = response.json()
                results["fee_estimates"] = {"success": True, "data": fee_data}
                print("Successfully retrieved fee estimates")
                
                # بررسی فیلدهای مهم در تخمین کارمزدها
                if not any(key in fee_data for key in ['fastestFee', 'halfHourFee', 'hourFee', 'economyFee', 'minimumFee']):
                    print("Warning: Missing important fee estimate fields")
                else:
                    print("All required fee estimate fields are present")
            else:
                results["fee_estimates"] = {"success": False, "error": f"Status code: {response.status_code}"}
        except Exception as e:
            print(f"Error testing mempool.space fee estimates: {e}")
            results["fee_estimates"] = {"success": False, "error": str(e)}
        
        # تست دسترسی به بلاک‌های ممپول (برای هیستوگرام کارمزدها)
        try:
            blocks_url = f"{api}/v1/fees/mempool-blocks"
            response = requests.get(blocks_url)
            if response.status_code == 200:
                blocks_data = response.json()
                results["mempool_blocks"] = {"success": True, "data": blocks_data}
                print("Successfully retrieved mempool blocks (fee histogram)")
                
                # بررسی ساختار داده‌های هیستوگرام
                if not blocks_data or not isinstance(blocks_data, list) or len(blocks_data) == 0:
                    print("Warning: Mempool blocks data is empty or has wrong format")
                else:
                    print(f"Mempool blocks data contains {len(blocks_data)} projected blocks")
            else:
                results["mempool_blocks"] = {"success": False, "error": f"Status code: {response.status_code}"}
        except Exception as e:
            print(f"Error testing mempool.space mempool blocks: {e}")
            results["mempool_blocks"] = {"success": False, "error": str(e)}
    
    def test_blockstream(self):
        """تست API های blockstream.info"""
        api = self.blockstream_api
        results = self.test_results["blockstream"]
        
        # تست دسترسی به اطلاعات بلاک
        try:
            # دریافت ارتفاع آخرین بلاک
            tip_url = f"{api}/blocks/tip/height"
            response = requests.get(tip_url)
            if response.status_code == 200:
                block_height = int(response.text)
                results["block_tip"] = {"success": True, "data": block_height}
                
                # دریافت هش بلاک با ارتفاع مشخص
                height_url = f"{api}/block-height/{block_height}"
                response = requests.get(height_url)
                if response.status_code == 200:
                    block_hash = response.text
                    results["block_hash"] = {"success": True, "data": block_hash}
                    
                    # دریافت اطلاعات کامل بلاک
                    block_url = f"{api}/block/{block_hash}"
                    response = requests.get(block_url)
                    if response.status_code == 200:
                        block_data = response.json()
                        results["block_data"] = {"success": True, "data": block_data}
                        print(f"Successfully retrieved block data for height {block_height}")
                        
                        # بررسی فیلدهای مهم در داده‌های بلاک
                        fields = ['height', 'timestamp', 'tx_count', 'weight', 'version']
                        missing_fields = [f for f in fields if f not in block_data]
                        if missing_fields:
                            print(f"Warning: Missing fields in block data: {missing_fields}")
                        else:
                            print("All required block fields are present")
                    else:
                        results["block_data"] = {"success": False, "error": f"Status code: {response.status_code}"}
                else:
                    results["block_hash"] = {"success": False, "error": f"Status code: {response.status_code}"}
            else:
                results["block_tip"] = {"success": False, "error": f"Status code: {response.status_code}"}
        except Exception as e:
            print(f"Error testing blockstream block data: {e}")
            results["block_data"] = {"success": False, "error": str(e)}
        
        # تست دسترسی به تخمین کارمزدها
        try:
            fee_url = f"{api}/fee-estimates"
            response = requests.get(fee_url)
            if response.status_code == 200:
                fee_data = response.json()
                results["fee_estimates"] = {"success": True, "data": fee_data}
                print("Successfully retrieved fee estimates")
            else:
                results["fee_estimates"] = {"success": False, "error": f"Status code: {response.status_code}"}
        except Exception as e:
            print(f"Error testing blockstream fee estimates: {e}")
            results["fee_estimates"] = {"success": False, "error": str(e)}
    
    def test_coingecko(self):
        """تست API های coingecko"""
        api = self.coingecko_api
        results = self.test_results["coingecko"]
        
        # تست دسترسی به قیمت بیت‌کوین
        try:
            price_url = f"{api}/simple/price?ids=bitcoin&vs_currencies=usd"
            response = requests.get(price_url)
            if response.status_code == 200:
                price_data = response.json()
                results["bitcoin_price"] = {"success": True, "data": price_data}
                print("Successfully retrieved Bitcoin price")
                
                # بررسی ساختار داده‌ها
                if 'bitcoin' not in price_data or 'usd' not in price_data['bitcoin']:
                    print("Warning: Bitcoin price data has wrong format")
                else:
                    print(f"Current Bitcoin price: ${price_data['bitcoin']['usd']}")
            else:
                results["bitcoin_price"] = {"success": False, "error": f"Status code: {response.status_code}"}
        except Exception as e:
            print(f"Error testing CoinGecko Bitcoin price: {e}")
            results["bitcoin_price"] = {"success": False, "error": str(e)}
        
        # تست دسترسی به تاریخچه قیمت
        try:
            # قیمت‌های ۷ روز گذشته
            chart_url = f"{api}/coins/bitcoin/market_chart?vs_currency=usd&days=7"
            response = requests.get(chart_url)
            if response.status_code == 200:
                chart_data = response.json()
                results["bitcoin_market_chart"] = {"success": True, "data": chart_data}
                print("Successfully retrieved Bitcoin price history")
                
                # بررسی ساختار داده‌ها
                if 'prices' not in chart_data or not chart_data['prices'] or not isinstance(chart_data['prices'], list):
                    print("Warning: Bitcoin market chart data has wrong format")
                else:
                    print(f"Retrieved {len(chart_data['prices'])} price points")
            else:
                results["bitcoin_market_chart"] = {"success": False, "error": f"Status code: {response.status_code}"}
        except Exception as e:
            print(f"Error testing CoinGecko market chart: {e}")
            results["bitcoin_market_chart"] = {"success": False, "error": str(e)}
    
    def create_sample_record(self):
        """ایجاد یک رکورد نمونه با استفاده از داده‌های جمع‌آوری شده"""
        try:
            # بررسی اینکه آیا تست‌ها با موفقیت انجام شده‌اند
            if not self.test_results["mempool_space"].get("block_data", {}).get("success", False):
                print("Failed to create sample record: Block data not available")
                return False
            
            # استخراج داده‌ها از نتایج تست
            block_data = self.test_results["mempool_space"]["block_data"]["data"]
            mempool_stats = self.test_results["mempool_space"].get("mempool_stats", {}).get("data", {})
            fee_estimates = self.test_results["mempool_space"].get("fee_estimates", {}).get("data", {})
            mempool_blocks = self.test_results["mempool_space"].get("mempool_blocks", {}).get("data", [])
            price_data = self.test_results["coingecko"].get("bitcoin_price", {}).get("data", {})
            
            # ساخت یک رکورد نمونه
            timestamp = block_data.get('timestamp')
            dt = datetime.fromtimestamp(timestamp) if timestamp else datetime.now()
            
            record = {
                'timestamp': timestamp,
                'block_height': block_data.get('height'),
                'block_median_fee_rate': None,  # نیاز به محاسبه دارد
                'mempool_size_bytes': mempool_stats.get('vsize'),
                'mempool_tx_count': mempool_stats.get('count'),
                'mempool_min_fee_rate': fee_estimates.get('minimumFee'),
                'mempool_median_fee_rate': fee_estimates.get('halfHourFee'),
                'block_weight': block_data.get('weight'),
                'block_tx_count': block_data.get('tx_count'),
                'block_interval_seconds': None,  # نیاز به بلاک قبلی دارد
                'difficulty': None,  # نیاز به API دارد
                'hash_rate': None,  # نیاز به API دارد
                'btc_price_usd': price_data.get('bitcoin', {}).get('usd'),
                'block_version': block_data.get('version'),
                'mempool_avg_fee_rate': None,  # نیاز به محاسبه دارد
                'mempool_fee_rate_stddev': None,  # نیاز به محاسبه دارد
                'mempool_fee_histogram': str(mempool_blocks[0]) if mempool_blocks else None,
                'time_of_day': dt.hour,
                'day_of_week': dt.weekday()
            }
            
            # اضافه کردن به دیتافریم
            self.sample_df = pd.concat([self.sample_df, pd.DataFrame([record])], ignore_index=True)
            
            print("Successfully created a sample record")
            print(self.sample_df.iloc[0])
            
            return True
        except Exception as e:
            print(f"Error creating sample record: {e}")
            return False
    
    def export_sample_to_csv(self, filename="bitcoin_sample_data.csv"):
        """ذخیره دیتافریم نمونه به صورت CSV"""
        try:
            if not self.sample_df.empty:
                self.sample_df.to_csv(filename, index=False)
                print(f"Sample data saved to {filename}")
                return True
            else:
                print("No sample data to export")
                return False
        except Exception as e:
            print(f"Error saving sample data to CSV: {e}")
            return False
    
    def export_results_to_json(self, filename="api_test_detailed_results.json"):
        """ذخیره نتایج تست به صورت JSON"""
        try:
            # فیلتر کردن داده‌های بزرگ قبل از ذخیره
            filtered_results = {}
            for api_name, endpoints in self.test_results.items():
                filtered_results[api_name] = {}
                for endpoint_name, result in endpoints.items():
                    filtered_result = result.copy()
                    if 'data' in filtered_result:
                        # کوتاه کردن داده‌های بزرگ
                        if isinstance(filtered_result['data'], dict) and len(filtered_result['data']) > 5:
                            filtered_result['data'] = {k: filtered_result['data'][k] for k in list(filtered_result['data'])[:5]}
                            filtered_result['data']['...'] = '(truncated)'
                        elif isinstance(filtered_result['data'], list) and len(filtered_result['data']) > 2:
                            filtered_result['data'] = filtered_result['data'][:2]
                            filtered_result['data'].append('... (truncated)')
                    filtered_results[api_name][endpoint_name] = filtered_result
            
            with open(filename, 'w') as f:
                json.dump(filtered_results, f, indent=2)
            print(f"Test results saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving test results to JSON: {e}")
            return False
    
    def evaluate_data_availability(self):
        """ارزیابی دسترسی به داده‌های مورد نیاز"""
        required_fields = [
            'timestamp', 'block_height', 'block_median_fee_rate', 'mempool_size_bytes',
            'mempool_tx_count', 'mempool_min_fee_rate', 'mempool_median_fee_rate',
            'block_weight', 'block_tx_count', 'block_interval_seconds',
            'difficulty', 'hash_rate', 'btc_price_usd', 'block_version',
            'mempool_avg_fee_rate', 'mempool_fee_rate_stddev',
            'mempool_fee_histogram'
        ]
        
        print("\n=== Data Availability Evaluation ===")
        
        # بررسی هر فیلد
        for field in required_fields:
            # بررسی اینکه آیا این فیلد مستقیماً در دسترس است یا نیاز به محاسبه دارد
            directly_available = False
            calculable = False
            source_api = "Unknown"
            
            # بررسی داده‌های بلاک
            if field in ['timestamp', 'block_height', 'block_weight', 'block_tx_count', 'block_version']:
                if self.test_results["mempool_space"].get("block_data", {}).get("success", False):
                    block_data = self.test_results["mempool_space"]["block_data"]["data"]
                    if field in block_data or (field == 'timestamp' and 'timestamp' in block_data):
                        directly_available = True
                        source_api = "mempool.space"
                elif self.test_results["blockstream"].get("block_data", {}).get("success", False):
                    block_data = self.test_results["blockstream"]["block_data"]["data"]
                    if field in block_data or (field == 'timestamp' and 'timestamp' in block_data):
                        directly_available = True
                        source_api = "blockstream.info"
            
            # بررسی داده‌های ممپول
            elif field in ['mempool_size_bytes', 'mempool_tx_count']:
                if self.test_results["mempool_space"].get("mempool_stats", {}).get("success", False):
                    mempool_stats = self.test_results["mempool_space"]["mempool_stats"]["data"]
                    directly_available = ('vsize' in mempool_stats and field == 'mempool_size_bytes') or \
                                         ('count' in mempool_stats and field == 'mempool_tx_count')
                    source_api = "mempool.space"
            
            # بررسی تخمین کارمزدها
            elif field in ['mempool_min_fee_rate', 'mempool_median_fee_rate']:
                if self.test_results["mempool_space"].get("fee_estimates", {}).get("success", False):
                    fee_estimates = self.test_results["mempool_space"]["fee_estimates"]["data"]
                    directly_available = ('minimumFee' in fee_estimates and field == 'mempool_min_fee_rate') or \
                                        ('halfHourFee' in fee_estimates and field == 'mempool_median_fee_rate')
                    source_api = "mempool.space"
                elif self.test_results["blockstream"].get("fee_estimates", {}).get("success", False):
                    calculable = True
                    source_api = "blockstream.info (requires calculation)"
            
            # بررسی هیستوگرام کارمزدها
            elif field == 'mempool_fee_histogram':
                if self.test_results["mempool_space"].get("mempool_blocks", {}).get("success", False):
                    directly_available = True
                    source_api = "mempool.space"
            
            # بررسی قیمت بیت‌کوین
            elif field == 'btc_price_usd':
                if self.test_results["coingecko"].get("bitcoin_price", {}).get("success", False):
                    directly_available = True
                    source_api = "coingecko"
            
            # فیلدهایی که نیاز به محاسبه دارند
            elif field in ['block_median_fee_rate', 'block_interval_seconds', 'mempool_avg_fee_rate', 'mempool_fee_rate_stddev']:
                calculable = True
                source_api = "Requires calculation from available data"
            
            # فیلدهایی که در حال حاضر در دسترس نیستند
            elif field in ['difficulty', 'hash_rate']:
                calculable = True
                source_api = "Requires additional API call (mempool.space)"
            
            # وضعیت دسترسی
            status = "✓ Available" if directly_available else ("⚠ Calculable" if calculable else "✗ Not available")
            
            print(f"{field}: {status} (Source: {source_api})")
        
        # خلاصه وضعیت
        print("\n=== Summary ===")
        print(f"Total fields required: {len(required_fields)}")
        print(f"Recommended primary API: mempool.space")
        print(f"Recommended price API: coingecko")
        if not self.sample_df.empty:
            print(f"Sample record created successfully")
            available_fields = sum(1 for field in required_fields if field in self.sample_df.columns and self.sample_df.iloc[0][field] is not None)
            print(f"Fields with data in sample: {available_fields}/{len(required_fields)}")
    
    def run_all_tests(self):
        """اجرای تمام تست‌ها"""
        print("=== Starting Bitcoin Data API Tests ===")
        print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\n=== Testing mempool.space API ===")
        self.test_mempool_space()
        time.sleep(1)  # تأخیر بین API calls
        
        print("\n=== Testing blockstream.info API ===")
        self.test_blockstream()
        time.sleep(1)  # تأخیر بین API calls
        
        print("\n=== Testing coingecko API ===")
        self.test_coingecko()
        
        print("\n=== Creating sample record ===")
        self.create_sample_record()
        
        print("\n=== Evaluating data availability ===")
        self.evaluate_data_availability()
        
        print("\n=== Exporting results ===")
        self.export_results_to_json()
        if not self.sample_df.empty:
            self.export_sample_to_csv()
        
        print("\nAll tests completed.")

if __name__ == "__main__":
    tester = BitcoinDataTest()
    tester.run_all_tests() 