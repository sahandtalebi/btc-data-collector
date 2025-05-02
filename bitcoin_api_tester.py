#!/usr/bin/env python3
import requests
import json
import time
from datetime import datetime
import pandas as pd

class BitcoinAPITester:
    def __init__(self):
        # لیست API‌های مورد آزمایش
        self.apis = {
            "mempool.space": {
                "base_url": "https://mempool.space/api",
                "endpoints": {
                    "block_tip": "/blocks/tip/height",
                    "block_by_height": "/block-height/{height}",
                    "block_by_hash": "/block/{hash}",
                    "mempool_stats": "/mempool",
                    "fee_estimates": "/v1/fees/recommended",
                    "difficulty": "/v1/difficulty-adjustment",
                    "mempool_blocks": "/v1/fees/mempool-blocks"
                }
            },
            "blockstream.info": {
                "base_url": "https://blockstream.info/api",
                "endpoints": {
                    "block_tip": "/blocks/tip/height",
                    "block_by_height": "/block-height/{height}",
                    "block_by_hash": "/block/{hash}",
                    "mempool_stats": "/mempool",
                    "fee_estimates": "/fee-estimates"
                }
            },
            "blockchain.info": {
                "base_url": "https://blockchain.info",
                "endpoints": {
                    "block_by_height": "/block-height/{height}?format=json",
                    "mempool_stats": "/unconfirmed-transactions?format=json",
                    "fee_estimates": "/mempool/fees"
                }
            },
            "coingecko": {
                "base_url": "https://api.coingecko.com/api/v3",
                "endpoints": {
                    "bitcoin_price": "/simple/price?ids=bitcoin&vs_currencies=usd",
                    "bitcoin_market_chart": "/coins/bitcoin/market_chart?vs_currency=usd&days=1"
                }
            },
            "bitgo": {
                "base_url": "https://www.bitgo.com/api/v2",
                "endpoints": {
                    "fee_estimates": "/btc/tx/fee"
                }
            }
        }
        
        self.results = {}
    
    def test_api_endpoint(self, api_name, endpoint_name, url, params=None):
        """تست یک نقطه پایانی API خاص"""
        start_time = time.time()
        success = False
        status_code = None
        response_data = None
        error_msg = None
        
        try:
            full_url = url
            if params:
                for key, value in params.items():
                    full_url = full_url.replace(f"{{{key}}}", str(value))
            
            print(f"Testing {api_name} - {endpoint_name}: {full_url}")
            response = requests.get(full_url, timeout=10)
            status_code = response.status_code
            
            if status_code == 200:
                success = True
                try:
                    response_data = response.json()
                except:
                    response_data = response.text
            else:
                error_msg = f"HTTP Error: {status_code}"
        except Exception as e:
            error_msg = str(e)
        
        elapsed_time = time.time() - start_time
        
        result = {
            "success": success,
            "status_code": status_code,
            "response_time": elapsed_time,
            "error": error_msg,
            "data_sample": self._get_data_sample(response_data)
        }
        
        return result
    
    def _get_data_sample(self, data):
        """دریافت یک نمونه کوچک از داده‌ها برای نمایش"""
        if data is None:
            return None
        
        if isinstance(data, dict):
            # فقط حداکثر ۵ کلید اول را نمایش می‌دهیم
            sample = {}
            for i, (key, value) in enumerate(data.items()):
                if i >= 5:
                    sample["..."] = "..."
                    break
                sample[key] = self._truncate_value(value)
            return sample
        
        if isinstance(data, list):
            # فقط اولین آیتم را نمایش می‌دهیم
            if len(data) > 0:
                first_item = data[0]
                if isinstance(first_item, dict):
                    return {"first_item": self._get_data_sample(first_item), "count": len(data)}
                else:
                    return {"first_item": self._truncate_value(first_item), "count": len(data)}
            return {"count": 0}
        
        return self._truncate_value(data)
    
    def _truncate_value(self, value):
        """کوتاه کردن مقادیر بزرگ برای نمایش"""
        if isinstance(value, str) and len(value) > 100:
            return value[:100] + "..."
        return value
    
    def run_tests(self):
        """اجرای تست‌ها برای تمام API‌ها"""
        for api_name, api_info in self.apis.items():
            print(f"\n=== Testing {api_name} API ===")
            self.results[api_name] = {}
            
            base_url = api_info["base_url"]
            for endpoint_name, endpoint_path in api_info["endpoints"].items():
                url = base_url + endpoint_path
                
                # پارامترهای پیش‌فرض برای برخی از نقاط پایانی
                params = None
                if "{height}" in url:
                    params = {"height": "780000"}  # یک بلاک نسبتاً جدید
                elif "{hash}" in url:
                    # نیاز به دریافت هش بلاک
                    if api_name == "mempool.space":
                        try:
                            block_height_url = base_url + api_info["endpoints"]["block_by_height"].replace("{height}", "780000")
                            block_hash = requests.get(block_height_url).text
                            params = {"hash": block_hash}
                        except:
                            params = {"hash": "000000000000000000000d9c5b244db93ed1ad30f5ed14d2db182bac3f2468f6"}  # هش یک بلاک معروف
                    else:
                        params = {"hash": "000000000000000000000d9c5b244db93ed1ad30f5ed14d2db182bac3f2468f6"}
                
                result = self.test_api_endpoint(api_name, endpoint_name, url, params)
                self.results[api_name][endpoint_name] = result
                
                # اضافه کردن تأخیر بین درخواست‌ها برای جلوگیری از محدودیت نرخ
                time.sleep(1)
    
    def print_results(self):
        """چاپ نتایج تست‌ها"""
        print("\n\n=== API Test Results ===")
        
        for api_name, endpoints in self.results.items():
            print(f"\n{api_name}:")
            successful_endpoints = 0
            total_endpoints = len(endpoints)
            avg_response_time = 0
            
            for endpoint_name, result in endpoints.items():
                success_marker = "✓" if result["success"] else "✗"
                response_time = f"{result['response_time']:.2f}s"
                print(f"  {success_marker} {endpoint_name}: {response_time}")
                
                if result["success"]:
                    successful_endpoints += 1
                    avg_response_time += result["response_time"]
            
            if successful_endpoints > 0:
                avg_response_time /= successful_endpoints
                
            success_rate = (successful_endpoints / total_endpoints) * 100
            print(f"  Success Rate: {success_rate:.1f}% ({successful_endpoints}/{total_endpoints})")
            print(f"  Avg Response Time: {avg_response_time:.2f}s")
    
    def export_to_json(self, filename="api_test_results.json"):
        """ذخیره نتایج به صورت JSON"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults exported to {filename}")
    
    def recommend_best_apis(self):
        """پیشنهاد بهترین API‌ها براساس نتایج"""
        api_scores = {}
        
        for api_name, endpoints in self.results.items():
            successful_endpoints = 0
            total_endpoints = len(endpoints)
            avg_response_time = 0
            data_completeness = 0
            
            for endpoint_name, result in endpoints.items():
                if result["success"]:
                    successful_endpoints += 1
                    avg_response_time += result["response_time"]
                    
                    # بررسی کامل بودن داده‌ها (ساده)
                    if result["data_sample"] is not None:
                        data_completeness += 1
            
            if successful_endpoints > 0:
                avg_response_time /= successful_endpoints
                
            success_rate = (successful_endpoints / total_endpoints)
            completeness_rate = data_completeness / total_endpoints
            
            # محاسبه امتیاز کلی (وزن بیشتر برای نرخ موفقیت، سپس کامل بودن، سپس زمان پاسخ)
            # تبدیل زمان پاسخ به امتیاز بین 0 تا 1 (زمان کمتر = امتیاز بیشتر)
            response_time_score = 1.0 if avg_response_time == 0 else min(1.0, 5.0 / avg_response_time)
            
            score = (0.5 * success_rate) + (0.3 * completeness_rate) + (0.2 * response_time_score)
            api_scores[api_name] = {
                "score": score,
                "success_rate": success_rate,
                "avg_response_time": avg_response_time,
                "data_completeness": completeness_rate
            }
        
        # مرتب‌سازی API‌ها براساس امتیاز کلی
        sorted_apis = sorted(api_scores.items(), key=lambda x: x[1]["score"], reverse=True)
        
        print("\n=== Recommended APIs ===")
        for i, (api_name, metrics) in enumerate(sorted_apis):
            print(f"{i+1}. {api_name}")
            print(f"   Overall Score: {metrics['score']:.2f}")
            print(f"   Success Rate: {metrics['success_rate']*100:.1f}%")
            print(f"   Avg Response Time: {metrics['avg_response_time']:.2f}s")
            print(f"   Data Completeness: {metrics['data_completeness']*100:.1f}%")
            print()

if __name__ == "__main__":
    print("Starting Bitcoin API Test...")
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tester = BitcoinAPITester()
    tester.run_tests()
    tester.print_results()
    tester.export_to_json()
    tester.recommend_best_apis() 