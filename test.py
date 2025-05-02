#!/usr/bin/env python3
import requests
import json
from datetime import datetime, timedelta

def test_mempool_api():
    """Test the Mempool.space API endpoints"""
    print("Testing Mempool.space API...")
    
    # Base URL
    base_url = "https://mempool.space/api"
    
    # Test endpoints
    endpoints = [
        "/blocks/tip/height",  # Get current block height
        "/mempool",  # Get mempool info
        "/v1/fees/recommended",  # Get fee estimates
        "/v1/difficulty-adjustment"  # Get difficulty info
    ]
    
    for endpoint in endpoints:
        url = base_url + endpoint
        print(f"Testing endpoint: {url}")
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json() if endpoint != "/blocks/tip/height" else response.text
                print(f"Success! Response: {data if endpoint == '/blocks/tip/height' else '[data returned]'}")
            else:
                print(f"Error: Status code {response.status_code}")
        except Exception as e:
            print(f"Exception: {str(e)}")
        print("---")
    
    # Test block endpoint with a specific height
    current_height = int(requests.get(base_url + "/blocks/tip/height").text)
    test_height = current_height - 10  # Use a block that's already confirmed
    
    print(f"Testing block-height endpoint with height {test_height}")
    try:
        response = requests.get(f"{base_url}/block-height/{test_height}")
        if response.status_code == 200:
            block_hash = response.text
            print(f"Block hash: {block_hash}")
            
            # Now get block details
            block_response = requests.get(f"{base_url}/block/{block_hash}")
            if block_response.status_code == 200:
                block_data = block_response.json()
                print(f"Successfully retrieved block {test_height} data")
                print(f"Timestamp: {datetime.fromtimestamp(block_data.get('timestamp', 0))}")
                print(f"Number of transactions: {block_data.get('tx_count', 0)}")
            else:
                print(f"Error getting block details: {block_response.status_code}")
        else:
            print(f"Error: Status code {response.status_code}")
    except Exception as e:
        print(f"Exception: {str(e)}")
    print("---")

def test_coingecko_api():
    """Test the CoinGecko API endpoints"""
    print("Testing CoinGecko API...")
    
    # Base URL
    base_url = "https://api.coingecko.com/api/v3"
    
    # Test endpoint for Bitcoin price
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    # Convert to Unix timestamps
    start_timestamp = int(start_date.timestamp())
    end_timestamp = int(end_date.timestamp())
    
    url = f"{base_url}/coins/bitcoin/market_chart/range"
    params = {
        "vs_currency": "usd",
        "from": start_timestamp,
        "to": end_timestamp
    }
    
    print(f"Testing endpoint: {url} with params {params}")
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            print(f"Success! Got {len(data.get('prices', []))} price data points")
            
            # Print first and last price point
            if data.get('prices'):
                first_price = data['prices'][0]
                last_price = data['prices'][-1]
                print(f"First price point: {datetime.fromtimestamp(first_price[0]/1000)}: ${first_price[1]}")
                print(f"Last price point: {datetime.fromtimestamp(last_price[0]/1000)}: ${last_price[1]}")
        else:
            print(f"Error: Status code {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Exception: {str(e)}")
    print("---")

if __name__ == "__main__":
    print("Starting API tests...")
    test_mempool_api()
    test_coingecko_api()
    print("Tests completed.") 