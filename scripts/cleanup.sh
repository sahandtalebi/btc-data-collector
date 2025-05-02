#!/bin/bash

# Cleanup script for removing unused files

echo "Starting cleanup process..."

# Old version files
echo "Removing old version files..."
rm -f bitcoin_data_collector.py 
rm -f bitcoin_data_collector_v2.py
rm -f btc_collector_service.py

# Test files
echo "Removing test files..."
rm -f test.py
rm -f test_collector.py
rm -f test_height_range.csv
rm -f test_single_block.csv
rm -f bitcoin_sample_data.csv
rm -f bitcoin_data_test.py 
rm -f bitcoin_api_tester.py

# Test data files
echo "Removing test data files..."
rm -f api_test_detailed_results.json
rm -f api_test_results.json

# Old config files
echo "Removing old config files..."
rm -f btc-collector.service
rm -f README_VPS.md 
rm -f setup_vps.sh

# Mac OS files
echo "Removing Mac OS files..."
rm -rf __MACOSX

echo "Cleanup completed."
echo "The project structure is now clean and ready for GitHub." 