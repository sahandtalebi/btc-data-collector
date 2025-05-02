#!/bin/bash

# GitHub repository initialization script for the Bitcoin Data Collector project

echo "Starting GitHub Repository Initialization..."

# Remove unnecessary files that shouldn't be pushed to GitHub
# (test files, data, and other non-essential files)
echo "Cleaning up unnecessary files..."

# Remove old test/unnecessary files
rm -f test_height_range.csv test_single_block.csv bitcoin_sample_data.csv
rm -f api_test_detailed_results.json api_test_results.json
rm -f test_collector.py bitcoin_data_test.py bitcoin_api_tester.py
rm -f test.py btc_collector_service.py btc-collector.service
rm -f README_VPS.md setup_vps.sh
rm -rf __MACOSX

# Initialize git
echo "Initializing Git repository..."
git init

# Add files to git
echo "Adding files to Git..."
git add .

# Create LICENSE file
echo "Creating MIT License file..."
cat > LICENSE << 'EOL'
MIT License

Copyright (c) 2024 YOUR_NAME

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOL

git add LICENSE

# Initial commit
echo "Creating initial commit..."
git commit -m "Initial commit - Bitcoin Data Collector"

echo ""
echo "Repository initialized successfully!"
echo ""
echo "To connect to GitHub, create a new repository at https://github.com/new"
echo "Then run the following commands:"
echo ""
echo "git remote add origin https://github.com/YOUR_USERNAME/btc-data-collector.git"
echo "git branch -M main"
echo "git push -u origin main"
echo ""
echo "Don't forget to replace YOUR_USERNAME with your actual GitHub username." 