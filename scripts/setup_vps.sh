#!/bin/bash

# Installation and setup script for Bitcoin Data Collector on VPS

# Set variables
USERNAME=$(whoami)
PROJECT_DIR=$(pwd)
SERVICE_NAME="btc-collector@${USERNAME}"

echo "Starting setup for Bitcoin Data Collector..."
echo "Installing required packages..."

# Install required packages
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv

# Create Python virtual environment
echo "Creating Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p data logs

# Copy service file to systemd location
echo "Installing systemd service..."
sudo cp config/btc-collector.service /etc/systemd/system/btc-collector@.service

# Start service
echo "Starting service..."
sudo systemctl daemon-reload
sudo systemctl enable $SERVICE_NAME
sudo systemctl start $SERVICE_NAME

echo "Service installation completed."
echo "To check service status, run: sudo systemctl status $SERVICE_NAME"
echo "To view logs, run: tail -f logs/collector_*.log"
echo "Data will be saved in the data/ directory" 