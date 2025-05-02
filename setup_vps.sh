#!/bin/bash

# اسکریپت نصب و راه‌اندازی جمع‌کننده داده‌های بیت‌کوین در VPS

# تنظیم متغیرها
USERNAME=$(whoami)
PROJECT_DIR=$(pwd)
SERVICE_NAME="btc-collector"

echo "Starting setup for Bitcoin Data Collector..."
echo "Installing required packages..."

# نصب پکیج‌های مورد نیاز
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv

# ایجاد محیط مجازی پایتون
echo "Creating Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# نصب وابستگی‌ها
echo "Installing Python dependencies..."
pip install -r requirements.txt

# ایجاد دایرکتوری‌های مورد نیاز
echo "Creating necessary directories..."
mkdir -p data logs

# ویرایش فایل سرویس - جایگزینی نام کاربری
echo "Configuring systemd service file..."
sed -i "s/root/$USERNAME/g" btc-collector.service

# کپی فایل سرویس به محل systemd
echo "Installing systemd service..."
sudo cp btc-collector.service /etc/systemd/system/

# راه‌اندازی سرویس
echo "Starting service..."
sudo systemctl daemon-reload
sudo systemctl enable $SERVICE_NAME
sudo systemctl start $SERVICE_NAME

echo "Service installation completed."
echo "To check service status, run: sudo systemctl status $SERVICE_NAME"
echo "To view logs, run: tail -f logs/collector_*.log"
echo "Data will be saved in the data/ directory" 