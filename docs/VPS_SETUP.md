# VPS Installation and Setup Guide

This guide will help you install and set up the Bitcoin Data Collector on a VPS server.

## Requirements

- VPS with Linux operating system (Ubuntu or Debian recommended)
- Minimum 1GB RAM
- Minimum 20GB disk space (for long-term data storage)
- Root or sudo access
- Python 3.6+

## Installation Steps

### 1. Prepare the Server

First, update your server:

```bash
sudo apt-get update
sudo apt-get upgrade -y
```

### 2. Install Git (if not installed)

```bash
sudo apt-get install git -y
```

### 3. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/btc-data-collector.git
cd btc-data-collector
```

### 4. Run the Installation Script

```bash
chmod +x scripts/setup_vps.sh
./scripts/setup_vps.sh
```

This script automatically:
- Installs required packages
- Creates a Python virtual environment
- Installs Python dependencies
- Sets up and starts the systemd service

## Service Management

### Check Service Status

```bash
sudo systemctl status btc-collector@$USER
```

### Restart Service

```bash
sudo systemctl restart btc-collector@$USER
```

### Stop Service

```bash
sudo systemctl stop btc-collector@$USER
```

### Disable Automatic Start at System Boot

```bash
sudo systemctl disable btc-collector@$USER
```

## Viewing Logs

### Service Logs

```bash
tail -f logs/collector_*.log
```

### Systemd Logs

```bash
sudo journalctl -u btc-collector@$USER -f
```

## Testing the Service

To test the service without starting it as a service, you can run the script directly:

```bash
source .venv/bin/activate
python scripts/btc_collector_service.py
```

Press Ctrl+C to stop it.

## Troubleshooting

### Problem 1: Service Does Not Start

Check systemd logs:

```bash
sudo journalctl -u btc-collector@$USER -n 50
```

### Problem 2: Data Is Not Being Collected

Check application logs:

```bash
tail -n 100 logs/collector_*.log
```

### Problem 3: API Rate Limiting

If you encounter API rate limiting, you can edit the `scripts/btc_collector_service.py` file and increase the `SAVE_INTERVAL` value to create more delay between requests.

## Advanced Settings

### Change Data Collection Time Range

To change the time range from 6 months to another value, edit the `scripts/btc_collector_service.py` file and modify the `timedelta(days=180)` value.

### Change Interval Between Collection Cycles

By default, after completing a collection cycle, the service waits 24 hours. To change this value, edit the `scripts/btc_collector_service.py` file and modify the `time.sleep(24 * 3600)` value.

## Data Backup

It is recommended to periodically back up your collected data:

```bash
# Compress the data directory
tar -czvf btc_data_backup_$(date +%Y%m%d).tar.gz data/

# Transfer to another system
scp btc_data_backup_*.tar.gz user@your-backup-server:/path/to/backups/
``` 