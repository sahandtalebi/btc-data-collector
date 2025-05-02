# Bitcoin Network Data Collector

This repository contains tools to automatically collect data from the Bitcoin network. With these tools, you can gather data about blocks, mempool, fees, and Bitcoin price.

## Features

- Collect data from the Bitcoin network using APIs
- Extract detailed information about fees, mempool size, and block characteristics
- Collect data based on block height range or date range
- Periodic saving to prevent data loss
- Automatic service setup on VPS

## Requirements

- Python 3.6+
- Internet connection to fetch data from APIs

## Installation

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Project Structure

```
├── config/               # Configuration files
│   └── btc-collector.service  # systemd service file
├── data/                 # Collected data
├── logs/                 # Log files
├── scripts/              # Executable scripts
│   ├── btc_collector_service.py  # Data collection service
│   └── setup_vps.sh      # Automatic installation script
├── src/                  # Source code
│   └── collector/        # Data collection modules
│       └── bitcoin_data_collector.py  # Main data collector class
├── .gitignore
├── README.md
└── requirements.txt
```

## Usage

### Collect Data by Block Height

To collect data for a specific range of blocks:

```bash
python src/collector/bitcoin_data_collector.py --mode blocks --start_height 700000 --end_height 701000 --output data/blocks_data.csv
```

### Collect Data by Date Range

To collect data for a specific date range:

```bash
python src/collector/bitcoin_data_collector.py --mode dates --start_date 2022-01-01 --end_date 2022-06-30 --output data/date_range_data.csv
```

### Run Automatic Collection Service (Last 6 Months)

To run the automatic data collection service for the last 6 months:

```bash
python scripts/btc_collector_service.py
```

### Install on VPS

To install and run the service on a VPS, follow these steps:

1. Clone the code to your VPS:
```bash
git clone https://github.com/YOUR_USERNAME/btc-data-collector.git
cd btc-data-collector
```

2. Run the installation script:
```bash
chmod +x scripts/setup_vps.sh
./scripts/setup_vps.sh
```

For more details on VPS installation and usage, see [VPS Setup Guide](docs/VPS_SETUP.md).

## Data Format

The collected data is saved in CSV format with the following columns:

- timestamp: Block timestamp
- block_height: Block height
- block_median_fee_rate: Block median fee rate (satoshis/vbyte)
- mempool_size_bytes: Mempool size (bytes)
- mempool_tx_count: Number of pending transactions in mempool
- mempool_min_fee_rate: Minimum fee rate recommended for mempool acceptance
- mempool_median_fee_rate: Median fee rate in mempool
- and other columns...

## API Sources

This project uses the following APIs:

- Mempool.space API: For block and mempool information
- CoinGecko API: For Bitcoin price data

## GitHub Setup

To prepare this project for GitHub with a professional structure, use the included script:

```bash
chmod +x scripts/init_github.sh
./scripts/init_github.sh
```

This script:
1. Removes unnecessary and test files
2. Creates a new git repository
3. Creates an MIT license file
4. Makes the initial commit
5. Displays the commands needed to connect to GitHub

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for more information. 