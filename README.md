# Kalshi Trading Bot 🎯

A Python-based automated trading bot for weather derivatives on [Kalshi](https://kalshi.com). Features live P&L tracking, forecast fusion, risk management, and portfolio analytics.

## Features

- **Automated Trading**: Real-time decision making based on weather forecasts and market prices
- **Forecast Fusion**: Combines multiple weather sources (NWS, OpenMeteo, ensemble methods)
- **Risk Management**: Portfolio-level risk constraints, position sizing, concentration limits
- **Order Management**: Smart order amendment for improved fill execution
- **P&L Tracking**: Real-time profit/loss analytics with daily breakdown
- **Forecast Accuracy**: Calibration tracking and forecast outcome resolution
- **Correlation Hedging**: Portfolio correlation analysis to reduce systematic risk
- **Greeks-lite**: Delta exposure and volatility detection
- **Live Dashboard**: Streamlit dashboard for performance visualization

## Setup

### Prerequisites
- Python 3.10+
- pip / conda
- Git

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/kalshi-trading-bot.git
cd kalshi-trading-bot

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

1. Create `.env` file with your settings:
```bash
# API Keys
KALSHI_API_KEY=your_key_here
KALSHI_API_SECRET=your_secret_here

# Trading Parameters
MAX_TOTAL_RISK_DOLLARS=60.0
TRADE_STEP=1
BASE_TARGET=1
MAX_TARGET=2
MIN_EDGE_CENTS=0.5

# Feature Toggles
ENABLE_CORRELATION_HEDGE=true
ENABLE_ORDER_AMENDING=true
ENABLE_GREEK_ALERTS=true
```

2. Place API keys in `keys/` directory (never commit!)

## Running

### Main Bot
```bash
python src/run_bot.py
```

### Dashboard (Streamlit)
```bash
streamlit run src/dashboard.py
```

### Diagnostics
```bash
python src/check_pnl.py           # P&L analysis
python src/check_fills.py         # Trade statistics
python src/check_forecast.py      # Forecast accuracy
python src/check_correlation.py   # Portfolio diversification
python src/check_balance.py       # Account balance
python src/check_greeks.py        # Portfolio Greeks
python src/check_trading_activity.py  # Recent trades
```

## Project Structure

```
├── src/
│   ├── run_bot.py              # Main trading loop
│   ├── db.py                   # Database schema & analytics
│   ├── kalshi_client.py        # Kalshi API client
│   ├── weather_sources.py      # Forecast providers
│   ├── nws.py                  # NWS integration
│   ├── dashboard.py            # Streamlit analytics UI
│   ├── check_*.py              # Diagnostic scripts
│   ├── liquidate_positions.py  # Emergency position liquidation
│   └── backfill_climo.py       # Historical data backfill
├── data/
│   └── kalshi_quotes.sqlite    # SQLite database (trading data)
├── keys/                        # API credentials (DO NOT COMMIT)
├── .env                         # Configuration (DO NOT COMMIT)
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Database Schema

- **quotes**: Market price snapshots (bid/ask)
- **trades**: Executed trades (buys/sells)
- **positions**: Current holdings by ticker
- **forecast_predictions**: Model predictions
- **provider_readings**: Raw weather data
- **outcomes**: Resolved market outcomes
- **correlations**: Position correlations

## Key Concepts

### Edge Calculation
The bot identifies profitable trades by computing the expected value vs. market prices. Only trades with edge > MIN_EDGE_CENTS are executed.

### Risk Management
- `MAX_TOTAL_RISK_DOLLARS`: Portfolio-level risk cap
- `MAX_ABS_QTY_PER_TICKER`: Concentration limit per market
- `PERSIST_SECONDS`: Minimum hold time to reduce noise

### Forecast Accuracy
Predictions are tracked and compared to actual outcomes for model calibration. Accuracy metrics by provider/series help tune parameters.

### P&L Components
- **Realized**: Closed positions (locked-in gains/losses)
- **Unrealized**: Open positions marked-to-market
- **Daily breakdown**: Performance by trading date

## Deployment

### Streamlit Cloud
1. Push code to GitHub
2. Go to [streamlit.io](https://streamlit.io)
3. Connect repo and deploy `src/dashboard.py`
4. Add secrets in Streamlit dashboard settings

### AWS / Azure
See `DEPLOYMENT.md` for cloud deployment options.

## Performance Metrics

Real-time metrics displayed on dashboard:
- **Sharpe Ratio**: Risk-adjusted returns (annualized)
- **Win Rate**: % of profitable trades
- **Portfolio Risk**: Current risk usage vs. limit
- **Equity Curve**: Cumulative P&L over time
- **Daily P&L**: Breakdown by date

## Trading Logic

1. **Forecast Fusion**: Combine multiple weather forecasts
2. **Market Analysis**: Get current quotes and compute fair value
3. **Candidate Generation**: Identify mispricings vs. forecasts
4. **Risk Checks**: Verify position doesn't exceed constraints
5. **Order Execution**: Place limit orders via Kalshi API
6. **Order Management**: Monitor and amend stale orders
7. **P&L Tracking**: Log trades and update positions

## Safety Features

- **Emergency Liquidation**: `liquidate_positions.py` for rapid exit
- **Stale Quote Detection**: Skip trades when prices are old
- **Liquidity Checks**: Verify bid-ask depth before trading
- **Manual Controls**: Easy parameter tuning via `.env`

## Troubleshooting

### Bot not trading?
```bash
python src/check_greeks.py      # Check portfolio Greeks
python src/check_trading_activity.py  # Verify recent trades
python src/check_balance.py     # Confirm account funding
```

### Database corruption?
```bash
rm data/kalshi_quotes.sqlite*
# Bot will reinitialize on next run
```

### API connection issues?
- Verify `KALSHI_API_KEY` and `KALSHI_API_SECRET` in `.env`
- Check network connectivity and Kalshi API status
- Review logs in terminal output

## Contributing

Contributions welcome! Please:
1. Create feature branch
2. Add tests for new functionality
3. Update docs and README
4. Submit pull request

## Monitoring

Real-time alerts and diagnostics already built in:
- Volatility spike detection
- Portfolio concentration warnings
- Fill quality analytics
- Forecast calibration tracking

## License

MIT License - See LICENSE file

## Disclaimer

This bot trades real capital on Kalshi. Use at your own risk. Past performance is not indicative of future results. Always test thoroughly in paper trading first.

## Support

For issues or questions:
1. Check diagnostics: `python src/check_*.py`
2. Review logs in terminal
3. Check `.env` configuration
4. Open GitHub issue with details

---

**Last Updated**: February 2026  
**Status**: Active (Trading Live)  
**Account**: ~$20 with $60 risk limit
