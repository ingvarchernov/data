# Fast Indicators

Rust-based technical indicators for high-performance trading analysis.

## Features

- SMA (Simple Moving Average)
- EMA (Exponential Moving Average) 
- RSI (Relative Strength Index)
- ATR (Average True Range)
- OBV (On-Balance Volume)
- VWAP (Volume Weighted Average Price)
- And more...

## Usage

```python
import fast_indicators as fi
import numpy as np

# Calculate SMA
prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
sma_result = fi.sma(prices, 3)
```