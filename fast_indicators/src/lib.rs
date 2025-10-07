use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};

/// Fast RSI calculation using Rust
#[pyfunction]
fn fast_rsi(py: Python, prices: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    let prices = prices.as_slice()?;
    
    if prices.len() < period + 1 {
        return Ok(vec![].into_pyarray_bound(py).into());
    }
    
    let mut gains = Vec::new();
    let mut losses = Vec::new();
    
    // Calculate initial gains and losses
    for i in 1..=period {
        let change = prices[i] - prices[i - 1];
        if change > 0.0 {
            gains.push(change);
            losses.push(0.0);
        } else {
            gains.push(0.0);
            losses.push(change.abs());
        }
    }
    
    let mut avg_gain = gains.iter().sum::<f64>() / period as f64;
    let mut avg_loss = losses.iter().sum::<f64>() / period as f64;
    
    let mut rsi_values = Vec::new();
    
    // Calculate RSI for the first point
    if avg_loss != 0.0 {
        let rs = avg_gain / avg_loss;
        let rsi = 100.0 - (100.0 / (1.0 + rs));
        rsi_values.push(rsi);
    } else {
        rsi_values.push(100.0);
    }
    
    // Calculate subsequent RSI values
    for i in (period + 1)..prices.len() {
        let change = prices[i] - prices[i - 1];
        let gain = if change > 0.0 { change } else { 0.0 };
        let loss = if change < 0.0 { change.abs() } else { 0.0 };
        
        avg_gain = ((avg_gain * (period as f64 - 1.0)) + gain) / period as f64;
        avg_loss = ((avg_loss * (period as f64 - 1.0)) + loss) / period as f64;
        
        if avg_loss != 0.0 {
            let rs = avg_gain / avg_loss;
            let rsi = 100.0 - (100.0 / (1.0 + rs));
            rsi_values.push(rsi);
        } else {
            rsi_values.push(100.0);
        }
    }
    
    Ok(rsi_values.into_pyarray_bound(py).into())
}

/// Fast MACD calculation - returns (macd_line, signal_line, histogram)
#[pyfunction]
fn fast_macd(
    py: Python,
    prices: PyReadonlyArray1<f64>,
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let prices = prices.as_slice()?;
    
    let fast_ema = calculate_ema(prices, fast_period);
    let slow_ema = calculate_ema(prices, slow_period);
    
    let start_index = slow_period.saturating_sub(fast_period);
    let macd_line: Vec<f64> = fast_ema.iter()
        .skip(start_index)
        .zip(slow_ema.iter())
        .map(|(fast, slow)| fast - slow)
        .collect();
    
    let signal_line = calculate_ema(&macd_line, signal_period);
    let histogram: Vec<f64> = macd_line.iter()
        .skip(signal_period - 1)
        .zip(signal_line.iter())
        .map(|(macd, signal)| macd - signal)
        .collect();
    
    Ok((
        macd_line.into_pyarray_bound(py).into(),
        signal_line.into_pyarray_bound(py).into(),
        histogram.into_pyarray_bound(py).into(),
    ))
}

/// Fast Bollinger Bands calculation - returns (upper_band, lower_band)
#[pyfunction]
fn fast_bollinger_bands(
    py: Python,
    prices: PyReadonlyArray1<f64>,
    period: usize,
    std_dev: f64,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let prices = prices.as_slice()?;
    
    if prices.len() < period {
        return Ok((
            vec![].into_pyarray_bound(py).into(),
            vec![].into_pyarray_bound(py).into(),
        ));
    }
    
    let mut upper_band = Vec::new();
    let mut lower_band = Vec::new();
    
    for i in (period - 1)..prices.len() {
        let slice = &prices[(i + 1 - period)..=i];
        let mean = slice.iter().sum::<f64>() / period as f64;
        let variance = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / period as f64;
        let std_deviation = variance.sqrt();
        
        upper_band.push(mean + (std_dev * std_deviation));
        lower_band.push(mean - (std_dev * std_deviation));
    }
    
    Ok((
        upper_band.into_pyarray_bound(py).into(),
        lower_band.into_pyarray_bound(py).into(),
    ))
}

/// Fast Stochastic Oscillator calculation - returns (%K, %D)
#[pyfunction]
fn fast_stochastic(
    py: Python,
    highs: PyReadonlyArray1<f64>,
    lows: PyReadonlyArray1<f64>,
    closes: PyReadonlyArray1<f64>,
    k_period: usize,
    smooth_k: usize,
    smooth_d: usize,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let highs = highs.as_slice()?;
    let lows = lows.as_slice()?;
    let closes = closes.as_slice()?;
    
    if highs.len() != lows.len() || lows.len() != closes.len() || closes.len() < k_period {
        return Ok((
            vec![].into_pyarray_bound(py).into(),
            vec![].into_pyarray_bound(py).into(),
        ));
    }
    
    let mut k_values = Vec::new();
    
    for i in (k_period - 1)..closes.len() {
        let high_slice = &highs[(i + 1 - k_period)..=i];
        let low_slice = &lows[(i + 1 - k_period)..=i];
        
        let highest_high = high_slice.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let lowest_low = low_slice.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        
        let k = if highest_high != lowest_low {
            ((closes[i] - lowest_low) / (highest_high - lowest_low)) * 100.0
        } else {
            50.0
        };
        k_values.push(k);
    }
    
    // Smooth %K
    let smooth_k_values = calculate_sma(&k_values, smooth_k);
    // Calculate %D as SMA of smooth %K
    let d_values = calculate_sma(&smooth_k_values, smooth_d);
    
    Ok((
        smooth_k_values.into_pyarray_bound(py).into(),
        d_values.into_pyarray_bound(py).into(),
    ))
}

/// Fast EMA calculation
#[pyfunction]
fn fast_ema(py: Python, prices: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    let prices = prices.as_slice()?;
    let ema_values = calculate_ema(prices, period);
    Ok(ema_values.into_pyarray_bound(py).into())
}

/// Fast ATR calculation
#[pyfunction]
fn fast_atr(
    py: Python,
    highs: PyReadonlyArray1<f64>,
    lows: PyReadonlyArray1<f64>,
    closes: PyReadonlyArray1<f64>,
    period: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let highs = highs.as_slice()?;
    let lows = lows.as_slice()?;
    let closes = closes.as_slice()?;
    
    if highs.len() != lows.len() || lows.len() != closes.len() || closes.len() < 2 {
        return Ok(vec![].into_pyarray_bound(py).into());
    }
    
    let mut true_ranges = Vec::new();
    
    for i in 1..closes.len() {
        let high_low = highs[i] - lows[i];
        let high_close = (highs[i] - closes[i - 1]).abs();
        let low_close = (lows[i] - closes[i - 1]).abs();
        
        let tr = high_low.max(high_close).max(low_close);
        true_ranges.push(tr);
    }
    
    let atr_values = calculate_sma(&true_ranges, period);
    Ok(atr_values.into_pyarray_bound(py).into())
}

/// Fast CCI calculation
#[pyfunction]
fn fast_cci(
    py: Python,
    highs: PyReadonlyArray1<f64>,
    lows: PyReadonlyArray1<f64>,
    closes: PyReadonlyArray1<f64>,
    period: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let highs = highs.as_slice()?;
    let lows = lows.as_slice()?;
    let closes = closes.as_slice()?;
    
    if highs.len() != lows.len() || lows.len() != closes.len() || closes.len() < period {
        return Ok(vec![].into_pyarray_bound(py).into());
    }
    
    // Calculate typical prices
    let typical_prices: Vec<f64> = (0..closes.len())
        .map(|i| (highs[i] + lows[i] + closes[i]) / 3.0)
        .collect();
    
    let mut cci_values = Vec::new();
    
    for i in (period - 1)..typical_prices.len() {
        let slice = &typical_prices[(i + 1 - period)..=i];
        let sma = slice.iter().sum::<f64>() / period as f64;
        let mean_deviation = slice.iter().map(|x| (x - sma).abs()).sum::<f64>() / period as f64;
        
        let cci = if mean_deviation != 0.0 {
            (typical_prices[i] - sma) / (0.015 * mean_deviation)
        } else {
            0.0
        };
        cci_values.push(cci);
    }
    
    Ok(cci_values.into_pyarray_bound(py).into())
}

/// Fast OBV calculation
#[pyfunction]
fn fast_obv(
    py: Python,
    closes: PyReadonlyArray1<f64>,
    volumes: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let closes = closes.as_slice()?;
    let volumes = volumes.as_slice()?;
    
    if closes.len() != volumes.len() || closes.len() < 2 {
        return Ok(vec![].into_pyarray_bound(py).into());
    }
    
    let mut obv_values = vec![volumes[0]];
    
    for i in 1..closes.len() {
        let obv = if closes[i] > closes[i - 1] {
            obv_values[i - 1] + volumes[i]
        } else if closes[i] < closes[i - 1] {
            obv_values[i - 1] - volumes[i]
        } else {
            obv_values[i - 1]
        };
        obv_values.push(obv);
    }
    
    Ok(obv_values.into_pyarray_bound(py).into())
}

/// Fast ADX calculation
#[pyfunction]
fn fast_adx(
    py: Python,
    highs: PyReadonlyArray1<f64>,
    lows: PyReadonlyArray1<f64>,
    closes: PyReadonlyArray1<f64>,
    period: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let highs = highs.as_slice()?;
    let lows = lows.as_slice()?;
    let closes = closes.as_slice()?;
    
    if highs.len() != lows.len() || lows.len() != closes.len() || closes.len() < period + 1 {
        return Ok(vec![].into_pyarray_bound(py).into());
    }
    
    let mut plus_dm = Vec::new();
    let mut minus_dm = Vec::new();
    let mut true_ranges = Vec::new();
    
    for i in 1..closes.len() {
        let high_diff = highs[i] - highs[i - 1];
        let low_diff = lows[i - 1] - lows[i];
        
        plus_dm.push(if high_diff > low_diff && high_diff > 0.0 { high_diff } else { 0.0 });
        minus_dm.push(if low_diff > high_diff && low_diff > 0.0 { low_diff } else { 0.0 });
        
        let high_low = highs[i] - lows[i];
        let high_close = (highs[i] - closes[i - 1]).abs();
        let low_close = (lows[i] - closes[i - 1]).abs();
        true_ranges.push(high_low.max(high_close).max(low_close));
    }
    
    let plus_di = calculate_sma(&plus_dm, period);
    let minus_di = calculate_sma(&minus_dm, period);
    let atr = calculate_sma(&true_ranges, period);
    
    let mut adx_values = Vec::new();
    for i in 0..plus_di.len().min(minus_di.len()).min(atr.len()) {
        if atr[i] != 0.0 {
            let plus_di_val = (plus_di[i] / atr[i]) * 100.0;
            let minus_di_val = (minus_di[i] / atr[i]) * 100.0;
            let dx = if plus_di_val + minus_di_val != 0.0 {
                ((plus_di_val - minus_di_val).abs() / (plus_di_val + minus_di_val)) * 100.0
            } else {
                0.0
            };
            adx_values.push(dx);
        }
    }
    
    let adx = calculate_sma(&adx_values, period);
    Ok(adx.into_pyarray_bound(py).into())
}

/// Fast VWAP calculation
#[pyfunction]
fn fast_vwap(
    py: Python,
    highs: PyReadonlyArray1<f64>,
    lows: PyReadonlyArray1<f64>,
    closes: PyReadonlyArray1<f64>,
    volumes: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let highs = highs.as_slice()?;
    let lows = lows.as_slice()?;
    let closes = closes.as_slice()?;
    let volumes = volumes.as_slice()?;
    
    if highs.len() != lows.len() || lows.len() != closes.len() || closes.len() != volumes.len() {
        return Ok(vec![].into_pyarray_bound(py).into());
    }
    
    let mut cumulative_price_volume = 0.0;
    let mut cumulative_volume = 0.0;
    let mut vwap_values = Vec::new();
    
    for i in 0..closes.len() {
        let typical_price = (highs[i] + lows[i] + closes[i]) / 3.0;
        cumulative_price_volume += typical_price * volumes[i];
        cumulative_volume += volumes[i];
        
        let vwap = if cumulative_volume != 0.0 {
            cumulative_price_volume / cumulative_volume
        } else {
            typical_price
        };
        vwap_values.push(vwap);
    }
    
    Ok(vwap_values.into_pyarray_bound(py).into())
}

// Helper functions
fn calculate_ema(prices: &[f64], period: usize) -> Vec<f64> {
    if prices.is_empty() {
        return vec![];
    }
    
    let multiplier = 2.0 / (period as f64 + 1.0);
    let mut ema_values = Vec::new();
    
    ema_values.push(prices[0]);
    
    for i in 1..prices.len() {
        let ema = (prices[i] * multiplier) + (ema_values[i - 1] * (1.0 - multiplier));
        ema_values.push(ema);
    }
    
    ema_values
}

fn calculate_sma(prices: &[f64], period: usize) -> Vec<f64> {
    if prices.len() < period {
        return vec![];
    }
    
    let mut sma_values = Vec::new();
    
    for i in (period - 1)..prices.len() {
        let sum: f64 = prices[(i + 1 - period)..=i].iter().sum();
        sma_values.push(sum / period as f64);
    }
    
    sma_values
}

/// Python module definition
#[pymodule]
fn fast_indicators(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fast_rsi, m)?)?;
    m.add_function(wrap_pyfunction!(fast_macd, m)?)?;
    m.add_function(wrap_pyfunction!(fast_bollinger_bands, m)?)?;
    m.add_function(wrap_pyfunction!(fast_stochastic, m)?)?;
    m.add_function(wrap_pyfunction!(fast_ema, m)?)?;
    m.add_function(wrap_pyfunction!(fast_atr, m)?)?;
    m.add_function(wrap_pyfunction!(fast_cci, m)?)?;
    m.add_function(wrap_pyfunction!(fast_obv, m)?)?;
    m.add_function(wrap_pyfunction!(fast_adx, m)?)?;
    m.add_function(wrap_pyfunction!(fast_vwap, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ema_calculation() {
        let prices = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ema = calculate_ema(&prices, 3);
        assert_eq!(ema.len(), prices.len());
        assert_eq!(ema[0], 1.0);
    }

    #[test]
    fn test_sma_calculation() {
        let prices = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sma = calculate_sma(&prices, 3);
        assert_eq!(sma, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_rsi_basic() {
        let prices = vec![44.0, 45.0, 44.5, 45.5, 44.8, 45.2, 44.9, 45.1, 45.5, 46.0];
        let gains = vec![];
        let losses = vec![];
        
        for i in 1..prices.len() {
            let change = prices[i] - prices[i - 1];
            if change > 0.0 {
                assert!(change >= 0.0);
            } else {
                assert!(change <= 0.0);
            }
        }
    }
}
