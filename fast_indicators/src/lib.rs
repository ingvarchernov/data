use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};

// Helper: SMA calculation
fn sma_calc(data: &[f64], period: usize) -> Vec<f64> {
    let len = data.len();
    let mut result = vec![f64::NAN; len];

    if len < period || period == 0 {
        return result;
    }

    for i in (period - 1)..len {
        let mut sum = 0.0;
        let mut count = 0;
        let start_idx = i.saturating_sub(period - 1);
        for j in start_idx..=i {
            if j < data.len() && !data[j].is_nan() {
                sum += data[j];
                count += 1;
            }
        }
        if count > 0 {
            result[i] = sum / count as f64;
        }
    }

    result
}

// Helper: EMA calculation
fn ema_calc(data: &[f64], period: usize) -> Vec<f64> {
    let len = data.len();
    let mut result = vec![f64::NAN; len];

    if len < period {
        return result;
    }

    let multiplier = 2.0 / (period as f64 + 1.0);

    let mut sum = 0.0;
    let mut count = 0;
    for i in 0..period {
        if !data[i].is_nan() {
            sum += data[i];
            count += 1;
        }
    }

    if count > 0 {
        result[period - 1] = sum / count as f64;

        for i in period..len {
            if !data[i].is_nan() && !result[i - 1].is_nan() {
                result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1];
            }
        }
    }

    result
}

// Simple Moving Average
#[pyfunction]
fn sma(py: Python, data: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    let data = data.as_slice()?;
    let result = sma_calc(data, period);
    Ok(result.into_pyarray_bound(py).into())
}

// Exponential Moving Average
#[pyfunction]
fn ema(py: Python, data: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    let data = data.as_slice()?;
    let result = ema_calc(data, period);
    Ok(result.into_pyarray_bound(py).into())
}

// RSI
#[pyfunction]
fn rsi(py: Python, prices: PyReadonlyArray1<f64>, period: usize) -> PyResult<Py<PyArray1<f64>>> {
    let prices = prices.as_slice()?;
    let len = prices.len();

    if len < period + 1 {
        return Ok(vec![f64::NAN; len].into_pyarray_bound(py).into());
    }

    let mut result = vec![f64::NAN; len];
    let mut avg_gain = 0.0;
    let mut avg_loss = 0.0;

    for i in 1..=period {
        let change = prices[i] - prices[i - 1];
        if change > 0.0 {
            avg_gain += change;
        } else {
            avg_loss += change.abs();
        }
    }

    avg_gain /= period as f64;
    avg_loss /= period as f64;

    result[period] = if avg_loss != 0.0 {
        let rs = avg_gain / avg_loss;
        100.0 - (100.0 / (1.0 + rs))
    } else {
        100.0
    };

    for i in (period + 1)..len {
        let change = prices[i] - prices[i - 1];
        let gain = if change > 0.0 { change } else { 0.0 };
        let loss = if change < 0.0 { change.abs() } else { 0.0 };

        avg_gain = ((avg_gain * (period as f64 - 1.0)) + gain) / period as f64;
        avg_loss = ((avg_loss * (period as f64 - 1.0)) + loss) / period as f64;

        result[i] = if avg_loss != 0.0 {
            let rs = avg_gain / avg_loss;
            100.0 - (100.0 / (1.0 + rs))
        } else {
            100.0
        };
    }

    Ok(result.into_pyarray_bound(py).into())
}

// ATR
#[pyfunction]
fn atr(
    py: Python,
    high: PyReadonlyArray1<f64>,
    low: PyReadonlyArray1<f64>,
    close: PyReadonlyArray1<f64>,
    period: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let len = close.len();

    let mut tr = vec![f64::NAN; len];

    for i in 1..len {
        let tr1 = high[i] - low[i];
        let tr2 = (high[i] - close[i - 1]).abs();
        let tr3 = (low[i] - close[i - 1]).abs();
        tr[i] = tr1.max(tr2).max(tr3);
    }

    let result = sma_calc(&tr, period);
    Ok(result.into_pyarray_bound(py).into())
}

// OBV
#[pyfunction]
fn obv(
    py: Python,
    close: PyReadonlyArray1<f64>,
    volume: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let close = close.as_slice()?;
    let volume = volume.as_slice()?;
    let len = close.len();

    let mut result = vec![0.0; len];

    for i in 1..len {
        let direction = if close[i] > close[i - 1] {
            1.0
        } else if close[i] < close[i - 1] {
            -1.0
        } else {
            0.0
        };

        result[i] = result[i - 1] + (direction * volume[i]);
    }

    Ok(result.into_pyarray_bound(py).into())
}

// VWAP
#[pyfunction]
fn vwap(
    py: Python,
    high: PyReadonlyArray1<f64>,
    low: PyReadonlyArray1<f64>,
    close: PyReadonlyArray1<f64>,
    volume: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let volume = volume.as_slice()?;
    let len = close.len();

    let mut result = vec![f64::NAN; len];
    let mut cumulative_tpv = 0.0;
    let mut cumulative_volume = 0.0;

    for i in 0..len {
        let typical_price = (high[i] + low[i] + close[i]) / 3.0;
        cumulative_tpv += typical_price * volume[i];
        cumulative_volume += volume[i];

        if cumulative_volume > 0.0 {
            result[i] = cumulative_tpv / cumulative_volume;
        }
    }

    Ok(result.into_pyarray_bound(py).into())
}

// Rolling STD
#[pyfunction]
fn rolling_std(
    py: Python,
    data: PyReadonlyArray1<f64>,
    period: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let data = data.as_slice()?;
    let len = data.len();
    let mut result = vec![f64::NAN; len];

    if period == 0 || len < period {
        return Ok(result.into_pyarray_bound(py).into());
    }

    for i in (period - 1)..len {
        let start_idx = i.saturating_sub(period - 1);
        let mean: f64 = data[start_idx..=i].iter().sum::<f64>() / period as f64;
        let variance: f64 = data[start_idx..=i]
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / period as f64;
        result[i] = variance.sqrt();
    }

    Ok(result.into_pyarray_bound(py).into())
}

// Historical Volatility
#[pyfunction]
fn historical_volatility(
    py: Python,
    returns: PyReadonlyArray1<f64>,
    period: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let returns = returns.as_slice()?;
    let len = returns.len();
    let mut result = vec![f64::NAN; len];

    if period == 0 || len < period {
        return Ok(result.into_pyarray_bound(py).into());
    }

    for i in (period - 1)..len {
        let start_idx = i.saturating_sub(period - 1);
        let mean: f64 = returns[start_idx..=i].iter().sum::<f64>() / period as f64;
        let variance: f64 = returns[start_idx..=i]
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / period as f64;
        result[i] = variance.sqrt() * (252.0_f64).sqrt();
    }

    Ok(result.into_pyarray_bound(py).into())
}

// ============ ADVANCED FEATURES (AI FACTOR) ============

// RSI Divergence Detection
#[pyfunction]
fn rsi_divergence(
    py: Python,
    prices: PyReadonlyArray1<f64>,
    rsi: PyReadonlyArray1<f64>,
    period: usize,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let prices = prices.as_slice()?;
    let rsi = rsi.as_slice()?;
    let len = prices.len();

    let mut bull_div = vec![0.0; len];
    let mut bear_div = vec![0.0; len];

    if len < period + 1 {
        return Ok((bull_div.into_pyarray_bound(py).into(), bear_div.into_pyarray_bound(py).into()));
    }

    for i in period..len {
        let price_change = prices[i] - prices[i - period];
        let rsi_change = rsi[i] - rsi[i - period];

        // Bullish divergence: price down, RSI up
        if price_change < 0.0 && rsi_change > 0.0 {
            bull_div[i] = 1.0;
        }

        // Bearish divergence: price up, RSI down
        if price_change > 0.0 && rsi_change < 0.0 {
            bear_div[i] = 1.0;
        }
    }

    Ok((bull_div.into_pyarray_bound(py).into(), bear_div.into_pyarray_bound(py).into()))
}

// MACD Histogram Acceleration
#[pyfunction]
fn macd_histogram_accel(
    py: Python,
    macd: PyReadonlyArray1<f64>,
    signal: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let macd = macd.as_slice()?;
    let signal = signal.as_slice()?;
    let len = macd.len();

    let mut result = vec![f64::NAN; len];

    for i in 2..len {
        let hist_prev = macd[i-1] - signal[i-1];
        let hist_curr = macd[i] - signal[i];
        result[i] = hist_curr - hist_prev;
    }

    Ok(result.into_pyarray_bound(py).into())
}

// Bollinger Squeeze Detection
#[pyfunction]
fn bollinger_squeeze(
    py: Python,
    bb_width: PyReadonlyArray1<f64>,
    period: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let bb_width = bb_width.as_slice()?;
    let len = bb_width.len();
    let mut result = vec![0.0; len];

    for i in period..len {
        let min_width = bb_width[i - period..i]
            .iter()
            .filter(|x| !x.is_nan())
            .fold(f64::INFINITY, |a, &b| a.min(b));

        if (bb_width[i] - min_width).abs() < 1e-10 {
            result[i] = 1.0;
        }
    }

    Ok(result.into_pyarray_bound(py).into())
}

// Volume Spike Detection
#[pyfunction]
fn volume_spike(
    py: Python,
    volume: PyReadonlyArray1<f64>,
    threshold: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let volume = volume.as_slice()?;
    let len = volume.len();
    let mut result = vec![0.0; len];

    let period = 20;
    for i in period..len {
        let mean_vol: f64 = volume[i - period..i].iter().sum::<f64>() / period as f64;
        if volume[i] > mean_vol * threshold {
            result[i] = 1.0;
        }
    }

    Ok(result.into_pyarray_bound(py).into())
}

// Trend Direction Multi-Timeframe
#[pyfunction]
fn trend_direction(
    py: Python,
    prices: PyReadonlyArray1<f64>,
    period: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let prices = prices.as_slice()?;
    let len = prices.len();
    let mut result = vec![f64::NAN; len];

    for i in period..len {
        if prices[i] > prices[i - period] {
            result[i] = 1.0;
        } else {
            result[i] = 0.0;
        }
    }

    Ok(result.into_pyarray_bound(py).into())
}

// True Range
#[pyfunction]
fn true_range(
    py: Python,
    high: PyReadonlyArray1<f64>,
    low: PyReadonlyArray1<f64>,
    close: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let len = high.len();

    let mut result = vec![f64::NAN; len];

    result[0] = high[0] - low[0];

    for i in 1..len {
        let hl = high[i] - low[i];
        let hc = (high[i] - close[i-1]).abs();
        let lc = (low[i] - close[i-1]).abs();
        result[i] = hl.max(hc).max(lc);
    }

    Ok(result.into_pyarray_bound(py).into())
}

// Support/Resistance Levels
#[pyfunction]
fn support_resistance(
    py: Python,
    high: PyReadonlyArray1<f64>,
    low: PyReadonlyArray1<f64>,
    period: usize,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let len = high.len();

    let mut resistance = vec![f64::NAN; len];
    let mut support = vec![f64::NAN; len];

    for i in period..len {
        resistance[i] = high[i - period..i]
            .iter()
            .filter(|x| !x.is_nan())
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        support[i] = low[i - period..i]
            .iter()
            .filter(|x| !x.is_nan())
            .fold(f64::INFINITY, |a, &b| a.min(b));
    }

    Ok((resistance.into_pyarray_bound(py).into(), support.into_pyarray_bound(py).into()))
}

// Price-Volume Correlation
#[pyfunction]
fn price_volume_correlation(
    py: Python,
    prices: PyReadonlyArray1<f64>,
    volume: PyReadonlyArray1<f64>,
    period: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let prices = prices.as_slice()?;
    let volume = volume.as_slice()?;
    let len = prices.len();

    let mut result = vec![f64::NAN; len];

    for i in period..len {
        let price_change = (prices[i] - prices[i - period]) / prices[i - period];
        let volume_change = (volume[i] - volume[i - period]) / volume[i - period];
        result[i] = price_change * volume_change;
    }

    Ok(result.into_pyarray_bound(py).into())
}

#[pymodule]
fn fast_indicators(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Original indicators
    m.add_function(wrap_pyfunction!(sma, m)?)?;
    m.add_function(wrap_pyfunction!(ema, m)?)?;
    m.add_function(wrap_pyfunction!(rsi, m)?)?;
    m.add_function(wrap_pyfunction!(atr, m)?)?;
    m.add_function(wrap_pyfunction!(obv, m)?)?;
    m.add_function(wrap_pyfunction!(vwap, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_std, m)?)?;
    m.add_function(wrap_pyfunction!(historical_volatility, m)?)?;

    // Advanced features (AI Factor)
    m.add_function(wrap_pyfunction!(rsi_divergence, m)?)?;
    m.add_function(wrap_pyfunction!(macd_histogram_accel, m)?)?;
    m.add_function(wrap_pyfunction!(bollinger_squeeze, m)?)?;
    m.add_function(wrap_pyfunction!(volume_spike, m)?)?;
    m.add_function(wrap_pyfunction!(trend_direction, m)?)?;
    m.add_function(wrap_pyfunction!(true_range, m)?)?;
    m.add_function(wrap_pyfunction!(support_resistance, m)?)?;
    m.add_function(wrap_pyfunction!(price_volume_correlation, m)?)?;

    Ok(())
}
