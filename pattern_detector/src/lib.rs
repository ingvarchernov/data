use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Compression Zone - зона стиснення ціни
#[derive(Clone, Debug)]
struct CompressionZone {
    start_idx: usize,
    end_idx: usize,
    duration: usize,
    avg_volume: f64,
    avg_range_pct: f64,
    resistance: f64,
    support: f64,
    
}

/// Розрахунок ATR (Average True Range)
fn calculate_atr(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> Vec<f64> {
    let n = closes.len();
    let mut atr = vec![0.0; n];
    if n < period { return atr; }

    let mut tr_values = vec![0.0; n];
    tr_values[0] = highs[0] - lows[0];

    for i in 1..n {
        let hl = highs[i] - lows[i];
        let hc = (highs[i] - closes[i-1]).abs();
        let lc = (lows[i] - closes[i-1]).abs();
        tr_values[i] = hl.max(hc).max(lc);
    }

    let first_atr: f64 = tr_values.iter().take(period).sum::<f64>() / period as f64;
    atr[period - 1] = first_atr;

    let multiplier = 2.0 / (period as f64 + 1.0);
    for i in period..n {
        atr[i] = (tr_values[i] - atr[i-1]) * multiplier + atr[i-1];
    }
    atr
}

/// Розрахунок EMA
fn calculate_ema(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut ema = vec![0.0; n];
    if n < period { return ema; }

    let mut sum: f64 = data.iter().take(period).sum();
    ema[period - 1] = sum / period as f64;

    let multiplier = 2.0 / (period as f64 + 1.0);
    for i in period..n {
        ema[i] = (data[i] - ema[i-1]) * multiplier + ema[i-1];
    }
    ema
}

/// Пошук compression zones
fn find_compression_zones(
    highs: &[f64], lows: &[f64], closes: &[f64], volumes: &[f64],
    min_candles: usize, max_range_pct: f64
) -> Vec<CompressionZone> {
    let n = closes.len();
    let mut zones = Vec::new();
    let atr = calculate_atr(highs, lows, closes, 14);

    let mut i = 14;
    while i < n.saturating_sub(5) {
        let mut high_max = highs[i];
        let mut low_min = lows[i];
        let mut vol_sum = volumes[i];
        let mut count = 1;

        let mut j = i + 1;
        while j < n.saturating_sub(5) {
            high_max = high_max.max(highs[j]);
            low_min = low_min.min(lows[j]);
            vol_sum += volumes[j];
            count += 1;

            let range_pct = (high_max - low_min) / low_min * 100.0;
            let atr_pct = (atr[j] / closes[j]) * 100.0;

            if range_pct > max_range_pct || atr_pct > 1.5 { break; }

            if count >= min_candles {
                zones.push(CompressionZone {
                    start_idx: i, end_idx: j, duration: count,
                    avg_range_pct: range_pct, avg_volume: vol_sum / count as f64,
                    resistance: high_max, support: low_min,
                });
            }
            j += 1;
        }
        i = if count >= min_candles { j + 1 } else { i + 1 };
    }
    zones
}

/// Перевірка breakout після compression zone
fn check_breakout(
    zone: &CompressionZone,
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
    volumes: &[f64],
    ema9: &[f64],
    ema21: &[f64],
    ema50: &[f64],
    volume_threshold: f64
) -> Option<(usize, String, f64, f64, f64, f64, f64)> {
    // Перевіряємо 5 свічок після зони
    let start = zone.end_idx + 1;
    let end = (start + 5).min(highs.len());
    
    for i in start..end {
        let close = closes[i];
        let volume = volumes[i];
        let open = if i > 0 { closes[i-1] } else { close };
        let body_pct = ((close - open).abs() / open) * 100.0;
        
        // ❌ ВИМКНЕНО LONG - не працює в backtest
        
        // ⭐ SHORT breakout (пробій вниз) - шукаємо ПАМП ВНИЗ
        if close < zone.support
            && volume > zone.avg_volume * volume_threshold
            && close < ema21[i]
            && body_pct > 1.0  // Сильна свічка
        {
            let entry = close;
            let sl = zone.resistance;
            
            // ⭐ EXIT = коли ціна повернулась до compression зони
            // АБО trailing stop
            // АБО reversal pattern
            let tp = entry - (sl - entry) * 3.0;  // Дефолтний TP для бектесту
            
            // Підвищено мінімальний потенціал з 2% до 3%
            let tp_distance_pct = (entry - tp) / entry * 100.0;
            if tp_distance_pct < 3.0 {
                continue;
            }
            
            // Розрахунок confidence
            let vol_ratio = volume / zone.avg_volume;
            let mut confidence = 50.0;
            
            // Посилено вимоги до volume
            if vol_ratio > 3.0 {
                confidence += 25.0;
            } else if vol_ratio > 2.0 {
                confidence += 15.0;
            } else if vol_ratio > 1.5 {
                confidence += 5.0;
            }
            
            // Посилено вимоги до body size
            if body_pct > 3.0 {
                confidence += 20.0;
            } else if body_pct > 2.0 {
                confidence += 10.0;
            }
            
            // EMA alignment (обов'язково!)
            if ema9[i] < ema21[i] && ema21[i] < ema50[i] {
                confidence += 15.0;
            } else {
                continue;  // Якщо EMA не вирівняні - пропускаємо
            }
            
            // Фільтр: тільки HIGH confidence (70+)
            if confidence < 70.0 {
                continue;
            }
            
            let final_confidence = if confidence > 100.0 { 100.0 } else { confidence };
            
            return Some((
                i,
                "SHORT".to_string(),
                final_confidence,
                entry,
                sl,
                tp,
                zone.duration as f64
            ));
        }
    }
    
    None
}

/// Детекція BREAKOUTS (головна функція)
#[pyfunction]
fn detect_breakouts(
    py: Python,
    highs: Vec<f64>,
    lows: Vec<f64>,
    closes: Vec<f64>,
    volumes: Vec<f64>,
    min_compression_candles: Option<usize>,
    max_range_pct: Option<f64>,
    volume_threshold: Option<f64>
) -> PyResult<Vec<Py<PyDict>>> {
    // 🛡️ Валідація довжини (Захист від сегфолтів/паніки)
    let n = closes.len();
    if highs.len() != n || lows.len() != n || volumes.len() != n {
        return Err(PyValueError::new_err("Всі вхідні масиви повинні мати однакову довжину"));
    }

    let min_c = min_compression_candles.unwrap_or(12);
    let max_r = max_range_pct.unwrap_or(1.2);
    let vol_t = volume_threshold.unwrap_or(1.5);

    let ema9 = calculate_ema(&closes, 9);
    let ema21 = calculate_ema(&closes, 21);
    let ema50 = calculate_ema(&closes, 50);
    let zones = find_compression_zones(&highs, &lows, &closes, &volumes, min_c, max_r);

    let mut results = Vec::new();
    let mut last_idx = 0;

    for zone in zones {
        let start = zone.end_idx + 1;
        let end = (start + 5).min(n);

        for i in start..end {
            let close = closes[i];
            let open = closes[i-1];
            let body_pct = ((close - open).abs() / open) * 100.0;
            let vol_ratio = volumes[i] / zone.avg_volume;

            // SHORT Breakout logic
            if close < zone.support && vol_ratio > vol_t && close < ema21[i] && body_pct > 1.0 {
                if ema9[i] >= ema21[i] || ema21[i] >= ema50[i] { continue; } // EMA Alignment check

                let mut confidence = 50.0 + (vol_ratio * 5.0).min(30.0) + (body_pct * 5.0).min(20.0);
                if confidence < 70.0 || i <= last_idx + 10 { continue; }

                let dict = PyDict::new(py);
                dict.set_item("index", i)?;
                dict.set_item("direction", "SHORT")?;
                dict.set_item("confidence", confidence.min(100.0))?;
                dict.set_item("entry_price", close)?;
                dict.set_item("sl_price", zone.resistance)?;
                dict.set_item("tp_price", close - (zone.resistance - close) * 3.0)?;
                dict.set_item("compression_start_idx", zone.start_idx)?;
                dict.set_item("compression_end_idx", zone.end_idx)?;
                
                results.push(dict.into());
                last_idx = i;
                break; 
            }
        }
    }
    Ok(results)
}

/// Розрахунок RSI (Relative Strength Index)
fn calculate_rsi(prices: &[f64], period: usize) -> Vec<f64> {
    let n = prices.len();
    let mut rsi = vec![0.0; n];
    
    if n < period + 1 {
        return rsi;
    }
    
    let mut gains = vec![0.0; n];
    let mut losses = vec![0.0; n];
    
    // Розрахунок gains/losses
    for i in 1..n {
        let change = prices[i] - prices[i-1];
        if change > 0.0 {
            gains[i] = change;
            losses[i] = 0.0;
        } else {
            gains[i] = 0.0;
            losses[i] = -change;
        }
    }
    
    // Перший RSI використовує SMA
    let mut avg_gain = 0.0;
    let mut avg_loss = 0.0;
    for i in 1..=(period) {
        if i < gains.len() {
            avg_gain += gains[i];
            avg_loss += losses[i];
        }
    }
    avg_gain /= period as f64;
    avg_loss /= period as f64;
    
    if avg_loss != 0.0 {
        let rs = avg_gain / avg_loss;
        rsi[period] = 100.0 - (100.0 / (1.0 + rs));
    }
    
    // Наступні RSI використовують smoothed averages
    for i in (period + 1)..n {
        if i < gains.len() && i < losses.len() {
            avg_gain = (avg_gain * (period as f64 - 1.0) + gains[i]) / period as f64;
            avg_loss = (avg_loss * (period as f64 - 1.0) + losses[i]) / period as f64;
            
            if avg_loss != 0.0 {
                let rs = avg_gain / avg_loss;
                rsi[i] = 100.0 - (100.0 / (1.0 + rs));
            }
        }
    }
    
    rsi
}

/// Розрахунок MACD
fn calculate_macd(prices: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let ema12 = calculate_ema(prices, 12);
    let ema26 = calculate_ema(prices, 26);
    
    let mut macd_line = vec![0.0; prices.len()];
    for i in 0..prices.len() {
        macd_line[i] = ema12[i] - ema26[i];
    }
    
    let macd_signal = calculate_ema(&macd_line, 9);
    
    let mut macd_histogram = vec![0.0; prices.len()];
    for i in 0..prices.len() {
        macd_histogram[i] = macd_line[i] - macd_signal[i];
    }
    
    (macd_line, macd_signal, macd_histogram)
}

/// Розрахунок Bollinger Bands
fn calculate_bollinger_bands(prices: &[f64], period: usize, std_dev: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = prices.len();
    let mut sma = vec![0.0; n];
    let mut upper = vec![0.0; n];
    let mut lower = vec![0.0; n];
    
    if n < period {
        return (sma, upper, lower);
    }
    
    // Розрахунок SMA та Bollinger Bands
    for i in (period - 1)..n {
        // Перевірка меж
        let start_idx = i.saturating_sub(period - 1);
        
        let mut sum = 0.0;
        let mut count = 0;
        for j in start_idx..=i {
            if j < n {
                sum += prices[j];
                count += 1;
            }
        }
        
        if count > 0 {
            sma[i] = sum / count as f64;
            
            // Розрахунок стандартного відхилення
            let mut variance = 0.0;
            for j in start_idx..=i {
                if j < n {
                    let diff = prices[j] - sma[i];
                    variance += diff * diff;
                }
            }
            
            if count > 1 {
                let std = (variance / (count - 1) as f64).sqrt();
                upper[i] = sma[i] + std_dev * std;
                lower[i] = sma[i] - std_dev * std;
            }
        }
    }
    
    (sma, upper, lower)
}

/// Основна функція для розрахунку всіх індикаторів
#[pyfunction]
fn calculate_indicators(py: Python, closes: Vec<f64>) -> PyResult<PyObject> {
    let rsi = calculate_rsi(&closes, 14);
    let ema20 = calculate_ema(&closes, 20);
    let ema50 = calculate_ema(&closes, 50);
    let (macd_line, macd_signal, macd_hist) = calculate_macd(&closes);
    let (bb_sma, bb_upper, bb_lower) = calculate_bollinger_bands(&closes, 20, 2.0);
    
    let dict = PyDict::new(py);
    dict.set_item("rsi", rsi)?;
    dict.set_item("ema20", ema20)?;
    dict.set_item("ema50", ema50)?;
    dict.set_item("macd_line", macd_line)?;
    dict.set_item("macd_signal", macd_signal)?;
    dict.set_item("macd_histogram", macd_hist)?;
    dict.set_item("bb_sma", bb_sma)?;
    dict.set_item("bb_upper", bb_upper)?;
    dict.set_item("bb_lower", bb_lower)?;
    
    Ok(dict.into())
}

#[pymodule]
fn pattern_detector(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(detect_breakouts, m)?)?;
    Ok(())
}
