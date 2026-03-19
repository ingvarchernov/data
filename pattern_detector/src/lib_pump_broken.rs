use pyo3::prelude::*;
use pyo3::types::PyDict;

/// ⭐ НОВИЙ ПІДХІД: Шукаємо ПАМПИ/ДАМПИ - великі рухи 3-10%
/// 
/// Логіка:
/// 1. Compression zone (накопичення) - LOW volatility перед рухом
/// 2. Breakout (початок руху) - VOLUME spike + сильна свічка
/// 3. Momentum (продовження) - декілька свічок в одному напрямку
/// 4. Exit (кінець руху) - reversal pattern АБО повернення до compression

/// Розрахунок EMA
fn calculate_ema(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut ema = vec![0.0; n];
    
    if n < period {
        return ema;
    }
    
    // Перша EMA = SMA
    let mut sum = 0.0;
    for i in 0..period {
        sum += data[i];
    }
    ema[period - 1] = sum / period as f64;
    
    // Наступні EMA
    let multiplier = 2.0 / (period as f64 + 1.0);
    for i in period..n {
        ema[i] = (data[i] - ema[i-1]) * multiplier + ema[i-1];
    }
    
    ema
}

/// Шукаємо ПАМПИ - швидкі рухи 3-10% за кілька свічок
#[pyfunction]
fn detect_pumps(
    py: Python,
    opens: Vec<f64>,
    highs: Vec<f64>,
    lows: Vec<f64>,
    closes: Vec<f64>,
    volumes: Vec<f64>,
    min_pump_pct: Option<f64>,  // Мінімальний рух % (default 3%)
    max_candles: Option<usize>   // Максимум свічок для руху (default 10)
) -> PyResult<Vec<Py<PyDict>>> {
    let min_move = min_pump_pct.unwrap_or(3.0);
    let max_duration = max_candles.unwrap_or(10);
    
    let n = closes.len();
    if n < 50 {
        return Ok(Vec::new());
    }
    
    // EMA для trend filter
    let ema21 = calculate_ema(&closes, 21);
    let ema50 = calculate_ema(&closes, 50);
    
    let mut results = Vec::new();
    let mut i = 50;  // Починаємо після EMA warm-up
    
    while i < n - max_duration {
        let start_price = closes[i];
        let start_volume = volumes[i];
        
        // Шукаємо рух в наступних свічках
        for duration in 3..=max_duration {
            if i + duration >= n {
                break;
            }
            
            let end_idx = i + duration;
            let end_price = closes[end_idx];
            
            // Розрахунок руху
            let move_pct = ((end_price - start_price) / start_price).abs() * 100.0;
            
            if move_pct >= min_move {
                // Знайшли рух! Перевіряємо якість
                
                // Volume confirmation - повинен бути spike
                let mut avg_volume = 0.0_f64;
                let mut max_volume = 0.0_f64;
                for j in i..(i + duration) {
                    avg_volume += volumes[j];
                    max_volume = max_volume.max(volumes[j]);
                }
                avg_volume /= duration as f64;
                
                let vol_ratio = max_volume / start_volume;
                if vol_ratio < 1.3 {
                    continue;  // Немає volume spike
                }
                
                // Напрямок руху
                let direction = if end_price > start_price {
                    "LONG"
                } else {
                    "SHORT"
                };
                
                // Trend alignment - для SHORT потрібен downtrend
                let trend_ok = if direction == "SHORT" {
                    ema21[end_idx] < ema50[end_idx]
                } else {
                    ema21[end_idx] > ema50[end_idx]
                };
                
                if !trend_ok {
                    continue;
                }
                
                // Confidence на базі move_pct та volume
                let mut confidence = 50.0;
                
                if move_pct > 7.0 {
                    confidence += 30.0;
                } else if move_pct > 5.0 {
                    confidence += 20.0;
                } else if move_pct > 3.0 {
                    confidence += 10.0;
                }
                
                if vol_ratio > 2.5 {
                    confidence += 20.0;
                } else if vol_ratio > 1.8 {
                    confidence += 10.0;
                }
                
                let final_confidence = if confidence > 100.0 { 100.0 } else { confidence };
                
                // Тільки high confidence
                if final_confidence < 70.0 {
                    continue;
                }
                
                // Створюємо сигнал
                let dict = PyDict::new(py);
                dict.set_item("index", i)?;
                dict.set_item("direction", direction)?;
                dict.set_item("pattern_name", format!("PUMP {} ({:.1}% in {}c)", direction, move_pct, duration))?;
                dict.set_item("confidence", final_confidence)?;
                dict.set_item("entry_price", start_price)?;
                dict.set_item("move_pct", move_pct)?;
                dict.set_item("duration_candles", duration)?;
                dict.set_item("vol_ratio", vol_ratio)?;
                
                // SL/TP для backtest
                let sl_pct = 0.02;  // 2% SL
                let tp_pct = move_pct / 100.0 * 0.5;  // TP = 50% від руху
                
                let (sl_price, tp_price) = if direction == "SHORT" {
                    (start_price * (1.0 + sl_pct), start_price * (1.0 - tp_pct))
                } else {
                    (start_price * (1.0 - sl_pct), start_price * (1.0 + tp_pct))
                };
                
                dict.set_item("sl_price", sl_price)?;
                dict.set_item("tp_price", tp_price)?;
                
                results.push(dict.into());
                
                // Пропускаємо знайдений рух
                i = end_idx + 5;
                break;
            }
        }
        
        i += 1;
    }
    
    Ok(results)
}

#[pymodule]
fn pattern_detector(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(detect_pumps, m)?)?;
    Ok(())
}
