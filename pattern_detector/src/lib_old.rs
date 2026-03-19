use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Структура для зберігання OHLC даних
#[pyclass]
struct CandleData {
    #[pyo3(get)]
    open: Vec<f64>,
    #[pyo3(get)]
    high: Vec<f64>,
    #[pyo3(get)]
    low: Vec<f64>,
    #[pyo3(get)]
    close: Vec<f64>,
}

/// Compression Zone - зона стиснення ціни
#[derive(Clone, Debug)]
struct CompressionZone {
    start_idx: usize,
    end_idx: usize,
    duration: usize,
    avg_range_pct: f64,
    avg_volume: f64,
    resistance: f64,
    support: f64,
    mid_price: f64,
}

#[pymethods]
impl CandleData {
    #[new]
    fn new(open: Vec<f64>, high: Vec<f64>, low: Vec<f64>, close: Vec<f64>) -> Self {
        CandleData { open, high, low, close }
    }
}

/// Результат виявлення патерна
#[pyclass]
#[derive(Clone)]
struct PatternResult {
    #[pyo3(get)]
    pattern_name: String,
    #[pyo3(get)]
    confidence: f64,
    #[pyo3(get)]
    direction: String,  // "LONG" або "SHORT"
    #[pyo3(get)]
    index: usize,
    #[pyo3(get)]
    entry_price: f64,
    #[pyo3(get)]
    sl_price: f64,
    #[pyo3(get)]
    tp_price: f64,
    #[pyo3(get)]
    compression_candles: usize,
}

#[pymethods]
impl PatternResult {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "Pattern({}, {}, conf={:.2}%, idx={})",
            self.pattern_name, self.direction, self.confidence, self.index
        ))
    }
}

/// 1. DOUBLE TOP / DOUBLE BOTTOM (Подвійна вершина / дно)
fn detect_double_top(highs: &[f64], window: usize, tolerance: f64) -> Vec<PatternResult> {
    let mut results = Vec::new();
    let len = highs.len();
    
    if len < window * 3 {
        return results;
    }
    
    for i in window..(len - window * 2) {
        let peak1 = highs[i];
        let valley = highs[i + window];
        
        // Шукаємо другу вершину
        for j in (i + window)..(len - window) {
            let peak2 = highs[j];
            
            // Перевірка: дві вершини приблизно на одному рівні
            if (peak1 - peak2).abs() / peak1 < tolerance && valley < peak1 * 0.97 {
                results.push(PatternResult {
                    pattern_name: "Double Top".to_string(),
                    confidence: 75.0,
                    direction: "SHORT".to_string(),
                    index: j,
                });
                break;
            }
        }
    }
    
    results
}

fn detect_double_bottom(lows: &[f64], window: usize, tolerance: f64) -> Vec<PatternResult> {
    let mut results = Vec::new();
    let len = lows.len();
    
    if len < window * 3 {
        return results;
    }
    
    for i in window..(len - window * 2) {
        let bottom1 = lows[i];
        let peak = lows[i + window];
        
        // Шукаємо друге дно
        for j in (i + window)..(len - window) {
            let bottom2 = lows[j];
            
            // Перевірка: два дна приблизно на одному рівні
            if (bottom1 - bottom2).abs() / bottom1 < tolerance && peak > bottom1 * 1.03 {
                results.push(PatternResult {
                    pattern_name: "Double Bottom".to_string(),
                    confidence: 75.0,
                    direction: "LONG".to_string(),
                    index: j,
                });
                break;
            }
        }
    }
    
    results
}

/// 2. HEAD AND SHOULDERS (Голова-Плечі)
fn detect_head_and_shoulders(highs: &[f64], _lows: &[f64], window: usize) -> Vec<PatternResult> {
    let mut results = Vec::new();
    let len = highs.len();
    
    if len < window * 5 {
        return results;
    }
    
    for i in window..(len - window * 4) {
        let left_shoulder = highs[i];
        let head = highs[i + window * 2];
        let right_shoulder = highs[i + window * 4];
        
        // Перевірка: голова вища за плечі
        if head > left_shoulder * 1.02 && 
           head > right_shoulder * 1.02 &&
           (left_shoulder - right_shoulder).abs() / left_shoulder < 0.03 {
            
            results.push(PatternResult {
                pattern_name: "Head and Shoulders".to_string(),
                confidence: 80.0,
                direction: "SHORT".to_string(),
                index: i + window * 4,
            });
        }
    }
    
    results
}

fn detect_inverse_head_and_shoulders(_highs: &[f64], lows: &[f64], window: usize) -> Vec<PatternResult> {
    let mut results = Vec::new();
    let len = lows.len();
    
    if len < window * 5 {
        return results;
    }
    
    for i in window..(len - window * 4) {
        let left_shoulder = lows[i];
        let head = lows[i + window * 2];
        let right_shoulder = lows[i + window * 4];
        
        // Перевірка: голова нижча за плечі
        if head < left_shoulder * 0.98 && 
           head < right_shoulder * 0.98 &&
           (left_shoulder - right_shoulder).abs() / left_shoulder < 0.03 {
            
            results.push(PatternResult {
                pattern_name: "Inverse Head and Shoulders".to_string(),
                confidence: 80.0,
                direction: "LONG".to_string(),
                index: i + window * 4,
            });
        }
    }
    
    results
}

/// 3. TRIANGLE (Трикутник)
fn detect_triangle(highs: &[f64], lows: &[f64], window: usize) -> Vec<PatternResult> {
    let mut results = Vec::new();
    let len = highs.len();
    
    if len < window * 3 {
        return results;
    }
    
    // Симетричний трикутник: highs падають, lows ростуть
    for i in 0..(len - window * 3) {
        let high_slope = (highs[i + window * 3] - highs[i]) / (window * 3) as f64;
        let low_slope = (lows[i + window * 3] - lows[i]) / (window * 3) as f64;
        
        // Перевірка: highs падають, lows ростуть
        if high_slope < -0.0001 && low_slope > 0.0001 {
            // Напрямок визначається пробиттям
            let recent_close = highs[i + window * 3];
            let avg_high = (highs[i] + highs[i + window * 3]) / 2.0;
            
            let direction = if recent_close > avg_high {
                "LONG".to_string()
            } else {
                "SHORT".to_string()
            };
            
            results.push(PatternResult {
                pattern_name: "Symmetrical Triangle".to_string(),
                confidence: 70.0,
                direction,
                index: i + window * 3,
            });
        }
    }
    
    results
}

/// 4. FLAG / PENNANT (Прапор / Вимпел)
fn detect_flag(highs: &[f64], lows: &[f64], closes: &[f64], window: usize) -> Vec<PatternResult> {
    let mut results = Vec::new();
    let len = closes.len();
    
    if len < window * 3 {
        return results;
    }
    
    for i in window..(len - window * 2) {
        // Перевірка на сильний тренд перед прапором
        let trend_start = closes[i - window];
        let trend_end = closes[i];
        let trend_change = (trend_end - trend_start) / trend_start;
        
        // Сильний рух > 3%
        if trend_change.abs() > 0.03 {
            // Прапор: консолідація після тренду
            let flag_high = highs[i..i + window].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let flag_low = lows[i..i + window].iter().cloned().fold(f64::INFINITY, f64::min);
            let flag_range = (flag_high - flag_low) / flag_low;
            
            // Консолідація (низька волатильність)
            if flag_range < 0.02 {
                let direction = if trend_change > 0.0 {
                    "LONG".to_string()
                } else {
                    "SHORT".to_string()
                };
                
                results.push(PatternResult {
                    pattern_name: "Flag".to_string(),
                    confidence: 72.0,
                    direction,
                    index: i + window,
                });
            }
        }
    }
    
    results
}

/// 5. WEDGE (Клин)
fn detect_wedge(highs: &[f64], lows: &[f64], window: usize) -> Vec<PatternResult> {
    let mut results = Vec::new();
    let len = highs.len();
    
    if len < window * 3 {
        return results;
    }
    
    for i in 0..(len - window * 3) {
        let high_slope = (highs[i + window * 3] - highs[i]) / (window * 3) as f64;
        let low_slope = (lows[i + window * 3] - lows[i]) / (window * 3) as f64;
        
        // Rising wedge (ведмежий): обидва ростуть, але highs повільніше
        if high_slope > 0.0 && low_slope > 0.0 && low_slope > high_slope {
            results.push(PatternResult {
                pattern_name: "Rising Wedge".to_string(),
                confidence: 68.0,
                direction: "SHORT".to_string(),
                index: i + window * 3,
            });
        }
        
        // Falling wedge (бичачий): обидва падають, але lows швидше
        if high_slope < 0.0 && low_slope < 0.0 && low_slope < high_slope {
            results.push(PatternResult {
                pattern_name: "Falling Wedge".to_string(),
                confidence: 68.0,
                direction: "LONG".to_string(),
                index: i + window * 3,
            });
        }
    }
    
    results
}

/// 6. THREE WHITE SOLDIERS / THREE BLACK CROWS
fn detect_three_soldiers_crows(opens: &[f64], closes: &[f64]) -> Vec<PatternResult> {
    let mut results = Vec::new();
    let len = closes.len();
    
    if len < 3 {
        return results;
    }
    
    for i in 0..(len - 2) {
        // Three White Soldiers
        let is_bullish = 
            closes[i] > opens[i] &&
            closes[i + 1] > opens[i + 1] &&
            closes[i + 2] > opens[i + 2] &&
            closes[i + 1] > closes[i] &&
            closes[i + 2] > closes[i + 1];
        
        if is_bullish {
            results.push(PatternResult {
                pattern_name: "Three White Soldiers".to_string(),
                confidence: 77.0,
                direction: "LONG".to_string(),
                index: i + 2,
            });
        }
        
        // Three Black Crows
        let is_bearish = 
            closes[i] < opens[i] &&
            closes[i + 1] < opens[i + 1] &&
            closes[i + 2] < opens[i + 2] &&
            closes[i + 1] < closes[i] &&
            closes[i + 2] < closes[i + 1];
        
        if is_bearish {
            results.push(PatternResult {
                pattern_name: "Three Black Crows".to_string(),
                confidence: 77.0,
                direction: "SHORT".to_string(),
                index: i + 2,
            });
        }
    }
    
    results
}

// ============================================================================
// НОВИЙ BREAKOUT DETECTOR - Compression Zones + Volume Breakouts
// ============================================================================

/// Розрахунок ATR (Average True Range)
fn calculate_atr(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> Vec<f64> {
    let len = highs.len();
    let mut atr = vec![0.0; len];
    
    if len < period + 1 {
        return atr;
    }
    
    // True Range для кожної свічки
    let mut tr = vec![0.0; len];
    for i in 1..len {
        let high_low = highs[i] - lows[i];
        let high_close = (highs[i] - closes[i - 1]).abs();
        let low_close = (lows[i] - closes[i - 1]).abs();
        tr[i] = high_low.max(high_close).max(low_close);
    }
    
    // Перший ATR = SMA(TR)
    let first_atr: f64 = tr[1..=period].iter().sum::<f64>() / period as f64;
    atr[period] = first_atr;
    
    // Наступні ATR = EMA
    for i in (period + 1)..len {
        atr[i] = (tr[i] + (period - 1) as f64 * atr[i - 1]) / period as f64;
    }
    
    atr
}

/// Розрахунок EMA (Exponential Moving Average)
fn calculate_ema(data: &[f64], period: usize) -> Vec<f64> {
    let len = data.len();
    let mut ema = vec![0.0; len];
    
    if len < period {
        return ema;
    }
    
    // Перший EMA = SMA
    let first_ema: f64 = data[0..period].iter().sum::<f64>() / period as f64;
    ema[period - 1] = first_ema;
    
    // Множник для EMA
    let multiplier = 2.0 / (period + 1) as f64;
    
    // Наступні значення
    for i in period..len {
        ema[i] = (data[i] - ema[i - 1]) * multiplier + ema[i - 1];
    }
    
    ema
}

/// Знаходить compression zones (зони низької волатильності)
fn find_compression_zones(
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
    volumes: &[f64],
    min_candles: usize,
    max_range_pct: f64,
) -> Vec<CompressionZone> {
    let len = closes.len();
    let mut zones = Vec::new();
    
    if len < min_candles + 50 {
        return zones;
    }
    
    // Розраховуємо ATR для визначення волатильності
    let atr = calculate_atr(highs, lows, closes, 14);
    
    let mut in_compression = false;
    let mut comp_start = 0;
    
    for i in 50..(len - 5) {
        // Перевіряємо чи свічка в compression
        let range_pct = (highs[i] - lows[i]) / lows[i] * 100.0;
        let atr_pct = atr[i] / closes[i] * 100.0;
        
        let is_compressed = range_pct < max_range_pct && atr_pct < 1.5;
        
        if is_compressed && !in_compression {
            // Початок compression zone
            in_compression = true;
            comp_start = i;
        } else if !is_compressed && in_compression {
            // Кінець compression zone
            let comp_end = i - 1;
            let duration = comp_end - comp_start + 1;
            
            if duration >= min_candles {
                // Розраховуємо характеристики зони
                let zone_highs = &highs[comp_start..=comp_end];
                let zone_lows = &lows[comp_start..=comp_end];
                let zone_vols = &volumes[comp_start..=comp_end];
                
                let resistance = zone_highs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let support = zone_lows.iter().cloned().fold(f64::INFINITY, f64::min);
                let mid_price = (resistance + support) / 2.0;
                
                let avg_volume: f64 = zone_vols.iter().sum::<f64>() / zone_vols.len() as f64;
                
                let ranges: Vec<f64> = (comp_start..=comp_end)
                    .map(|j| (highs[j] - lows[j]) / lows[j] * 100.0)
                    .collect();
                let avg_range_pct = ranges.iter().sum::<f64>() / ranges.len() as f64;
                
                zones.push(CompressionZone {
                    start_idx: comp_start,
                    end_idx: comp_end,
                    duration,
                    avg_range_pct,
                    avg_volume,
                    resistance,
                    support,
                    mid_price,
                });
            }
            
            in_compression = false;
        }
    }
    
    zones
}

/// Перевіряє breakout після compression zone
fn check_breakout(
    zone: &CompressionZone,
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
    volumes: &[f64],
    ema9: &[f64],
    ema21: &[f64],
    ema50: &[f64],
    volume_threshold: f64,
) -> Option<PatternResult> {
    let end_idx = zone.end_idx;
    let len = closes.len();
    
    // Беремо 5 свічок після compression для перевірки breakout
    if end_idx + 5 >= len {
        return None;
    }
    
    for i in (end_idx + 1)..=(end_idx + 5).min(len - 1) {
        let close = closes[i];
        let volume = volumes[i];
        let body_pct = (close - highs[i].min(lows[i])).abs() / highs[i].min(lows[i]) * 100.0;
        
        // Перевірка LONG breakout (пробій вгору)
        if close > zone.resistance 
            && volume > zone.avg_volume * volume_threshold
            && close > ema21[i]
            && body_pct > 0.5 
        {
            let entry = close;
            let sl = zone.support;
            let risk = entry - sl;
            let tp = entry + risk * 3.0;
            
            // Мінімальний потенціал 2%
            let tp_distance_pct = (tp - entry) / entry * 100.0;
            if tp_distance_pct < 2.0 {
                continue;
            }
            
            // Розрахунок confidence
            let vol_ratio = volume / zone.avg_volume;
            let mut confidence = 50.0;
            
            if vol_ratio > 2.0 {
                confidence += 20.0;
            } else if vol_ratio > 1.5 {
                confidence += 10.0;
            }
            
            if body_pct > 2.0 {
                confidence += 15.0;
            } else if body_pct > 1.0 {
                confidence += 10.0;
            }
            
            // EMA alignment
            if ema9[i] > ema21[i] && ema21[i] > ema50[i] {
                confidence += 15.0;
            }
            
            return Some(PatternResult {
                pattern_name: format!("Breakout LONG ({}c compression)", zone.duration),
                confidence: if confidence > 100.0 { 100.0 } else { confidence },
                direction: "LONG".to_string(),
                index: i,
                entry_price: entry,
                sl_price: sl,
                tp_price: tp,
                compression_candles: zone.duration,
            });
        }
        
        // Перевірка SHORT breakout (пробій вниз)
        if close < zone.support
            && volume > zone.avg_volume * volume_threshold
            && close < ema21[i]
            && body_pct > 0.5
        {
            let entry = close;
            let sl = zone.resistance;
            let risk = sl - entry;
            let tp = entry - risk * 3.0;
            
            // Мінімальний потенціал 2%
            let tp_distance_pct = (entry - tp) / entry * 100.0;
            if tp_distance_pct < 2.0 {
                continue;
            }
            
            // Розрахунок confidence
            let vol_ratio = volume / zone.avg_volume;
            let mut confidence = 50.0;
            
            if vol_ratio > 2.0 {
                confidence += 20.0;
            } else if vol_ratio > 1.5 {
                confidence += 10.0;
            }
            
            if body_pct > 2.0 {
                confidence += 15.0;
            } else if body_pct > 1.0 {
                confidence += 10.0;
            }
            
            // EMA alignment
            if ema9[i] < ema21[i] && ema21[i] < ema50[i] {
                confidence += 15.0;
            }
            
            return Some(PatternResult {
                pattern_name: format!("Breakout SHORT ({}c compression)", zone.duration),
                confidence: if confidence > 100.0 { 100.0 } else { confidence },
                direction: "SHORT".to_string(),
                index: i,
                entry_price: entry,
                sl_price: sl,
                tp_price: tp,
                compression_candles: zone.duration,
            });
        }
    }
    
    None
}

/// Детекція BREAKOUTS (головна функція для нової логіки)
#[pyfunction]
fn detect_breakouts(
    py: Python,
    _opens: Vec<f64>,
    highs: Vec<f64>,
    lows: Vec<f64>,
    closes: Vec<f64>,
    volumes: Vec<f64>,
    min_compression_candles: Option<usize>,
    max_range_pct: Option<f64>,
    volume_threshold: Option<f64>,
) -> PyResult<Vec<Py<PyDict>>> {
    let min_candles = min_compression_candles.unwrap_or(10);
    let max_range = max_range_pct.unwrap_or(1.5);
    let vol_thresh = volume_threshold.unwrap_or(1.3);
    
    // Розраховуємо EMA
    let ema9 = calculate_ema(&closes, 9);
    let ema21 = calculate_ema(&closes, 21);
    let ema50 = calculate_ema(&closes, 50);
    
    // Знаходимо compression zones
    let zones = find_compression_zones(&highs, &lows, &closes, &volumes, min_candles, max_range);
    
    // Перевіряємо кожну зону на breakout
    let mut breakouts = Vec::new();
    for zone in zones {
        if let Some(breakout) = check_breakout(
            &zone, 
            &highs, 
            &lows, 
            &closes, 
            &volumes,
            &ema9,
            &ema21,
            &ema50,
            vol_thresh
        ) {
            breakouts.push(breakout);
        }
    }
    
    // Конвертуємо в Python dict
    let py_results: Vec<Py<PyDict>> = breakouts
        .iter()
        .map(|r| {
            let dict = PyDict::new(py);
            dict.set_item("pattern_name", &r.pattern_name).unwrap();
            dict.set_item("confidence", r.confidence).unwrap();
            dict.set_item("direction", &r.direction).unwrap();
            dict.set_item("index", r.index).unwrap();
            dict.into()
        })
        .collect();
    
    Ok(py_results)
}

/// Головна функція детекції всіх патернів
#[pyfunction]
fn detect_patterns(
    py: Python,
    opens: Vec<f64>,
    highs: Vec<f64>,
    lows: Vec<f64>,
    closes: Vec<f64>,
    window: Option<usize>,
) -> PyResult<Vec<Py<PyDict>>> {
    let window = window.unwrap_or(10);
    let mut all_results = Vec::new();
    
    // 1. Double Top/Bottom
    all_results.extend(detect_double_top(&highs, window, 0.02));
    all_results.extend(detect_double_bottom(&lows, window, 0.02));
    
    // 2. Head and Shoulders
    all_results.extend(detect_head_and_shoulders(&highs, &lows, window));
    all_results.extend(detect_inverse_head_and_shoulders(&highs, &lows, window));
    
    // 3. Triangle
    all_results.extend(detect_triangle(&highs, &lows, window));
    
    // 4. Flag
    all_results.extend(detect_flag(&highs, &lows, &closes, window));
    
    // 5. Wedge
    all_results.extend(detect_wedge(&highs, &lows, window));
    
    // 6. Three Soldiers/Crows
    all_results.extend(detect_three_soldiers_crows(&opens, &closes));
    
    // Конвертуємо в Python dict
    let py_results: Vec<Py<PyDict>> = all_results
        .iter()
        .map(|r| {
            let dict = PyDict::new(py);
            dict.set_item("pattern_name", &r.pattern_name).unwrap();
            dict.set_item("confidence", r.confidence).unwrap();
            dict.set_item("direction", &r.direction).unwrap();
            dict.set_item("index", r.index).unwrap();
            dict.into()
        })
        .collect();
    
    Ok(py_results)
}

/// Швидка детекція останнього патерна (для real-time)
#[pyfunction]
fn detect_latest_pattern(
    py: Python,
    opens: Vec<f64>,
    highs: Vec<f64>,
    lows: Vec<f64>,
    closes: Vec<f64>,
) -> PyResult<Option<Py<PyDict>>> {
    let all_patterns = detect_patterns(py, opens, highs, lows, closes, Some(5))?;
    
    if all_patterns.is_empty() {
        return Ok(None);
    }
    
    // Повертаємо останній знайдений патерн
    Ok(Some(all_patterns.last().unwrap().clone()))
}

/// Модуль Python
#[pymodule]
fn pattern_detector(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(detect_patterns, m)?)?;
    m.add_function(wrap_pyfunction!(detect_latest_pattern, m)?)?;
    m.add_function(wrap_pyfunction!(detect_breakouts, m)?)?;  // НОВИЙ ДЕТЕКТОР
    m.add_class::<CandleData>()?;
    Ok(())
}
