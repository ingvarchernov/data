use std::time::Instant;

fn main() {
    println!("🚀 Fast Indicators - Production Performance Processor");
    println!("====================================================");
    
    //println!("🔧 Демо швидкодії оптимізованих технічних індикаторів:");
    //run_production_benchmark();
    
    println!("\n💡 Інтеграція з вашою системою:");
    println!("• Python модуль: import fast_indicators");
    println!("• Використовуйте db_utils.py та data_extraction.py для роботи з даними");
    println!("• Rust забезпечує максимальну швидкодію обчислень");
}
/*
fn run_production_benchmark() {
    let test_sizes = vec![1000, 10000, 100000];
    
    for size in test_sizes {
        println!("\n📊 Бенчмарк з {} свічками:", size);
        
        let test_data = generate_market_data(size);
        let start = Instant::now();
        
        // Швидкі розрахунки всіх індикаторів
        let _rsi = calculate_fast_rsi(&test_data.closes, 14);
        let _ema = calculate_fast_ema(&test_data.closes, 20);
        let (_macd, _signal, _hist) = calculate_fast_macd(&test_data.closes, 12, 26, 9);
        let (_upper, _lower) = calculate_fast_bollinger(&test_data.closes, 20, 2.0);
        let _vwap = calculate_fast_vwap(&test_data.highs, &test_data.lows, &test_data.closes, &test_data.volumes);
        let _atr = calculate_fast_atr(&test_data.highs, &test_data.lows, &test_data.closes, 14);
        let _cci = calculate_fast_cci(&test_data.highs, &test_data.lows, &test_data.closes, 20);
        
        let duration = start.elapsed();
        let ops_per_sec = size as f64 / duration.as_secs_f64();
        
        println!("   ⚡ Час виконання: {:?}", duration);
        println!("   🚄 Обробка: {:.0} свічок/сек", ops_per_sec);
    }
}
*/
struct MarketData {
    highs: Vec<f64>,
    lows: Vec<f64>,
    closes: Vec<f64>,
    volumes: Vec<f64>,
}
/*
fn generate_market_data(size: usize) -> MarketData {
    let mut data = MarketData {
        highs: Vec::with_capacity(size),
        lows: Vec::with_capacity(size),
        closes: Vec::with_capacity(size),
        volumes: Vec::with_capacity(size),
    };
    
    let mut price = 50000.0;
    
    for i in 0..size {
        let trend = (i as f64 * 0.001).sin() * 0.001;
        let volatility = ((i as f64 * 0.05).cos()).abs() * 0.02;
        let noise = ((i as f64 * 17.0).sin() * (i as f64 * 23.0).cos()) * 0.001;
        
        price *= 1.0 + trend + (noise * volatility);
        
        data.highs.push(price * (1.0 + volatility * 0.5));
        data.lows.push(price * (1.0 - volatility * 0.5));
        data.closes.push(price);
        data.volumes.push(1000.0 + (i as f64 * 7.0).sin().abs() * 5000.0);
    }
    
    data
}
 */
// Оптимізовані алгоритми для максимальної швидкодії
fn calculate_fast_rsi(prices: &[f64], period: usize) -> Vec<f64> {
    if prices.len() <= period {
        return vec![];
    }
    
    let mut gains = 0.0;
    let mut losses = 0.0;
    
    // Початковий період
    for i in 1..=period {
        let change = prices[i] - prices[i - 1];
        if change > 0.0 {
            gains += change;
        } else {
            losses -= change;
        }
    }
    
    let mut avg_gain = gains / period as f64;
    let mut avg_loss = losses / period as f64;
    let mut rsi_values = Vec::with_capacity(prices.len() - period);
    
    // Перший RSI
    if avg_loss != 0.0 {
        let rs = avg_gain / avg_loss;
        rsi_values.push(100.0 - (100.0 / (1.0 + rs)));
    } else {
        rsi_values.push(100.0);
    }
    
    // Решта значень з експоненціальним згладжуванням
    let alpha = 1.0 / period as f64;
    for i in (period + 1)..prices.len() {
        let change = prices[i] - prices[i - 1];
        let gain = if change > 0.0 { change } else { 0.0 };
        let loss = if change < 0.0 { -change } else { 0.0 };
        
        avg_gain = alpha * gain + (1.0 - alpha) * avg_gain;
        avg_loss = alpha * loss + (1.0 - alpha) * avg_loss;
        
        if avg_loss != 0.0 {
            let rs = avg_gain / avg_loss;
            rsi_values.push(100.0 - (100.0 / (1.0 + rs)));
        } else {
            rsi_values.push(100.0);
        }
    }
    
    rsi_values
}

fn calculate_fast_ema(prices: &[f64], period: usize) -> Vec<f64> {
    if prices.is_empty() {
        return vec![];
    }
    
    let alpha = 2.0 / (period as f64 + 1.0);
    let one_minus_alpha = 1.0 - alpha;
    let mut ema_values = Vec::with_capacity(prices.len());
    
    ema_values.push(prices[0]);
    
    for i in 1..prices.len() {
        let ema = alpha * prices[i] + one_minus_alpha * ema_values[i - 1];
        ema_values.push(ema);
    }
    
    ema_values
}

fn calculate_fast_macd(prices: &[f64], fast: usize, slow: usize, signal: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let fast_ema = calculate_fast_ema(prices, fast);
    let slow_ema = calculate_fast_ema(prices, slow);
    
    let macd_line: Vec<f64> = fast_ema.iter()
        .zip(slow_ema.iter())
        .map(|(f, s)| f - s)
        .collect();
    
    let signal_line = calculate_fast_ema(&macd_line, signal);
    
    let histogram: Vec<f64> = macd_line.iter()
        .zip(signal_line.iter())
        .map(|(m, s)| m - s)
        .collect();
    
    (macd_line, signal_line, histogram)
}

fn calculate_fast_bollinger(prices: &[f64], period: usize, std_multiplier: f64) -> (Vec<f64>, Vec<f64>) {
    if prices.len() < period {
        return (vec![], vec![]);
    }
    
    let mut upper_bands = Vec::with_capacity(prices.len() - period + 1);
    let mut lower_bands = Vec::with_capacity(prices.len() - period + 1);
    
    let period_f = period as f64;
    
    for i in (period - 1)..prices.len() {
        let window = &prices[(i + 1 - period)..=i];
        
        // Швидке обчислення середнього
        let sum: f64 = window.iter().sum();
        let mean = sum / period_f;
        
        // Швидке обчислення стандартного відхилення
        let variance: f64 = window.iter()
            .map(|x| {
                let diff = x - mean;
                diff * diff
            })
            .sum::<f64>() / period_f;
        
        let std_dev = variance.sqrt();
        let band_width = std_multiplier * std_dev;
        
        upper_bands.push(mean + band_width);
        lower_bands.push(mean - band_width);
    }
    
    (upper_bands, lower_bands)
}

fn calculate_fast_vwap(highs: &[f64], lows: &[f64], closes: &[f64], volumes: &[f64]) -> Vec<f64> {
    if highs.len() != lows.len() || lows.len() != closes.len() || closes.len() != volumes.len() {
        return vec![];
    }
    
    let mut vwap_values = Vec::with_capacity(closes.len());
    let mut cumulative_pv = 0.0;
    let mut cumulative_volume = 0.0;
    
    for i in 0..closes.len() {
        let typical_price = (highs[i] + lows[i] + closes[i]) / 3.0;
        cumulative_pv += typical_price * volumes[i];
        cumulative_volume += volumes[i];
        
        if cumulative_volume != 0.0 {
            vwap_values.push(cumulative_pv / cumulative_volume);
        } else {
            vwap_values.push(typical_price);
        }
    }
    
    vwap_values
}

fn calculate_fast_atr(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> Vec<f64> {
    if highs.len() != lows.len() || lows.len() != closes.len() || closes.len() < 2 {
        return vec![];
    }
    
    let mut true_ranges = Vec::with_capacity(closes.len() - 1);
    
    for i in 1..closes.len() {
        let high_low = highs[i] - lows[i];
        let high_close = (highs[i] - closes[i - 1]).abs();
        let low_close = (lows[i] - closes[i - 1]).abs();
        
        let tr = high_low.max(high_close).max(low_close);
        true_ranges.push(tr);
    }
    
    calculate_fast_sma(&true_ranges, period)
}

fn calculate_fast_cci(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> Vec<f64> {
    if highs.len() != lows.len() || lows.len() != closes.len() || closes.len() < period {
        return vec![];
    }
    
    let typical_prices: Vec<f64> = (0..closes.len())
        .map(|i| (highs[i] + lows[i] + closes[i]) / 3.0)
        .collect();
    
    let mut cci_values = Vec::with_capacity(typical_prices.len() - period + 1);
    let period_f = period as f64;
    
    for i in (period - 1)..typical_prices.len() {
        let window = &typical_prices[(i + 1 - period)..=i];
        
        let sum: f64 = window.iter().sum();
        let sma = sum / period_f;
        
        let mean_deviation: f64 = window.iter()
            .map(|x| (x - sma).abs())
            .sum::<f64>() / period_f;
        
        if mean_deviation != 0.0 {
            let cci = (typical_prices[i] - sma) / (0.015 * mean_deviation);
            cci_values.push(cci);
        } else {
            cci_values.push(0.0);
        }
    }
    
    cci_values
}

fn calculate_fast_sma(prices: &[f64], period: usize) -> Vec<f64> {
    if prices.len() < period {
        return vec![];
    }
    
    let mut sma_values = Vec::with_capacity(prices.len() - period + 1);
    let period_f = period as f64;
    
    for i in (period - 1)..prices.len() {
        let sum: f64 = prices[(i + 1 - period)..=i].iter().sum();
        sma_values.push(sum / period_f);
    }
    
    sma_values
}
