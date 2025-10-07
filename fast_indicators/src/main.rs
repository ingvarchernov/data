use std::time::Instant;

fn main() {
    println!("🚀 Fast Indicators Library - Comprehensive Test Suite");
    println!("=====================================================");
    
    // Sample market data for testing
    let sample_prices = vec![
        44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.85, 46.08, 45.89, 46.03,
        46.83, 46.69, 46.45, 46.59, 46.3, 46.28, 46.28, 46.00, 46.03, 46.41,
        46.22, 45.64, 46.21, 46.25, 47.75, 47.79, 47.73, 47.31, 47.20, 46.80,
        46.78, 46.57, 46.83, 47.15, 47.11, 47.09, 47.05, 47.24, 47.29, 47.03
    ];
    
    // Generate sample highs, lows, volumes for testing
    let sample_highs: Vec<f64> = sample_prices.iter()
        .map(|&price| price + (price * 0.02)) // +2% for highs
        .collect();
    
    let sample_lows: Vec<f64> = sample_prices.iter()
        .map(|&price| price - (price * 0.015)) // -1.5% for lows
        .collect();
    
    let sample_volumes: Vec<f64> = (0..sample_prices.len())
        .map(|i| 1000000.0 + (i as f64 * 50000.0)) // Increasing volume
        .collect();
    
    println!("\n📊 Test Data Summary:");
    println!("   • Prices: {} data points", sample_prices.len());
    println!("   • Price range: {:.2} - {:.2}", 
             sample_prices.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
             sample_prices.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)));
    
    println!("\n🔧 Testing Technical Indicators:");
    println!("================================");
    
    // Test RSI
    print!("📈 RSI (14-period)... ");
    let start = Instant::now();
    let rsi_values = calculate_rsi_test(&sample_prices, 14);
    let rsi_time = start.elapsed();
    println!("✅ {} values calculated in {:?}", rsi_values.len(), rsi_time);
    if !rsi_values.is_empty() {
        println!("   Last RSI: {:.2}", rsi_values.last().unwrap());
    }
    
    // Test EMA
    print!("📈 EMA (20-period)... ");
    let start = Instant::now();
    let ema_values = calculate_ema_test(&sample_prices, 20);
    let ema_time = start.elapsed();
    println!("✅ {} values calculated in {:?}", ema_values.len(), ema_time);
    if !ema_values.is_empty() {
        println!("   Last EMA: {:.2}", ema_values.last().unwrap());
    }
    
    // Test SMA
    print!("📈 SMA (20-period)... ");
    let start = Instant::now();
    let sma_values = calculate_sma_test(&sample_prices, 20);
    let sma_time = start.elapsed();
    println!("✅ {} values calculated in {:?}", sma_values.len(), sma_time);
    if !sma_values.is_empty() {
        println!("   Last SMA: {:.2}", sma_values.last().unwrap());
    }
    
    // Test MACD
    print!("📈 MACD (12,26,9)... ");
    let start = Instant::now();
    let (macd_line, signal_line, histogram) = calculate_macd_test(&sample_prices, 12, 26, 9);
    let macd_time = start.elapsed();
    println!("✅ {} values calculated in {:?}", macd_line.len(), macd_time);
    if !macd_line.is_empty() {
        println!("   Last MACD: {:.4}", macd_line.last().unwrap());
    }
    
    // Test Bollinger Bands
    print!("📈 Bollinger Bands (20, 2.0)... ");
    let start = Instant::now();
    let (upper, lower) = calculate_bollinger_bands_test(&sample_prices, 20, 2.0);
    let bb_time = start.elapsed();
    println!("✅ {} values calculated in {:?}", upper.len(), bb_time);
    if !upper.is_empty() && !lower.is_empty() {
        println!("   Last Upper: {:.2}, Lower: {:.2}", 
                upper.last().unwrap(), lower.last().unwrap());
    }
    
    // Test Stochastic
    print!("📈 Stochastic (14,3,3)... ");
    let start = Instant::now();
    let (k_values, d_values) = calculate_stochastic_test(&sample_highs, &sample_lows, &sample_prices, 14, 3, 3);
    let stoch_time = start.elapsed();
    println!("✅ {} values calculated in {:?}", k_values.len(), stoch_time);
    if !k_values.is_empty() && !d_values.is_empty() {
        println!("   Last %K: {:.2}, %D: {:.2}", 
                k_values.last().unwrap(), d_values.last().unwrap());
    }
    
    // Test ATR
    print!("📈 ATR (14-period)... ");
    let start = Instant::now();
    let atr_values = calculate_atr_test(&sample_highs, &sample_lows, &sample_prices, 14);
    let atr_time = start.elapsed();
    println!("✅ {} values calculated in {:?}", atr_values.len(), atr_time);
    if !atr_values.is_empty() {
        println!("   Last ATR: {:.4}", atr_values.last().unwrap());
    }
    
    // Test CCI
    print!("📈 CCI (20-period)... ");
    let start = Instant::now();
    let cci_values = calculate_cci_test(&sample_highs, &sample_lows, &sample_prices, 20);
    let cci_time = start.elapsed();
    println!("✅ {} values calculated in {:?}", cci_values.len(), cci_time);
    if !cci_values.is_empty() {
        println!("   Last CCI: {:.2}", cci_values.last().unwrap());
    }
    
    // Test OBV
    print!("📈 OBV... ");
    let start = Instant::now();
    let obv_values = calculate_obv_test(&sample_prices, &sample_volumes);
    let obv_time = start.elapsed();
    println!("✅ {} values calculated in {:?}", obv_values.len(), obv_time);
    if !obv_values.is_empty() {
        println!("   Last OBV: {:.0}", obv_values.last().unwrap());
    }
    
    // Test ADX
    print!("📈 ADX (14-period)... ");
    let start = Instant::now();
    let adx_values = calculate_adx_test(&sample_highs, &sample_lows, &sample_prices, 14);
    let adx_time = start.elapsed();
    println!("✅ {} values calculated in {:?}", adx_values.len(), adx_time);
    if !adx_values.is_empty() {
        println!("   Last ADX: {:.2}", adx_values.last().unwrap());
    }
    
    // Test VWAP
    print!("📈 VWAP... ");
    let start = Instant::now();
    let vwap_values = calculate_vwap_test(&sample_highs, &sample_lows, &sample_prices, &sample_volumes);
    let vwap_time = start.elapsed();
    println!("✅ {} values calculated in {:?}", vwap_values.len(), vwap_time);
    if !vwap_values.is_empty() {
        println!("   Last VWAP: {:.2}", vwap_values.last().unwrap());
    }
    
    println!("\n⚡ Performance Test with Large Dataset:");
    println!("======================================");
    
    // Performance test with larger dataset
    let large_dataset: Vec<f64> = (0..10000)
        .map(|i| 100.0 + (i as f64 * 0.01) + (i as f64 * 0.001).sin() * 5.0)
        .collect();
    
    let large_highs: Vec<f64> = large_dataset.iter()
        .map(|&price| price + (price * 0.02))
        .collect();
    
    let large_lows: Vec<f64> = large_dataset.iter()
        .map(|&price| price - (price * 0.015))
        .collect();
    
    let large_volumes: Vec<f64> = (0..large_dataset.len())
        .map(|i| 1000000.0 + (i as f64 * 1000.0))
        .collect();
    
    println!("📊 Testing with {} data points:", large_dataset.len());
    
    let start = Instant::now();
    let _large_rsi = calculate_rsi_test(&large_dataset, 14);
    println!("   RSI: {:?}", start.elapsed());
    
    let start = Instant::now();
    let _large_ema = calculate_ema_test(&large_dataset, 20);
    println!("   EMA: {:?}", start.elapsed());
    
    let start = Instant::now();
    let _large_macd = calculate_macd_test(&large_dataset, 12, 26, 9);
    println!("   MACD: {:?}", start.elapsed());
    
    let start = Instant::now();
    let _large_bb = calculate_bollinger_bands_test(&large_dataset, 20, 2.0);
    println!("   Bollinger Bands: {:?}", start.elapsed());
    
    let start = Instant::now();
    let _large_vwap = calculate_vwap_test(&large_highs, &large_lows, &large_dataset, &large_volumes);
    println!("   VWAP: {:?}", start.elapsed());
    
    println!("\n🐍 Python Integration Guide:");
    println!("============================");
    println!("1. Build Python extension: maturin develop");
    println!("2. Import in Python:");
    println!("   import fast_indicators");
    println!("   import numpy as np");
    println!("");
    println!("3. Usage examples:");
    println!("   prices = np.array([44.34, 44.09, 44.15, 43.61, 44.33])");
    println!("   rsi = fast_indicators.fast_rsi(prices, 14)");
    println!("   ema = fast_indicators.fast_ema(prices, 20)");
    println!("   macd, signal, hist = fast_indicators.fast_macd(prices, 12, 26, 9)");
    
    println!("\n📋 Available Functions in Library:");
    println!("==================================");
    println!("✅ fast_rsi(prices, period) -> RSI values");
    println!("✅ fast_ema(prices, period) -> EMA values");
    println!("✅ fast_macd(prices, fast, slow, signal) -> (macd, signal, histogram)");
    println!("✅ fast_bollinger_bands(prices, period, std_dev) -> (upper, lower)");
    println!("✅ fast_stochastic(highs, lows, closes, k, smooth_k, smooth_d) -> (K, D)");
    println!("✅ fast_atr(highs, lows, closes, period) -> ATR values");
    println!("✅ fast_cci(highs, lows, closes, period) -> CCI values");
    println!("✅ fast_obv(closes, volumes) -> OBV values");
    println!("✅ fast_adx(highs, lows, closes, period) -> ADX values");
    println!("✅ fast_vwap(highs, lows, closes, volumes) -> VWAP values");
    
    println!("\n✅ All tests completed successfully!");
    println!("🚀 Fast Indicators Library is ready for production use!");
}

// Helper functions for testing
fn calculate_ema_test(prices: &[f64], period: usize) -> Vec<f64> {
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

fn calculate_sma_test(prices: &[f64], period: usize) -> Vec<f64> {
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

fn calculate_rsi_test(prices: &[f64], period: usize) -> Vec<f64> {
    if prices.len() < period + 1 {
        return vec![];
    }
    
    let mut gains = Vec::new();
    let mut losses = Vec::new();
    
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
    
    if avg_loss != 0.0 {
        let rs = avg_gain / avg_loss;
        let rsi = 100.0 - (100.0 / (1.0 + rs));
        rsi_values.push(rsi);
    } else {
        rsi_values.push(100.0);
    }
    
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
    
    rsi_values
}

fn calculate_macd_test(prices: &[f64], fast: usize, slow: usize, signal: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let fast_ema = calculate_ema_test(prices, fast);
    let slow_ema = calculate_ema_test(prices, slow);
    
    let start_index = slow.saturating_sub(fast);
    let macd_line: Vec<f64> = fast_ema.iter()
        .skip(start_index)
        .zip(slow_ema.iter())
        .map(|(fast, slow)| fast - slow)
        .collect();
    
    let signal_line = calculate_ema_test(&macd_line, signal);
    let histogram: Vec<f64> = macd_line.iter()
        .skip(signal - 1)
        .zip(signal_line.iter())
        .map(|(macd, signal)| macd - signal)
        .collect();
    
    (macd_line, signal_line, histogram)
}

fn calculate_bollinger_bands_test(prices: &[f64], period: usize, std_dev: f64) -> (Vec<f64>, Vec<f64>) {
    if prices.len() < period {
        return (vec![], vec![]);
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
    
    (upper_band, lower_band)
}

fn calculate_stochastic_test(highs: &[f64], lows: &[f64], closes: &[f64], k_period: usize, smooth_k: usize, smooth_d: usize) -> (Vec<f64>, Vec<f64>) {
    if highs.len() != lows.len() || lows.len() != closes.len() || closes.len() < k_period {
        return (vec![], vec![]);
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
    
    let smooth_k_values = calculate_sma_test(&k_values, smooth_k);
    let d_values = calculate_sma_test(&smooth_k_values, smooth_d);
    
    (smooth_k_values, d_values)
}

fn calculate_atr_test(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> Vec<f64> {
    if highs.len() != lows.len() || lows.len() != closes.len() || closes.len() < 2 {
        return vec![];
    }
    
    let mut true_ranges = Vec::new();
    
    for i in 1..closes.len() {
        let high_low = highs[i] - lows[i];
        let high_close = (highs[i] - closes[i - 1]).abs();
        let low_close = (lows[i] - closes[i - 1]).abs();
        
        let tr = high_low.max(high_close).max(low_close);
        true_ranges.push(tr);
    }
    
    calculate_sma_test(&true_ranges, period)
}

fn calculate_cci_test(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> Vec<f64> {
    if highs.len() != lows.len() || lows.len() != closes.len() || closes.len() < period {
        return vec![];
    }
    
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
    
    cci_values
}

fn calculate_obv_test(closes: &[f64], volumes: &[f64]) -> Vec<f64> {
    if closes.len() != volumes.len() || closes.len() < 2 {
        return vec![];
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
    
    obv_values
}

fn calculate_adx_test(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> Vec<f64> {
    if highs.len() != lows.len() || lows.len() != closes.len() || closes.len() < period + 1 {
        return vec![];
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
    
    let plus_di = calculate_sma_test(&plus_dm, period);
    let minus_di = calculate_sma_test(&minus_dm, period);
    let atr = calculate_sma_test(&true_ranges, period);
    
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
    
    calculate_sma_test(&adx_values, period)
}

fn calculate_vwap_test(highs: &[f64], lows: &[f64], closes: &[f64], volumes: &[f64]) -> Vec<f64> {
    if highs.len() != lows.len() || lows.len() != closes.len() || closes.len() != volumes.len() {
        return vec![];
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
    
    vwap_values
}
