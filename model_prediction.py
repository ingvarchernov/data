# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
import os
import matplotlib.pyplot as plt
from datetime import datetime
from data_extraction import get_historical_data
from technical_indicators import (
    calculate_rsi, calculate_macd, calculate_bollinger_bands,
    calculate_stochastic, calculate_ema, calculate_atr,
    calculate_cci, calculate_obv, calculate_adx, calculate_vwap
)
from db_utils import get_historical_data_from_db, get_scaler_stats, insert_prediction, insert_symbol, insert_interval, insert_technical_indicators, insert_normalized_data, check_technical_indicators_exists
import glob

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def mape_metric(y_true, y_pred, target_mean, target_std):
    y_true_denorm = y_true * target_std + target_mean
    y_pred_denorm = y_pred * target_std + target_mean
    mape = tf.reduce_mean(tf.abs((y_true_denorm - y_pred_denorm) / tf.abs(y_true_denorm + 1e-6))) * 100
    return mape

def load_and_prepare_data(symbol, interval, days_back, look_back, api_key, api_secret):
    logger.info(f"Завантаження даних для {symbol} з інтервалом {interval} за {days_back} днів.")
    data = get_historical_data_from_db(symbol, interval, days_back, api_key, api_secret)
    if data.empty:
        logger.error("DataFrame порожній після завантаження даних.")
        raise ValueError("DataFrame порожній після завантаження даних.")

    logger.debug(f"Розмір DataFrame після завантаження: {data.shape}")
    if 'timestamp' in data.columns:
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        data.set_index('timestamp', inplace=True)
    elif not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index, unit='ms', errors='coerce')

    data = data[(data['close'] > 0) & (data['close'].pct_change().abs() < 0.1)].copy()

    # Зберігаємо data_id для використання
    data_ids = data['data_id'].values
    if data_ids is None or len(data_ids) == 0:
        logger.error("Стовпець data_id відсутній у даних.")
        return None, None, None, None, None, None
    data_ids = [int(x) for x in data_ids]

    data['close_lag1'] = data['close'].shift(1)
    data['close_lag2'] = data['close'].shift(2)
    data['close_diff'] = data['close'].diff()
    data['log_return'] = np.log(data['close'] / data['close_lag1'])
    if isinstance(data.index, pd.DatetimeIndex):
        data['hour_norm'] = (data.index.hour + data.index.minute / 60) / 24
        data['day_norm'] = (data.index.dayofweek + data.index.hour / 24) / 7
    else:
        data['hour_norm'] = np.sin(2 * np.pi * np.arange(len(data)) / 24)
        data['day_norm'] = np.sin(2 * np.pi * np.arange(len(data)) / (24 * 7))

    logger.info(f"Розрахунок технічних індикаторів для {symbol}.")
    try:
        data['RSI'] = calculate_rsi(data)
        data['MACD'], data['MACD_Signal'] = calculate_macd(data[['close']])
        data['Upper_Band'], data['Lower_Band'] = calculate_bollinger_bands(data)
        data['Stoch'], data['Stoch_Signal'] = calculate_stochastic(data)
        data['EMA'] = calculate_ema(data)
        data['ATR'] = calculate_atr(data)
        data['CCI'] = calculate_cci(data)
        data['OBV'] = calculate_obv(data)
        data['Volatility'] = data['close'].rolling(window=14).std()
        data['ADX'] = calculate_adx(data)
        data['VWAP'] = calculate_vwap(data)
        data['Volume_Pct'] = data['volume'] / data['volume'].rolling(window=24).sum()

        # Збереження технічних індикаторів
        symbol_id = insert_symbol(symbol)
        interval_id = insert_interval(interval)
        for i, (idx, row) in enumerate(data.iterrows()):
            if i >= len(data_ids):
                logger.warning(f"Недостатньо data_id для індексу {idx}. Пропускаю.")
                continue
            data_id = data_ids[i]
            if check_technical_indicators_exists(data_id):
                logger.debug(f"Технічні індикатори для data_id={data_id} уже існують, оновлюємо.")
            else:
                logger.debug(f"Вставка нових технічних індикаторів для data_id={data_id}.")

            indicators = {
                'data_id': data_id,
                'rsi': None if pd.isna(row['RSI']) else float(row['RSI']),
                'macd': None if pd.isna(row['MACD']) else float(row['MACD']),
                'macd_signal': None if pd.isna(row['MACD_Signal']) else float(row['MACD_Signal']),
                'upper_band': None if pd.isna(row['Upper_Band']) else float(row['Upper_Band']),
                'lower_band': None if pd.isna(row['Lower_Band']) else float(row['Lower_Band']),
                'stoch': None if pd.isna(row['Stoch']) else float(row['Stoch']),
                'stoch_signal': None if pd.isna(row['Stoch_Signal']) else float(row['Stoch_Signal']),
                'ema': None if pd.isna(row['EMA']) else float(row['EMA']),
                'atr': None if pd.isna(row['ATR']) else float(row['ATR']),
                'cci': None if pd.isna(row['CCI']) else float(row['CCI']),
                'obv': None if pd.isna(row['OBV']) else float(row['OBV']),
                'volatility': None if pd.isna(row['Volatility']) else float(row['Volatility']),
                'volume_pct': None if pd.isna(row['Volume_Pct']) else float(row['Volume_Pct']),
                'close_lag1': None if pd.isna(row['close_lag1']) else float(row['close_lag1']),
                'close_lag2': None if pd.isna(row['close_lag2']) else float(row['close_lag2']),
                'close_diff': None if pd.isna(row['close_diff']) else float(row['close_diff']),
                'log_return': None if pd.isna(row['log_return']) else float(row['log_return']),
                'hour_norm': None if pd.isna(row['hour_norm']) else float(row['hour_norm']),
                'day_norm': None if pd.isna(row['day_norm']) else float(row['day_norm']),
                'adx': None if pd.isna(row['ADX']) else float(row['ADX']),
                'vwap': None if pd.isna(row['VWAP']) else float(row['VWAP'])
            }
            if all(v is None for k, v in indicators.items() if k not in ['data_id', 'hour_norm', 'day_norm']):
                logger.debug(f"Пропускаю запис для data_id={data_ids[i]} через всі NaN.")
                continue
            insert_technical_indicators(data_id, indicators)
    except Exception as e:
        logger.error(f"Помилка при розрахунку технічних індикаторів: {e}")
        return None, None, None, None, None, None

    data = data.iloc[100:].dropna()
    if data.empty or len(data) < look_back:
        logger.error(f"Недостатньо даних після обробки для {symbol}. Залишилось {len(data)} рядків, потрібно {look_back}.")
        return None, None, None, None, None, None

    features = ['close', 'volume', 'quote_av', 'trades', 'RSI', 'MACD', 'MACD_Signal',
                'Upper_Band', 'Lower_Band', 'Stoch', 'Stoch_Signal', 'EMA', 'ATR', 'CCI', 'OBV', 'Volatility',
                'ADX', 'VWAP', 'Volume_Pct', 'close_lag1', 'close_lag2', 'close_diff', 'log_return', 'hour_norm', 'day_norm']

    scalers = {feat: tf.keras.layers.Normalization(axis=-1) for feat in features}
    for feat in features:
        scalers[feat].adapt(data[feat].values.reshape(-1, 1))
    scaled_data = np.column_stack([scalers[feat](data[feat].values.reshape(-1, 1)).numpy() for feat in features])

    # Збереження нормалізованих даних
    for i in range(len(scaled_data)):
        if i >= len(data_ids):
            logger.warning(f"Недостатньо data_id для нормалізованих даних з індексом {i}. Пропускаю.")
            continue
        normalized_values = {feat: scaled_data[i][j] for j, feat in enumerate(features)}
        insert_normalized_data(data_ids[i], normalized_values)

    X = scaled_data[-look_back:].reshape(1, look_back, len(features))
    last_price = data['close'].iloc[-1]
    last_timestamp = data.index[-1]

    return X, last_price, data, features, last_timestamp, scalers

def load_models(n_splits=5, target_mean=None, target_std=None, symbol=None, interval=None):
    if symbol is None or interval is None:
        logger.error("Параметри symbol і interval є обов'язковими для завантаження моделей.")
        return []

    models = []
    for fold in range(1, n_splits + 1):
        finetuned_pattern = os.path.join(BASE_DIR, f'lstm_model_fold_{fold}_finetuned_{symbol}_{interval}_*.keras')
        finetuned_models = glob.glob(finetuned_pattern)

        if finetuned_models:
            latest_finetuned_model_path = max(finetuned_models, key=os.path.getctime)
            try:
                model = tf.keras.models.load_model(latest_finetuned_model_path, custom_objects={
                    'mape_metric': lambda x, y: mape_metric(x, y, target_mean, target_std)
                }, safe_mode=False)
                models.append(model)
                logger.info(f"Завантажено найновішу донавчану модель для Fold {fold} з {latest_finetuned_model_path}.")
                continue
            except Exception as e:
                logger.error(f"Не вдалося завантажити донавчану модель для Fold {fold} з {latest_finetuned_model_path}: {e}")

        base_model_path = os.path.join(BASE_DIR, f'lstm_model_fold_{fold}_{symbol}_{interval}.keras')
        if not os.path.exists(base_model_path):
            logger.error(f"Файл базової моделі {base_model_path} не знайдено.")
            continue

        try:
            model = tf.keras.models.load_model(base_model_path, custom_objects={
                'mape_metric': lambda x, y: mape_metric(x, y, target_mean, target_std)
            }, safe_mode=False)
            models.append(model)
            logger.info(f"Базова модель для Fold {fold} успішно завантажена з {base_model_path}.")
        except Exception as e:
            logger.error(f"Не вдалося завантажити базову модель для Fold {fold} з {base_model_path}: {e}")

    if not models:
        logger.error(f"Не вдалося завантажити жодної моделі для {symbol}_{interval}.")

    return models

def predict_price_multi_step(models, X_initial, last_price, target_mean, target_std, steps, features, data, scalers, interval):
    predictions_per_model = {f'fold_{i+1}': [] for i in range(len(models))}
    current_X = X_initial.copy()
    current_price = float(last_price)  # Конвертація last_price у float
    all_timestamps = []

    interval_map = {'5m': 5, '15m': 15, '30m': 30, '1h': 60, '1w': 7*24*60, '1M': 30*24*60}
    minutes_per_step = interval_map.get(interval, 60)

    for step in range(steps):
        step_predictions = []
        for i, model in enumerate(models):
            pred = model.predict(current_X, verbose=0)
            pred_scalar = float(pred.flatten()[0])  # Конвертація прогнозу у float
            real_diff_pred = (pred_scalar * float(target_std)) + float(target_mean)
            real_pred = current_price + real_diff_pred
            step_predictions.append(real_pred)
            predictions_per_model[f'fold_{i+1}'].append(real_pred)

        avg_pred = float(np.mean(step_predictions))  # Конвертація середнього у float
        pred_std = float(np.std(step_predictions))  # Конвертація стандартного відхилення у float

        last_timestamp = data.index[-1] + pd.Timedelta(minutes=minutes_per_step * (step + 1))
        all_timestamps.append(last_timestamp)

        logger.info(f"Крок {step + 1}: Прогноз = {avg_pred:.2f} USDT (std: {pred_std:.2f})")

        new_row = update_input_data(current_X[0, -1, :], avg_pred, data, features, scalers, step, interval)
        current_X = np.roll(current_X, -1, axis=1)
        current_X[0, -1, :] = new_row
        current_price = avg_pred

    return predictions_per_model, all_timestamps

def update_input_data(last_row, predicted_price, data, features, scalers, step, interval):
    new_row = last_row.copy()
    last_close = data['close'].iloc[-1] + (last_row[features.index('close_diff')] * np.sqrt(scalers['close_diff'].variance.numpy()[0]) + scalers['close_diff'].mean.numpy()[0])

    # Конвертація predicted_price у масив NumPy перед reshape
    predicted_price_array = np.array(predicted_price).reshape(-1, 1)
    new_row[features.index('close')] = scalers['close'](predicted_price_array).numpy()[0, 0]

    # Аналогічно для інших значень
    last_close_array = np.array(last_close).reshape(-1, 1)
    new_row[features.index('close_lag1')] = scalers['close_lag1'](last_close_array).numpy()[0, 0]
    new_row[features.index('close_lag2')] = last_row[features.index('close_lag1')]

    close_diff_array = np.array(predicted_price - last_close).reshape(-1, 1)
    new_row[features.index('close_diff')] = scalers['close_diff'](close_diff_array).numpy()[0, 0]

    log_return_array = np.array(np.log(predicted_price / last_close + 1e-6)).reshape(-1, 1)
    new_row[features.index('log_return')] = scalers['log_return'](log_return_array).numpy()[0, 0]

    interval_map = {'5m': 5, '15m': 15, '30m': 30, '1h': 60, '1w': 7*24*60, '1M': 30*24*60}
    minutes_per_step = interval_map.get(interval, 60)
    next_timestamp = data.index[-1] + pd.Timedelta(minutes=minutes_per_step * (step + 1))
    next_hour = next_timestamp.hour + next_timestamp.minute / 60
    next_day = next_timestamp.dayofweek + next_hour / 24
    new_row[features.index('hour_norm')] = next_hour / 24
    new_row[features.index('day_norm')] = next_day / 7

    for feat in ['volume', 'quote_av', 'trades', 'RSI', 'MACD', 'MACD_Signal', 'Upper_Band', 'Lower_Band',
                 'Stoch', 'Stoch_Signal', 'EMA', 'ATR', 'CCI', 'OBV', 'Volatility', 'ADX', 'VWAP', 'Volume_Pct']:
        new_row[features.index(feat)] = last_row[features.index(feat)]

    return new_row

def plot_predictions(historical_data, predictions_df, symbol, interval):
    plt.figure(figsize=(12, 6))
    plt.plot(historical_data.index, historical_data['close'], label='Історичні ціни', color='blue')
    plt.plot(predictions_df['timestamp'], predictions_df['predicted_price'], label='Прогнозовані ціни', color='red', marker='o')
    plt.title(f'Прогноз цін для {symbol} ({interval})')
    plt.xlabel('Дата')
    plt.ylabel('Ціна (USDT)')
    plt.legend()
    plt.grid()
    plot_path = os.path.join(BASE_DIR, f'prediction_plot_{symbol}_{interval}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Графік прогнозів збережено у {plot_path}")

def main(symbol, interval, days_back, look_back, steps, api_key, api_secret):
    # Перевірка наявності моделей
    # Перевірка наявності моделей
    model_exists = any(os.path.exists(os.path.join(BASE_DIR, f'lstm_model_fold_{fold}_{symbol}_{interval}.keras')) for fold in range(1, 6))

    # Перевірка наявності scaler_stats
    symbol_id = insert_symbol(symbol)
    interval_id = insert_interval(interval)
    target_mean, target_std = get_scaler_stats(symbol_id, interval_id)

    if target_mean is None or target_std is None or not model_exists:
        logger.info(f"Статистика масштабування або моделі для {symbol}_{interval} відсутні. Запускаю тренування.")
        from model_training import train_lstm_model
        try:
            train_lstm_model(symbol=symbol, interval=interval, days_back=days_back, look_back=look_back, api_key=api_key, api_secret=api_secret)
        except Exception as e:
            logger.error(f"Помилка під час тренування: {e}")
            raise

        # Повторна перевірка після тренування
        target_mean, target_std = get_scaler_stats(symbol_id, interval_id)
        if target_mean is None or target_std is None:
            logger.error(f"Після тренування все ще не знайдено scaler_stats для {symbol}_{interval}. Перевірте логи тренування.")
            raise ValueError("Не вдалося зберегти статистику масштабування після тренування.")

    # Завантаження даних після тренування
    X, last_price, data, features, last_timestamp, scalers = load_and_prepare_data(symbol, interval, days_back, look_back, api_key, api_secret)
    if X is None:
        logger.error("Не вдалося підготувати дані для прогнозування. Перевірте логи щодо завантаження даних.")
        raise ValueError("Не вдалося підготувати дані для прогнозування.")

    logger.info(f"Дані для прогнозування: {len(data)} рядків, діапазон дат від {data.index.min()} до {data.index.max()}")
    logger.info(f"Останній timestamp: {last_timestamp}, остання ціна: {last_price}")

    models = load_models(n_splits=5, target_mean=target_mean, target_std=target_std, symbol=symbol, interval=interval)
    if not models:
        logger.error("Не вдалося завантажити жодної моделі після тренування. Перевірте логи тренування.")
        raise ValueError("Не вдалося завантажити жодної моделі після тренування.")

    try:
        predictions_per_model, timestamps = predict_price_multi_step(
            models, X, last_price, target_mean, target_std, steps, features, data, scalers, interval
        )
    except Exception as e:
        logger.error(f"Помилка при прогнозуванні: {e}")
        raise

    logger.info("\nПрогнози цін:")
    prediction_rows = []
    for step in range(steps):
        fold_predictions = {f'fold_{i+1}': predictions_per_model[f'fold_{i+1}'][step] for i in range(len(models))}
        predicted_price = np.mean([fold_predictions[f'fold_{i+1}'] for i in range(len(models))])
        logger.info(f"Крок {step+1} ({timestamps[step]}): Прогнозована ціна = {predicted_price:.2f} USDT, [Розкид між фолдами: {np.std(list(fold_predictions.values())):.2f}]")

        prediction_rows.append({
            'timestamp': timestamps[step],
            'predicted_price': predicted_price,
            **fold_predictions
        })

        prediction_data = {
            'symbol_id': symbol_id,
            'interval_id': interval_id,
            'timestamp': timestamps[step],
            'last_price': last_price if step == 0 else None,
            'predicted_price': predicted_price,
            **{f'fold_{i+1}_prediction': fold_predictions[f'fold_{i+1}'] for i in range(len(models))}
        }
        logger.debug(f"Зберігаю прогноз: {prediction_data}")
        insert_prediction(prediction_data)

    predictions_df = pd.DataFrame(prediction_rows)
    predictions_csv_path = os.path.join(BASE_DIR, f'predictions_{symbol}_{interval}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    predictions_df.to_csv(predictions_csv_path, index=False)
    logger.info(f"Прогнози збережено у {predictions_csv_path}")

    plot_predictions(data, predictions_df, symbol, interval)

    logger.info(f"Прогнози збережено в PostgreSQL для {symbol} ({interval}).")