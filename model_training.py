# -*- coding: utf-8 -*-
import time
import numpy as np
import logging
from keras import layers
from sklearn.model_selection import KFold
from technical_indicators import (
    calculate_rsi, calculate_macd, calculate_bollinger_bands,
    calculate_stochastic, calculate_ema, calculate_atr,
    calculate_cci, calculate_obv, calculate_adx, calculate_vwap
)
from data_extraction import get_historical_data
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
from db_utils import insert_symbol, insert_interval, insert_training_history, insert_technical_indicators, insert_normalized_data, insert_scaler_stats, check_technical_indicators_exists, get_historical_data_from_db

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def prepare_data(symbol, interval, days_back, look_back, api_key, api_secret, features=None):
    """Підготовка даних для тренування."""
    data = get_historical_data_from_db(symbol, interval, days_back, api_key, api_secret)
    if data is None or data.empty:
        logger.error(f"Дані для {symbol} не завантажено.")
        return None, None, None, None, None

    if 'timestamp' in data.columns:
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        data.set_index('timestamp', inplace=True)
    elif not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index, unit='ms', errors='coerce')

    logger.debug(f"Початкова кількість рядків: {len(data)}")
    data = data[(data['close'] > 0) & (data['close'].pct_change().abs() < 0.1)].copy()
    logger.debug(f"Після фільтрації аномалій: {len(data)}")

    data_ids = data['data_id'].values
    if data_ids is None or len(data_ids) == 0:
        logger.error("Стовпець data_id відсутній у даних.")
        return None, None, None, None, None

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

    logger.info("Розрахунок технічних індикаторів.")
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
        data['Volume_Pct'] = data['volume'] / data['volume'].rolling(window=14).sum()

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
        logger.error(f"Помилка при розрахунку технічних індикаторів: {str(e)}")
        raise

    logger.debug(f"Перед dropna: {len(data)}")
    data = data.dropna()
    logger.debug(f"Після dropna: {len(data)}")
    if data.empty or len(data) <= look_back:
        logger.error(f"Недостатньо даних після обробки для look_back={look_back}. Залишилось {len(data)} рядків.")
        return None, None, None, None, None

    if features is None:
        features = [
            'close', 'volume', 'quote_av', 'trades', 'RSI', 'MACD', 'MACD_Signal',
            'Upper_Band', 'Lower_Band', 'Stoch', 'Stoch_Signal', 'EMA', 'ATR', 'CCI', 'OBV', 'Volatility',
            'ADX', 'VWAP', 'Volume_Pct', 'close_lag1', 'close_lag2', 'close_diff', 'log_return', 'hour_norm', 'day_norm'
        ]

    scalers = {feat: tf.keras.layers.Normalization(axis=-1) for feat in features}
    for feat in features:
        scalers[feat].adapt(data[feat].values.reshape(-1, 1))
    scaled_data = np.column_stack([scalers[feat](data[feat].values.reshape(-1, 1)).numpy() for feat in features])

    for i in range(len(scaled_data)):
        if i >= len(data_ids):
            logger.warning(f"Недостатньо data_id для нормалізованих даних з індексом {i}. Пропускаю.")
            continue
        normalized_values = {feat: scaled_data[i][j] for j, feat in enumerate(features)}
        insert_normalized_data(data_ids[i], normalized_values)

    target_scaler = tf.keras.layers.Normalization(axis=-1)
    target_scaler.adapt(data['close_diff'].values.reshape(-1, 1))
    target = target_scaler(data['close_diff'].values.reshape(-1, 1)).numpy().flatten()

    return scaled_data, target, features, target_scaler, data

def append_missing_data(existing_data, symbol, interval, api_key, api_secret):
    logger.info("Додавання відсутніх даних до набору.")
    if 'timestamp' in existing_data.columns:
        existing_data['timestamp'] = pd.to_datetime(existing_data['timestamp'], unit='ms')
        existing_data.set_index('timestamp', inplace=True)

    last_timestamp = existing_data.index.max()
    logger.debug(f"Останній запис у даних: {last_timestamp}")

    current_time = pd.Timestamp.now(tz=last_timestamp.tz)
    logger.debug(f"Поточний час: {current_time}")

    hours_diff = int((current_time - last_timestamp).total_seconds() / 3600) + 1
    if hours_diff <= 0:
        logger.info("Дані вже актуальні, додаткове завантаження не потрібне.")
        return existing_data

    logger.info(f"Завантажую нові дані за {hours_diff} годин.")
    new_data = get_historical_data(symbol, interval=interval, days_back=hours_diff / 24, api_key=api_key, api_secret=api_secret)

    if new_data is None or new_data.empty:
        logger.error("Не вдалося завантажити нові дані.")
        return existing_data

    if 'timestamp' in new_data.columns:
        new_data['timestamp'] = pd.to_datetime(new_data['timestamp'], unit='ms')
        new_data.set_index('timestamp', inplace=True)

    combined_data = pd.concat([existing_data, new_data])
    combined_data = combined_data[~combined_data.index.duplicated(keep='last')].sort_index()

    logger.debug(f"Об’єднано дані: {len(combined_data)} рядків.")
    return combined_data

def create_dataset(scaled_data, target, look_back):
    X, y = [], []
    for i in range(look_back, len(scaled_data) - 1):
        X.append(scaled_data[i - look_back:i])
        y.append(target[i])
    return np.array(X), np.array(y)

def mape_metric(y_true, y_pred, target_mean, target_std):
    y_true_denorm = y_true * target_std + target_mean
    y_pred_denorm = y_pred * target_std + target_mean
    mape = tf.reduce_mean(tf.abs((y_true_denorm - y_pred_denorm) / tf.abs(y_true_denorm + 1e-6))) * 100
    return mape

def build_model_with_attention(input_shape):
    inputs = layers.Input(shape=(None, input_shape[-1]))
    norm_1 = layers.LayerNormalization()(inputs)
    lstm_1 = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.3,
                                              kernel_regularizer=tf.keras.regularizers.l2(0.001)))(norm_1)
    norm_2 = layers.LayerNormalization()(lstm_1)
    attention_1 = layers.MultiHeadAttention(num_heads=4, key_dim=64)(norm_2, norm_2)
    lstm_2 = layers.LSTM(128, return_sequences=True, dropout=0.3)(attention_1)
    norm_3 = layers.LayerNormalization()(lstm_2)
    pool_out = layers.GlobalAveragePooling1D()(norm_3)
    dense_1 = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(pool_out)
    norm_4 = layers.BatchNormalization()(dense_1)
    dropout_1 = layers.Dropout(0.3)(norm_4)
    dense_2 = layers.Dense(64, activation='relu')(dropout_1)
    outputs = layers.Dense(1, activation='linear', bias_initializer='zeros')(dense_2)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def train_lstm_model(symbol, interval, days_back, look_back, api_key, api_secret, epochs=200, batch_size=16, n_splits=5):
    logger.info(f"Завантаження історичних даних для {symbol} з інтервалом {interval} за {days_back} днів.")
    scaled_data, target, features, target_scaler, data = prepare_data(symbol, interval, days_back, look_back, api_key, api_secret)
    if scaled_data is None:
        return None, None

    target_mean = target_scaler.mean.numpy()[0]
    target_std = np.sqrt(target_scaler.variance.numpy()[0])
    logger.info(f"Зберігаю scaler stats: target_mean={target_mean}, target_std={target_std}")

    symbol_id = insert_symbol(symbol)
    interval_id = insert_interval(interval)
    insert_scaler_stats(symbol_id, interval_id, target_mean, target_std)

    scaler_stats = pd.DataFrame({
        'target_mean': [target_mean],
        'target_std': [target_std]
    })
    scaler_stats.to_csv(os.path.join(BASE_DIR, f'scaler_stats_{symbol}_{interval}.csv'), index=False)
    logger.info(f"Scaler stats збережено у scaler_stats_{symbol}_{interval}.csv")

    X, y = create_dataset(scaled_data, target, look_back)
    if X.shape[0] == 0:
        logger.error("Недостатньо даних для тренування.")
        return None, None

    logger.info(f"Форма X: {X.shape}, форма y: {y.shape}")

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    histories = []
    fold = 1

    for train_index, val_index in kfold.split(X):
        logger.info(f"Тренування Fold {fold}")
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        model = build_model_with_attention((None, X_train.shape[2]))
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(0.0003, decay_steps=epochs * 100)

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)
        model.compile(optimizer=optimizer,
                      loss='mean_squared_error',
                      metrics=['mae', lambda x, y: mape_metric(x, y, target_mean, target_std)])

        history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset,
                            callbacks=[early_stopping], verbose=1)

        # Розрахунок метрик для валідації
        val_predictions = model.predict(val_dataset)
        real_diff_preds = (val_predictions.flatten() * target_std) + target_mean
        real_diff_y_val = (y_val * target_std) + target_mean
        last_prices = data['close'].iloc[val_index + look_back - 1].values
        real_preds = last_prices + real_diff_preds
        real_y_val = last_prices + real_diff_y_val

        real_mae = np.mean(np.abs(real_preds - real_y_val))
        real_mape = np.mean(np.abs((real_preds - real_y_val) / real_y_val)) * 100

        logger.info(f"Fold {fold} - Реальний MAE: {real_mae:.2f} USDT")
        logger.info(f"Fold {fold} - Реальний MAPE: {real_mape:.2f}%")

        for epoch in range(len(history.history['loss'])):
            history_data = {
                'symbol_id': symbol_id,
                'interval_id': interval_id,
                'fold': fold,
                'epoch': epoch + 1,
                'loss': history.history['loss'][epoch],
                'mae': history.history['mae'][epoch],
                'mape': history.history['lambda'][epoch],
                'val_loss': history.history['val_loss'][epoch],
                'val_mae': history.history['val_mae'][epoch],
                'val_mape': history.history['val_lambda'][epoch],
                'real_mae': real_mae,
                'real_mape': real_mape
            }
            insert_training_history(history_data)

        history_df = pd.DataFrame(history.history)
        history_df['real_mae'] = real_mae
        history_df['real_mape'] = real_mape
        histories.append(history_df)

        model_path = os.path.join(BASE_DIR, f'lstm_model_fold_{fold}_{symbol}_{interval}.keras')
        model.save(model_path)
        logger.info(f"Модель для Fold {fold} збережено у {model_path}")
        tf.keras.backend.clear_session()
        fold += 1

    combined_history = pd.concat(histories, keys=range(1, n_splits + 1))
    combined_history.to_csv(os.path.join(BASE_DIR, f'training_history_{symbol}_{interval}.csv'))
    logger.info(f"Історію тренування збережено у training_history_{symbol}_{interval}.csv")

    plot_training_history(histories, n_splits, symbol, interval)
    fine_tune_model(
        symbol=symbol,
        interval=interval,
        days_back=days_back,
        look_back=look_back,
        api_key=api_key,
        api_secret=api_secret,
        data=data,  # Передаємо data у fine_tune_model
        epochs=50,
        batch_size=32
    )

    return histories, None

def custom_mape_metric(y_true, y_pred, target_mean, target_std):
    y_true_denorm = y_true * target_std + target_mean
    y_pred_denorm = y_pred * target_std + target_mean
    mape = tf.reduce_mean(tf.abs((y_true_denorm - y_pred_denorm) / tf.abs(y_true_denorm + 1e-6))) * 100
    return mape

class EpochLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logger.info(f"Завершено епоху {epoch + 1}, loss: {logs.get('loss'):.4f}, val_loss: {logs.get('val_loss'):.4f}")

def fine_tune_model(symbol, interval, days_back, look_back, api_key, api_secret, data=None, epochs=50, batch_size=32, n_splits=5):
    logger.info(f"Повне донавчання моделі для {symbol} з days_back={days_back} і інтервалом {interval}.")
    scaled_data, target, features, target_scaler, data = prepare_data(symbol, interval, days_back, look_back, api_key, api_secret)
    if scaled_data is None:
        return None

    X, y = create_dataset(scaled_data, target, look_back)
    if X.shape[0] == 0:
        logger.error("Недостатньо даних для донавчання.")
        return None

    logger.info(f"Форма X для донавчання: {X.shape}, форма y: {y.shape}")
    target_mean = target_scaler.mean.numpy()[0]
    target_std = np.sqrt(target_scaler.variance.numpy()[0])

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    models = []

    for fold, (train_index, val_index) in enumerate(kfold.split(X), 1):
        base_model_path = os.path.join(BASE_DIR, f'lstm_model_fold_{fold}_{symbol}_{interval}.keras')
        finetuned_model_path = os.path.join(BASE_DIR, f'lstm_model_fold_{fold}_finetuned_{symbol}_{interval}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.keras')

        if not os.path.exists(base_model_path):
            logger.info(f"Базова модель {base_model_path} не знайдено. Створюю нову.")
            model = build_model_with_attention((None, X.shape[2]))
        else:
            logger.info(f"Завантажую базову модель із {base_model_path}")
            model = tf.keras.models.load_model(base_model_path, custom_objects={
                'custom_mape_metric': lambda x, y: mape_metric(x, y, target_mean, target_std)
            }, safe_mode=False)

        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_custom_mape_metric',
            patience=15,
            mode='min',
            restore_best_weights=True,
            verbose=1
        )
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_custom_mape_metric',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        callbacks = [early_stopping, reduce_lr, EpochLogger()]

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005, clipnorm=1.0)
        model.compile(optimizer=optimizer,
                      loss='mean_squared_error',
                      metrics=['mae', lambda x, y: mape_metric(x, y, target_mean, target_std)])

        logger.info(f"Донавчання Fold {fold}")
        history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=callbacks, verbose=1)

        # Розрахунок метрик для валідації
        val_predictions = model.predict(val_dataset)
        real_diff_preds = (val_predictions.flatten() * target_std) + target_mean
        real_diff_y_val = (y_val * target_std) + target_mean
        last_prices = data['close'].iloc[val_index + look_back - 1].values
        real_preds = last_prices + real_diff_preds
        real_y_val = last_prices + real_diff_y_val
        real_mape = np.mean(np.abs((real_preds - real_y_val) / real_y_val)) * 100

        logger.info(f"Fold {fold} - Реальний MAPE: {real_mape:.4f}%")
        if real_mape > 0.4:
            logger.warning(f"Fold {fold} - MAPE {real_mape:.4f}% перевищує цільовий поріг 0.4%")

        symbol_id = insert_symbol(symbol)
        interval_id = insert_interval(interval)
        for epoch in range(len(history.history['loss'])):
            history_data = {
                'symbol_id': symbol_id,
                'interval_id': interval_id,
                'fold': fold,
                'epoch': epoch + 1,
                'loss': history.history['loss'][epoch],
                'mae': history.history['mae'][epoch],
                'mape': history.history['lambda'][epoch],
                'val_loss': history.history['val_loss'][epoch],
                'val_mae': history.history['val_mae'][epoch],
                'val_mape': history.history['val_lambda'][epoch],
                'real_mae': np.mean(np.abs(real_preds - real_y_val)),
                'real_mape': real_mape
            }
            insert_training_history(history_data)

        logger.info(f"Навчання Fold {fold} завершено з {len(history.history['loss'])} епохами")
        model.save(finetuned_model_path)
        logger.info(f"Донавчана модель Fold {fold} збережена у {finetuned_model_path}")
        models.append(model)
        tf.keras.backend.clear_session()

    return models

def auto_fine_tune(symbol, interval, initial_days_back, final_days_back, final_look_back, api_key, api_secret, interval_hours=1, max_iterations=10):
    logger.info("Запуск автоматичного донавчання.")
    iteration = 1

    try:
        models = fine_tune_model(
            symbol=symbol,
            interval=interval,
            days_back=initial_days_back,
            look_back=360,
            api_key=api_key,
            api_secret=api_secret,
            epochs=100,
            batch_size=32
        )
        if not models:
            logger.error(f"Початкове донавчання для {symbol} не вдалося.")
            return
    except Exception as e:
        logger.error(f"Помилка в початковому донавчанні: {e}")
        return

    while iteration <= max_iterations:
        logger.info(f"Ітерація донавчання {iteration}/{max_iterations} для {symbol} на {datetime.now()}")
        try:
            scaled_data, target, features, target_scaler, data = prepare_data(symbol, interval, final_days_back, final_look_back, api_key, api_secret)
            if scaled_data is None:
                logger.error(f"Недостатньо даних для донавчання з days_back={final_days_back}.")
                final_days_back += 5
                continue

            models = fine_tune_model(
                symbol=symbol,
                interval=interval,
                days_back=final_days_back,
                look_back=final_look_back,
                api_key=api_key,
                api_secret=api_secret,
                data=data,  # Передаємо data
                epochs=50,
                batch_size=32
            )
            if not models:
                logger.error(f"Ітерація донавчання {iteration} для {symbol} не вдалася.")
            else:
                logger.info(f"Ітерація донавчання {iteration} успішно завершена.")
        except Exception as e:
            logger.error(f"Помилка в ітерації донавчання {iteration}: {e}")

        iteration += 1
        time.sleep(interval_hours * 3600)

    logger.info(f"Досягнуто максимальної кількості ітерацій ({max_iterations}). Завершення донавчання.")

def plot_training_history(histories, n_splits, symbol, interval):
    plt.figure(figsize=(12, 8))
    for i, history in enumerate(histories, 1):
        plt.subplot(2, 1, 1)
        plt.plot(history['loss'], label=f'Fold {i} Train Loss')
        plt.plot(history['val_loss'], label=f'Fold {i} Val Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(history['mae'], label=f'Fold {i} Train MAE')
        plt.plot(history['val_mae'], label=f'Fold {i} Val MAE')
        plt.title('Training and Validation MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        logger.debug(f"Додано графік для Fold {i}")
        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, f'training_history_{symbol}_{interval}.png'))
    logger.info(f"Графіки збережено у 'training_history_{symbol}_{interval}.png'")
    plt.close()