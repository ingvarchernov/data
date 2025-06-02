# -*- coding: utf-8 -*-
import logging
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Завантаження конфігурації
load_dotenv()

# Перевірка наявності обов’язкових змінних
required_env_vars = ['DB_NAME', 'DB_USER', 'DB_PASSWORD', 'DB_HOST', 'DB_PORT']
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    logger.error(f"Відсутні обов’язкові змінні середовища: {', '.join(missing_vars)}")
    raise ValueError(f"Необхідно визначити змінні середовища: {', '.join(missing_vars)}")

# Налаштування SQLAlchemy engine з пулом з’єднань
db_string = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
engine = create_engine(db_string, pool_size=5, max_overflow=10)

def insert_symbol(symbol):
    """Вставка або отримання ID символу."""
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("INSERT INTO symbols (symbol) VALUES (:symbol) ON CONFLICT (symbol) DO NOTHING RETURNING symbol_id"),
                {"symbol": symbol}
            )
            symbol_id = result.scalar()
            if symbol_id is None:
                result = conn.execute(
                    text("SELECT symbol_id FROM symbols WHERE symbol = :symbol"),
                    {"symbol": symbol}
                )
                symbol_id = result.scalar()
            conn.commit()
        logger.debug(f"Символ {symbol} має symbol_id={symbol_id}")
        return symbol_id
    except Exception as e:
        logger.error(f"Помилка при вставці символу {symbol}: {e}")
        raise

def insert_interval(interval):
    """Вставка або отримання ID інтервалу."""
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("INSERT INTO intervals (interval) VALUES (:interval) ON CONFLICT (interval) DO NOTHING RETURNING interval_id"),
                {"interval": interval}
            )
            interval_id = result.scalar()
            if interval_id is None:
                result = conn.execute(
                    text("SELECT interval_id FROM intervals WHERE interval = :interval"),
                    {"interval": interval}
                )
                interval_id = result.scalar()
            conn.commit()
        logger.debug(f"Інтервал {interval} має interval_id={interval_id}")
        return interval_id
    except Exception as e:
        logger.error(f"Помилка при вставці інтервалу {interval}: {e}")
        raise

def insert_historical_data(data, symbol_id, interval_id):
    """Збереження історичних даних у PostgreSQL."""
    try:
        data = data.copy()
        data['symbol_id'] = symbol_id
        data['interval_id'] = interval_id

        columns = [
            'symbol_id', 'interval_id', 'timestamp', 'open', 'high', 'low', 'close',
            'volume', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av'
        ]

        with engine.connect() as conn:
            # Вставка даних
            data[columns].to_sql('historical_data', conn, if_exists='append', index=False, method='multi')
            # Отримання data_id для вставлених записів
            result = conn.execute(
                text("""
                    SELECT data_id
                    FROM historical_data
                    WHERE symbol_id = :symbol_id
                    AND interval_id = :interval_id
                    AND timestamp IN :timestamps
                    ORDER BY timestamp
                """),
                {'symbol_id': symbol_id, 'interval_id': interval_id, 'timestamps': tuple(data['timestamp'])}
            )
            data_ids = [row[0] for row in result.fetchall()]
            conn.commit()
        logger.info(f"Збережено {len(data_ids)} записів у PostgreSQL для symbol_id={symbol_id} ({interval_id}).")
        return data_ids
    except Exception as e:
        logger.error(f"Помилка при збереженні історичних даних: {e}")
        raise

def check_and_append_historical_data(symbol, interval, days_back, api_key, api_secret):
    from data_extraction import get_historical_data

    if not api_key or not api_secret:
        logger.error("API ключі (api_key або api_secret) не надані. Неможливо завантажити дані з Binance.")
        raise ValueError("API ключі (api_key або api_secret) не надані.")

    symbol_id = insert_symbol(symbol)
    interval_id = insert_interval(interval)
    current_time = datetime.now()

    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT MAX(timestamp)
                    FROM historical_data
                    WHERE symbol_id = :symbol_id AND interval_id = :interval_id
                """),
                {"symbol_id": symbol_id, "interval_id": interval_id}
            )
            last_timestamp = result.scalar()
    except Exception as e:
        logger.error(f"Помилка при перевірці наявності даних для {symbol} ({interval}): {e}")
        last_timestamp = None

    if last_timestamp is None:
        logger.info(f"Дані для {symbol} ({interval}) відсутні. Завантажую з Binance за {days_back} днів.")
        start_time = current_time - timedelta(days=days_back)
        data = get_historical_data(symbol, interval=interval, days_back=days_back, api_key=api_key, api_secret=api_secret)
        if data is None or data.empty:
            logger.error(f"Не вдалося завантажити дані для {symbol} ({interval}) з Binance. Перевірте API-ключі, підключення або доступність символу.")
            raise ValueError(f"Не вдалося завантажити дані для {symbol} ({interval}) з Binance.")
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        insert_historical_data(data, symbol_id, interval_id)
        logger.info(f"Додано нові дані з нуля: {len(data)} записів.")
        return True

    # Доповнення даних (якщо потрібно)
    logger.debug(f"Останній timestamp у базі: {last_timestamp}")
    time_diff = current_time - last_timestamp
    hours_diff = time_diff.total_seconds() / 3600
    logger.debug(f"Часова різниця: {hours_diff} годин")

    if hours_diff <= 1:
        logger.info(f"Дані для {symbol} ({interval}) актуальні (остання дата: {last_timestamp}).")
        return True

    logger.info(f"Доповнюю дані для {symbol} ({interval}) з {last_timestamp} до {current_time}.")
    days_to_fetch = hours_diff / 24 + 1
    new_data = get_historical_data(symbol, interval=interval, days_back=days_to_fetch, api_key=api_key, api_secret=api_secret)
    if new_data is None or new_data.empty:
        logger.error(f"Не вдалося завантажити нові дані для {symbol} ({interval}) з Binance. Перевірте API-ключі та підключення.")
        raise ValueError(f"Не вдалося завантажити нові дані для {symbol} ({interval}) з Binance.")

    new_data['timestamp'] = pd.to_datetime(new_data['timestamp'], unit='ms')
    new_data = new_data[new_data['timestamp'] > last_timestamp]

    if new_data.empty:
        logger.info(f"Немає нових даних для додавання для {symbol} ({interval}).")
        return True

    insert_historical_data(new_data, symbol_id, interval_id)
    logger.info(f"Додано нові дані: {len(new_data)} записів з {new_data['timestamp'].min()} до {new_data['timestamp'].max()}.")
    return True

def get_historical_data_from_db(symbol, interval, days_back, api_key, api_secret):
    """Отримання історичних даних із бази даних із перевіркою та доповненням."""
    # Перевіряємо та доповнюємо дані перед отриманням
    if not check_and_append_historical_data(symbol, interval, days_back, api_key, api_secret):
        logger.error(f"Не вдалося забезпечити наявність даних для {symbol} ({interval}).")
        return pd.DataFrame()

    try:
        start_time = datetime.now() - timedelta(days=days_back)
        symbol_id = insert_symbol(symbol)
        interval_id = insert_interval(interval)

        query = """
            SELECT
                h.data_id, h.timestamp, h.open, h.high, h.low, h.close, h.volume,
                h.quote_av, h.trades, h.tb_base_av, h.tb_quote_av
            FROM historical_data h
            WHERE h.symbol_id = :symbol_id AND h.interval_id = :interval_id AND h.timestamp >= :start_time
            ORDER BY h.timestamp
        """
        df = pd.read_sql_query(
            text(query),
            engine,
            params={"symbol_id": symbol_id, "interval_id": interval_id, "start_time": start_time}
        )

        logger.debug(f"Отримано {len(df)} записів із бази для {symbol} ({interval}).")
        return df
    except Exception as e:
        logger.error(f"Помилка при отриманні даних з бази для {symbol}: {e}")
        return pd.DataFrame()

def check_technical_indicators_exists(data_id):
    """Перевірка, чи існує запис у technical_indicators для даного data_id."""
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT 1 FROM technical_indicators WHERE data_id = :data_id"),
                {"data_id": data_id}
            )
            exists = result.scalar() is not None
        logger.debug(f"Запис для data_id={data_id} у technical_indicators існує: {exists}")
        return exists
    except Exception as e:
        logger.error(f"Помилка при перевірці існування запису для data_id={data_id}: {e}")
        raise

def insert_technical_indicators(data_id, indicators):
    """Збереження технічних індикаторів у PostgreSQL з оновленням при конфлікті."""
    try:
        with engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO technical_indicators (
                        data_id, rsi, macd, macd_signal, upper_band, lower_band,
                        stoch, stoch_signal, ema, atr, cci, obv, volatility,
                        volume_pct, close_lag1, close_lag2, close_diff, log_return,
                        hour_norm, day_norm, adx, vwap
                    ) VALUES (
                        :data_id, :rsi, :macd, :macd_signal, :upper_band, :lower_band,
                        :stoch, :stoch_signal, :ema, :atr, :cci, :obv, :volatility,
                        :volume_pct, :close_lag1, :close_lag2, :close_diff, :log_return,
                        :hour_norm, :day_norm, :adx, :vwap
                    )
                    ON CONFLICT (data_id) DO UPDATE SET
                        rsi = EXCLUDED.rsi,
                        macd = EXCLUDED.macd,
                        macd_signal = EXCLUDED.macd_signal,
                        upper_band = EXCLUDED.upper_band,
                        lower_band = EXCLUDED.lower_band,
                        stoch = EXCLUDED.stoch,
                        stoch_signal = EXCLUDED.stoch_signal,
                        ema = EXCLUDED.ema,
                        atr = EXCLUDED.atr,
                        cci = EXCLUDED.cci,
                        obv = EXCLUDED.obv,
                        volatility = EXCLUDED.volatility,
                        volume_pct = EXCLUDED.volume_pct,
                        close_lag1 = EXCLUDED.close_lag1,
                        close_lag2 = EXCLUDED.close_lag2,
                        close_diff = EXCLUDED.close_diff,
                        log_return = EXCLUDED.log_return,
                        hour_norm = EXCLUDED.hour_norm,
                        day_norm = EXCLUDED.day_norm,
                        adx = EXCLUDED.adx,
                        vwap = EXCLUDED.vwap
                """),
                indicators
            )
            conn.commit()
        logger.debug(f"Збережено або оновлено технічні індикатори для data_id={data_id}")
    except Exception as e:
        logger.error(f"Помилка при збереженні технічних індикаторів для data_id={data_id}: {e}")
        raise

def insert_normalized_data(data_id, normalized_values):
    """Збереження нормалізованих даних у PostgreSQL з оновленням при конфлікті."""
    try:
        data_id = int(data_id)
        records = [{"data_id": data_id, "feature": k, "normalized_value": float(v)} for k, v in normalized_values.items()]
        with engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO normalized_data (data_id, feature, normalized_value)
                    VALUES (:data_id, :feature, :normalized_value)
                    ON CONFLICT (data_id, feature) DO UPDATE SET
                        normalized_value = EXCLUDED.normalized_value
                """),
                records
            )
            conn.commit()
        logger.debug(f"Збережено або оновлено нормалізовані дані для data_id={data_id}")
    except Exception as e:
        logger.error(f"Помилка при збереженні нормалізованих даних для data_id={data_id}: {e}")
        raise

def insert_training_history(history_data):
    """Збереження історії тренування у PostgreSQL."""
    try:
        # Конвертація значень NumPy у прості типи float
        for key in ['loss', 'mae', 'mape', 'val_loss', 'val_mae', 'val_mape', 'real_mae', 'real_mape']:
            if isinstance(history_data[key], np.floating):
                history_data[key] = float(history_data[key])
            elif isinstance(history_data[key], np.ndarray):
                history_data[key] = float(history_data[key].item())

        with engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO training_history (
                        symbol_id, interval_id, fold, epoch, loss, mae, mape,
                        val_loss, val_mae, val_mape, real_mae, real_mape
                    ) VALUES (
                        :symbol_id, :interval_id, :fold, :epoch, :loss, :mae, :mape,
                        :val_loss, :val_mae, :val_mape, :real_mae, :real_mape
                    )
                    ON CONFLICT ON CONSTRAINT training_history_symbol_id_interval_id_fold_epoch_key
                    DO UPDATE SET
                        loss = EXCLUDED.loss,
                        mae = EXCLUDED.mae,
                        mape = EXCLUDED.mape,
                        val_loss = EXCLUDED.val_loss,
                        val_mae = EXCLUDED.val_mae,
                        val_mape = EXCLUDED.val_mape,
                        real_mae = EXCLUDED.real_mae,
                        real_mape = EXCLUDED.real_mape
                """),
                history_data
            )
            conn.commit()
        logger.debug(f"Збережено або оновлено історію тренування: fold={history_data['fold']}, epoch={history_data['epoch']}")
    except Exception as e:
        logger.error(f"Помилка при збереженні історії тренування: {e}")
        raise

def insert_prediction(prediction_data):
    """Збереження прогнозів у PostgreSQL."""
    try:
        # Конвертація значень NumPy у прості типи float
        for key in ['last_price', 'predicted_price', 'fold_1_prediction', 'fold_2_prediction',
                    'fold_3_prediction', 'fold_4_prediction', 'fold_5_prediction']:
            if key in prediction_data and prediction_data[key] is not None:
                if isinstance(prediction_data[key], np.floating):
                    prediction_data[key] = float(prediction_data[key])
                elif isinstance(prediction_data[key], np.ndarray):
                    prediction_data[key] = float(prediction_data[key].item())

        with engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO predictions (
                        symbol_id, interval_id, timestamp, last_price, predicted_price,
                        fold_1_prediction, fold_2_prediction, fold_3_prediction,
                        fold_4_prediction, fold_5_prediction
                    ) VALUES (
                        :symbol_id, :interval_id, :timestamp, :last_price, :predicted_price,
                        :fold_1_prediction, :fold_2_prediction, :fold_3_prediction,
                        :fold_4_prediction, :fold_5_prediction
                    )
                    ON CONFLICT ON CONSTRAINT predictions_symbol_id_interval_id_timestamp_key
                    DO UPDATE SET
                        last_price = EXCLUDED.last_price,
                        predicted_price = EXCLUDED.predicted_price,
                        fold_1_prediction = EXCLUDED.fold_1_prediction,
                        fold_2_prediction = EXCLUDED.fold_2_prediction,
                        fold_3_prediction = EXCLUDED.fold_3_prediction,
                        fold_4_prediction = EXCLUDED.fold_4_prediction,
                        fold_5_prediction = EXCLUDED.fold_5_prediction
                """),
                prediction_data
            )
            conn.commit()
        logger.debug(f"Збережено або оновлено прогноз для timestamp={prediction_data['timestamp']}")
    except Exception as e:
        logger.error(f"Помилка при збереженні прогнозу для timestamp={prediction_data['timestamp']}: {e}")
        raise

def insert_scaler_stats(symbol_id, interval_id, target_mean, target_std):
    """Збереження статистики масштабування."""
    try:
        with engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO scaler_stats (symbol_id, interval_id, target_mean, target_std)
                    VALUES (:symbol_id, :interval_id, :target_mean, :target_std)
                    ON CONFLICT (symbol_id, interval_id) DO UPDATE
                    SET target_mean = EXCLUDED.target_mean, target_std = EXCLUDED.target_std
                """),
                {
                    "symbol_id": symbol_id,
                    "interval_id": interval_id,
                    "target_mean": float(target_mean),
                    "target_std": float(target_std)
                }
            )
            conn.commit()
        logger.debug(f"Збережено статистику масштабування для symbol_id={symbol_id}, interval_id={interval_id}")
    except Exception as e:
        logger.error(f"Помилка при збереженні статистики масштабування для symbol_id={symbol_id}: {e}")
        raise

def get_scaler_stats(symbol_id, interval_id):
    """Отримання статистики масштабування."""
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT target_mean, target_std
                    FROM scaler_stats
                    WHERE symbol_id = :symbol_id AND interval_id = :interval_id
                """),
                {"symbol_id": symbol_id, "interval_id": interval_id}
            )
            row = result.fetchone()
        if row:
            logger.debug(f"Отримано статистику масштабування: mean={row[0]}, std={row[1]}")
            return row[0], row[1]
        logger.warning(f"Статистика масштабування для symbol_id={symbol_id}, interval_id={interval_id} не знайдена.")
        return None, None
    except Exception as e:
        logger.error(f"Помилка при отриманні статистики масштабування для symbol_id={symbol_id}: {e}")
        return None, None