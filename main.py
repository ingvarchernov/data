import argparse
import logging
import sys
import os
import time
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from model_training import train_lstm_model, auto_fine_tune
from model_prediction import main as predict_prices
from data_extraction import get_historical_data
from db_utils import check_and_append_historical_data, get_historical_data_from_db
from config import SYMBOL, INTERVAL, DAYS_BACK, LOOK_BACK, STEPS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_environment_variables():
    load_dotenv()
    required_vars = ['API_KEY', 'API_SECRET', 'DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER', 'DB_PASSWORD']
    env_vars = {}
    for var in required_vars:
        value = os.getenv(var)
        if value is None:
            logger.error(f"Помилка: Змінна середовища {var} не знайдена в .env файлі.")
            sys.exit(1)
        env_vars[var] = value
    return env_vars

def monitor_volatility(data, threshold=0.05):
    recent_data = data.tail(24)
    price_changes = recent_data['close'].pct_change().abs()
    max_change = price_changes.max()
    logger.info(f"Максимальна зміна ціни за останні 24 години: {max_change:.4f}")
    return max_change > threshold

def main():
    env_vars = load_environment_variables()
    api_key = env_vars['API_KEY']
    api_secret = env_vars['API_SECRET']
    parser = argparse.ArgumentParser(description="Скрипт для тренування та прогнозування цін криптовалют.")
    parser.add_argument("--symbol", type=str, default=SYMBOL, help=f"Торгова пара, наприклад, BTCUSDT (за замовчуванням: {SYMBOL})")
    parser.add_argument("--interval", type=str, default=INTERVAL, help=f"Інтервал часу: 5m, 15m, 30m, 1h, 1w, 1M (за замовчуванням: {INTERVAL})")
    parser.add_argument("--days_back", type=int, default=DAYS_BACK, help=f"Кількість днів для завантаження історичних даних (за замовчуванням: {DAYS_BACK})")
    parser.add_argument("--look_back", type=int, default=LOOK_BACK, help=f"Кількість попередніх періодів для аналізу (за замовчуванням: {LOOK_BACK})")
    parser.add_argument("--steps", type=int, default=STEPS, help=f"Кількість кроків прогнозування вперед (за замовчуванням: {STEPS})")
    args = parser.parse_args()

    while True:
        try:
            logger.info(f"Запуск циклу для {args.symbol} ({args.interval})")
            # Завантажуємо дані з Binance API
            new_data = get_historical_data(args.symbol, args.interval, args.days_back, api_key, api_secret)
            if new_data.empty:
                logger.error("Не вдалося завантажити дані з Binance. Спроба через годину.")
                time.sleep(3600)
                continue

            # Оновлюємо базу даних
            success = check_and_append_historical_data(args.symbol, args.interval, args.days_back, new_data)
            if not success:
                logger.error("Не вдалося оновити дані. Спроба через годину.")
                time.sleep(3600)
                continue

            # Отримуємо дані з бази
            data = get_historical_data_from_db(args.symbol, args.interval, args.days_back, api_key, api_secret, skip_append=True)
            if data.empty:
                logger.error("Не вдалося завантажити дані з бази. Спроба через годину.")
                time.sleep(3600)
                continue

            if monitor_volatility(data):
                logger.info(f"Виявлено високу волатильність. Запускаю повне перенавчання.")
                train_lstm_model(
                    symbol=args.symbol,
                    interval=args.interval,
                    days_back=args.days_back,
                    look_back=args.look_back,
                    api_key=api_key,
                    api_secret=api_secret
                )

            logger.info(f"Запуск прогнозування для {args.symbol} з інтервалом {args.interval}")
            predictions_df = predict_prices(
                symbol=args.symbol,
                interval=args.interval,
                days_back=args.days_back,
                look_back=args.look_back,
                steps=args.steps,
                api_key=api_key,
                api_secret=api_secret
            )
            if predictions_df is None or predictions_df.empty:
                logger.error("Прогнозування не вдалося.")
                time.sleep(3600)
                continue

            logger.info("\nРезультати прогнозування:")
            for _, row in predictions_df.iterrows():
                print(f"Час: {row['timestamp']}")
                print(f"Прогнозована ціна: {row['predicted_price']:.2f} USDT")
                for fold in ['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5']:
                    if fold in row and not pd.isna(row[fold]):
                        print(f"{fold}: {row[fold]:.2f} USDT")
                print("-" * 50)

            logger.info("Запускаю автоматичне донавчання.")
            auto_fine_tune(
                symbol=args.symbol,
                interval=args.interval,
                initial_days_back=args.days_back,
                final_days_back=args.days_back,
                final_look_back=args.look_back,
                api_key=api_key,
                api_secret=api_secret,
                interval_hours=1,
                max_iterations=1
            )

            logger.info("Очікування наступного циклу (1 година)...")
            time.sleep(3600)
        except Exception as e:
            logger.error(f"Помилка в циклі: {e}")
            time.sleep(3600)

if __name__ == "__main__":
    main()