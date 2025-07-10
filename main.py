import argparse
import logging
import sys
import os
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
from model_training import train_lstm_model
from model_prediction import main as predict_prices, mape_metric
from config import SYMBOL, INTERVAL, DAYS_BACK, LOOK_BACK, STEPS

# Налаштування логування
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
    """Завантажує змінні середовища з .env файлу."""
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

def main():
    # Завантаження змінних середовища
    env_vars = load_environment_variables()
    api_key = env_vars['API_KEY']
    api_secret = env_vars['API_SECRET']

    # Парсинг аргументів командного рядка з дефолтними значеннями з config.py
    parser = argparse.ArgumentParser(description="Скрипт для тренування моделі та прогнозування цін криптовалют.")
    parser.add_argument("--symbol", type=str, default=SYMBOL, help=f"Торгова пара, наприклад, BTCUSDT (за замовчуванням: {SYMBOL})")
    parser.add_argument("--interval", type=str, default=INTERVAL, help=f"Інтервал часу: 5m, 15m, 30m, 1h, 1w, 1M (за замовчуванням: {INTERVAL})")
    parser.add_argument("--days_back", type=int, default=DAYS_BACK, help=f"Кількість днів для завантаження історичних даних (за замовчуванням: {DAYS_BACK})")
    parser.add_argument("--look_back", type=int, default=LOOK_BACK, help=f"Кількість попередніх періодів для аналізу (за замовчуванням: {LOOK_BACK})")
    parser.add_argument("--steps", type=int, default=STEPS, help=f"Кількість кроків прогнозування вперед (за замовчуванням: {STEPS})")

    args = parser.parse_args()

    try:
        # Виконання прогнозування
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
            logger.error("Прогнозування не вдалося: отримані порожні або невалідні дані")
            sys.exit(1)

        # Виведення результатів прогнозування
        logger.info("\nРезультати прогнозування:")
        for _, row in predictions_df.iterrows():
            print(f"Час: {row['timestamp']}")
            print(f"Прогнозована ціна: {row['predicted_price']:.2f} USDT")
            for fold in ['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5']:
                if fold in row and not pd.isna(row[fold]):
                    print(f"{fold}: {row[fold]:.2f} USDT")
            print("-" * 50)

        logger.info("Програма успішно завершена.")

    except Exception as e:
        logger.error(f"Помилка під час виконання програми: {e}")
        raise

if __name__ == "__main__":
    main()