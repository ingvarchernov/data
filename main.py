import argparse
import logging
import sys
from datetime import datetime, timedelta
import os
from model_prediction import predict_main
from trading import execute_trade

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

def main():
    parser = argparse.ArgumentParser(description="Скрипт для тренування моделі та прогнозування цін криптовалют.")
    parser.add_argument("--symbol", type=str, required=True, help="Торгова пара, наприклад, BTCUSDT")
    parser.add_argument("--interval", type=str, required=True, help="Інтервал часу: 5m, 15m, 30m, 1h, 1w, 1M")
    parser.add_argument("--days_back", type=int, required=True, help="Кількість днів для завантаження історичних даних")
    parser.add_argument("--look_back", type=int, required=True, help="Кількість попередніх періодів для аналізу")
    parser.add_argument("--steps", type=int, required=True, help="Кількість кроків прогнозування вперед")
    parser.add_argument("--api_key", type=str, required=True, help="Binance API ключ")
    parser.add_argument("--api_secret", type=str, required=True, help="Binance API секрет")
    parser.add_argument("--strategy", type=str, default="trend", choices=["trend", "conservative", "aggressive"], help="Торгова стратегія")
    parser.add_argument("--auto_trade", action="store_true", help="Увімкнути автоторгівлю")
    parser.add_argument("--trade_quantity", type=float, default=0.001, help="Кількість для торгівлі (наприклад, 0.001 BTC)")
    parser.add_argument("--interactive", action="store_true", help="Увімкнути інтерактивний режим")

    args = parser.parse_args()

    try:
        # Запуск прогнозування
        trade_decisions = predict_main(
            symbol=args.symbol,
            interval=args.interval,
            days_back=args.days_back,
            look_back=args.look_back,
            steps=args.steps,
            api_key=args.api_key,
            api_secret=args.api_secret,
            strategy=args.strategy,
            auto_trade=args.auto_trade,
            trade_quantity=args.trade_quantity,
            interactive=args.interactive
        )

        # Інтерактивний режим
        if args.interactive and not args.auto_trade:
            for decision in trade_decisions:
                print(f"\nПропозиція торгівлі:")
                print(f"Символ: {decision['symbol']}")
                print(f"Час: {decision['timestamp']}")
                print(f"Дія: {decision['decision']}")
                print(f"Кількість: {decision['quantity']}")
                user_input = input("Виконати цю торгівлю? (y/n): ").strip().lower()

                if user_input == 'y':
                    execute_trade(
                        symbol=decision['symbol'],
                        decision=decision['decision'],
                        quantity=decision['quantity'],
                        api_key=args.api_key,
                        api_secret=args.api_secret
                    )
                else:
                    logger.info(f"Користувач відхилив торгівлю: {decision['symbol']} - {decision['decision']} at {decision['timestamp']}")

        logger.info("Програма успішно завершена.")
    except Exception as e:
        logger.error(f"Помилка під час виконання програми: {e}")
        raise

if __name__ == "__main__":
    main()