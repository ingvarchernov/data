#!/usr/bin/env python3
"""Utility for synchronising Binance market data into the local database."""
import argparse
import asyncio
import logging
import os
from typing import List, Tuple

from dotenv import load_dotenv

from intelligent_sys import UnifiedBinanceLoader
from optimized_db import OptimizedDatabaseManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Синхронізація історичних даних Binance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=[
            'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT',
            'DOTUSDT', 'BNBUSDT', 'XRPUSDT', 'LTCUSDT',
            'LINKUSDT', 'AVAXUSDT'
        ],
        help="Список символів для синхронізації"
    )
    parser.add_argument(
        "--interval",
        default="1h",
        help="Інтервал OHLCV (наприклад, 1m, 15m, 1h, 4h, 1d)"
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=180,
        help="Кількість днів історії для завантаження"
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=3,
        help="Максимальна кількість одночасних запитів до API"
    )
    parser.add_argument(
        "--use-public-data",
        action="store_true",
        help="Використовувати лише публічні endpoint (без підпису)"
    )
    parser.add_argument(
        "--testnet",
        action="store_true",
        default=None,
        help="Перемкнути на Binance testnet (якщо не передано, береться з USE_TESTNET)"
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Рівень логування"
    )
    return parser.parse_args()


async def sync_symbol(
    loader: UnifiedBinanceLoader,
    db_manager: OptimizedDatabaseManager,
    symbol: str,
    interval: str,
    days_back: int,
    semaphore: asyncio.Semaphore
) -> Tuple[str, int, Exception]:
    async with semaphore:
        try:
            saved = await loader.save_to_database(db_manager, symbol, interval, days_back)
            return symbol, saved, None
        except Exception as exc:  # pragma: no cover - operational logging
            return symbol, 0, exc


async def run_sync(args: argparse.Namespace) -> None:
    load_dotenv()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("data_sync")

    use_testnet_env = os.getenv("USE_TESTNET")
    if args.testnet is None and use_testnet_env is not None:
        args.testnet = use_testnet_env.lower() not in {"0", "false", "no"}
    elif args.testnet is None:
        # За замовчуванням активуємо testnet для безпечних запусків
        args.testnet = True

    logger.info(
        "🚀 Синхронізація: symbols=%s interval=%s days_back=%d testnet=%s public=%s",
        ",".join(args.symbols),
        args.interval,
        args.days_back,
        args.testnet,
        args.use_public_data
    )

    db_manager = OptimizedDatabaseManager()
    await db_manager.initialize()

    loader = UnifiedBinanceLoader(
        use_public_data=args.use_public_data,
        testnet=args.testnet
    )

    semaphore = asyncio.Semaphore(max(1, args.max_parallel))
    tasks: List[asyncio.Task] = []
    for symbol in args.symbols:
        tasks.append(asyncio.create_task(sync_symbol(
            loader, db_manager, symbol, args.interval, args.days_back, semaphore
        )))

    try:
        results = await asyncio.gather(*tasks)
        total_rows = 0
        errors: List[Tuple[str, Exception]] = []
        for symbol, saved, error in results:
            if error:
                logger.error("❌ %s: %s", symbol, error)
                errors.append((symbol, error))
            else:
                total_rows += saved
                logger.info("✅ %s: %d рядків", symbol, saved)

        logger.info("🎉 Синхронізацію завершено. Всього оновлено %d рядків", total_rows)
        if errors:
            logger.warning("⚠️ Помилки під час синхронізації: %d", len(errors))
    finally:
        await loader.close()
        await db_manager.close()


def main() -> None:
    args = parse_args()
    asyncio.run(run_sync(args))


if __name__ == "__main__":
    main()