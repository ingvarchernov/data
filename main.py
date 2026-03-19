#!/usr/bin/env python3
"""Spot Trading Bot - Simple pattern-based trading"""
import asyncio
import logging
from datetime import datetime
from pathlib import Path

from config import TRADING_CONFIG, BINANCE_CONFIG

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class TradingSession:
    def __init__(self):
        self.start_time = datetime.now()
        self.iterations = 0
        
    def duration(self):
        delta = datetime.now() - self.start_time
        hours = delta.seconds // 3600
        minutes = (delta.seconds % 3600) // 60
        return f"{hours}h {minutes}m"


async def main():
    logger.info("="*80)
    logger.info("SPOT TRADING BOT")
    logger.info("="*80)
    logger.info(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"API: {'TESTNET' if BINANCE_CONFIG['testnet'] else 'PRODUCTION'}")
    logger.info(f"Symbols: {', '.join(TRADING_CONFIG['symbols'])}")
    logger.info("="*80 + "\n")
    
    session = TradingSession()
    
    try:
        # Initialize
        logger.info("Initializing components...")
        bot = TradingBot()
        pattern_scanner = PatternScanner()
        volatility_filter = VolatilityFilter()
        
        # Get balance
        try:
            account = bot.client.get_account()
            balances = {b['asset']: float(b['free']) for b in account['balances'] if float(b['free']) > 0}
            usdt_balance = balances.get('USDT', 0.0)
            logger.info(f"USDT Balance: ${usdt_balance:.2f}")
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return
        
        await telegram_notifier.send_message(
            f"SPOT BOT STARTED\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Balance: ${usdt_balance:.2f} USDT"
        )
        
        logger.info("Initialization complete!\n")
        
        # Main loop
        interval = TRADING_CONFIG.get('scan_interval', 300)
        iteration = 0
        
        while True:
            iteration += 1
            session.iterations = iteration
            
            logger.info(f"\n{'='*80}")
            logger.info(f"ITERATION #{iteration}")
            logger.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"{'='*80}\n")
            
            try:
                # Scan for volatile symbols
                logger.info("Scanning for volatile symbols...")
                volatile_symbols = await asyncio.get_event_loop().run_in_executor(
                    None,
                    volatility_filter.get_top_volatile,
                    TRADING_CONFIG['symbols'],
                    10
                )
                
                if volatile_symbols:
                    logger.info(f"Found {len(volatile_symbols)} volatile symbols")
                    for sym, score in volatile_symbols[:5]:
                        logger.info(f"  {sym}: {score:.2f}")
                
                # Scan for patterns
                logger.info("\nScanning for patterns...")
                patterns_found = []
                
                for symbol, _ in volatile_symbols[:5]:
                    try:
                        patterns = await asyncio.get_event_loop().run_in_executor(
                            None,
                            pattern_scanner.scan_symbol,
                            symbol
                        )
                        
                        if patterns:
                            patterns_found.append((symbol, patterns))
                            logger.info(f"  {symbol}: {len(patterns)} patterns")
                    
                    except Exception as e:
                        logger.warning(f"  {symbol}: {e}")
                
                if patterns_found:
                    logger.info(f"\nTotal patterns: {sum(len(p) for _, p in patterns_found)}")
                
                # TODO: Add position management
                
                logger.info(f"\nWaiting {interval}s...")
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in iteration #{iteration}: {e}", exc_info=True)
                await asyncio.sleep(60)
    
    except KeyboardInterrupt:
        logger.info("\nBot stopped (Ctrl+C)")
        
        await telegram_notifier.send_message(
            f"SPOT BOT STOPPED\n"
            f"Duration: {session.duration()}\n"
            f"Iterations: {session.iterations}"
        )
    
    except Exception as e:
        logger.error(f"\nCRITICAL ERROR: {e}", exc_info=True)
        await telegram_notifier.send_message(f"CRITICAL ERROR: {str(e)[:200]}")
        raise
    
    finally:
        logger.info(f"\n{'='*80}")
        logger.info(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Duration: {session.duration()}")
        logger.info(f"Iterations: {session.iterations}")
        logger.info(f"{'='*80}\n")


if __name__ == "__main__":
    asyncio.run(main())
