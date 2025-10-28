#!/usr/bin/env python3
"""
🎯 MASTER CONTROL - Єдина точка входу для всієї торгової системи
Об'єднує: тренування, тестування, моніторинг, live trading
"""
import asyncio
import argparse
import logging
import sys
from pathlib import Path
from typing import List
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Додаємо до PATH
sys.path.insert(0, str(Path(__file__).parent))

from training.batch_train_rf import train_all_symbols
from training.simple_trend_classifier import SimpleTrendClassifier

# Default configuration
DEFAULT_SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 
    'ADAUSDT', 'XRPUSDT', 'DOGEUSDT', 'DOTUSDT',
    'MATICUSDT', 'AVAXUSDT'
]

class MasterControl:
    """Центральний контролер торгової системи"""
    
    def __init__(self):
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
    
    async def train_models(self, symbols: List[str] = None, days: int = 730, force: bool = False):
        """Тренування ML моделей"""
        logger.info("\n" + "="*80)
        logger.info("🤖 ТРЕНУВАННЯ ML МОДЕЛЕЙ")
        logger.info("="*80)
        
        symbols = symbols or DEFAULT_SYMBOLS
        logger.info(f"📊 Символів: {len(symbols)}")
        logger.info(f"📅 Історія: {days} днів (~{days//365} років)")
        logger.info(f"🔄 Force retrain: {force}\n")
        
        results = []
        for i, symbol in enumerate(symbols, 1):
            logger.info(f"\n[{i}/{len(symbols)}] 🎯 {symbol}")
            logger.info("-" * 80)
            
            try:
                model_dir = self.models_dir / f"simple_trend_{symbol}"
                model_file = model_dir / f"model_{symbol}_4h.pkl"
                
                if model_file.exists() and not force:
                    logger.info(f"⏭️  Модель існує, пропускаю (--force для перетренування)")
                    results.append({'symbol': symbol, 'status': 'skipped'})
                    continue
                
                classifier = SimpleTrendClassifier(symbol=symbol, interval='4h')
                await classifier.prepare_data(days=days)
                metrics = await classifier.train()
                classifier.save_model()
                
                acc = metrics.get('test_accuracy', 0)
                logger.info(f"✅ {symbol}: {acc:.2%} accuracy")
                
                results.append({
                    'symbol': symbol,
                    'status': 'success',
                    'accuracy': acc
                })
                
            except Exception as e:
                logger.error(f"❌ {symbol}: {e}")
                results.append({
                    'symbol': symbol,
                    'status': 'error',
                    'error': str(e)
                })
        
        # Summary
        self._print_training_summary(results)
        return results
    
    def _print_training_summary(self, results: List[dict]):
        """Підсумок тренування"""
        logger.info("\n" + "="*80)
        logger.info("📊 ПІДСУМОК ТРЕНУВАННЯ")
        logger.info("="*80 + "\n")
        
        success = [r for r in results if r['status'] == 'success']
        skipped = [r for r in results if r['status'] == 'skipped']
        errors = [r for r in results if r['status'] == 'error']
        
        if success:
            logger.info(f"✅ УСПІШНО НАТРЕНОВАНО: {len(success)}")
            sorted_success = sorted(success, key=lambda x: x.get('accuracy', 0), reverse=True)
            for r in sorted_success:
                logger.info(f"   {r['symbol']:12} - {r['accuracy']:.2%}")
        
        if skipped:
            logger.info(f"\n⏭️  ПРОПУЩЕНО: {len(skipped)}")
            for r in skipped:
                logger.info(f"   {r['symbol']}")
        
        if errors:
            logger.info(f"\n❌ ПОМИЛКИ: {len(errors)}")
            for r in errors:
                logger.info(f"   {r['symbol']:12} - {r.get('error', 'Unknown')}")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Всього: {len(results)} | Успіх: {len(success)} | Пропущено: {len(skipped)} | Помилки: {len(errors)}")
        logger.info(f"{'='*80}\n")
    
    def check_models(self, symbols: List[str] = None):
        """Перевірка наявності моделей"""
        logger.info("\n" + "="*80)
        logger.info("🔍 ПЕРЕВІРКА МОДЕЛЕЙ")
        logger.info("="*80 + "\n")
        
        symbols = symbols or DEFAULT_SYMBOLS
        existing = []
        missing = []
        
        for symbol in symbols:
            model_dir = self.models_dir / f"simple_trend_{symbol}"
            model_file = model_dir / f"model_{symbol}_4h.pkl"
            
            if model_file.exists():
                size_mb = model_file.stat().st_size / (1024 * 1024)
                logger.info(f"✅ {symbol:12} - {size_mb:.2f} MB")
                existing.append(symbol)
            else:
                logger.info(f"❌ {symbol:12} - відсутня")
                missing.append(symbol)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Існують: {len(existing)}/{len(symbols)} моделей")
        if missing:
            logger.info(f"Відсутні: {', '.join(missing)}")
        logger.info(f"{'='*80}\n")
        
        return {'existing': existing, 'missing': missing}
    
    async def monitor_positions(self):
        """Моніторинг поточних позицій"""
        logger.info("\n" + "="*80)
        logger.info("📊 МОНІТОРИНГ ПОЗИЦІЙ")
        logger.info("="*80 + "\n")
        
        try:
            from dotenv import load_dotenv
            from binance.client import Client
            
            load_dotenv()
            client = Client(
                os.getenv('FUTURES_API_KEY'),
                os.getenv('FUTURES_API_SECRET'),
                testnet=True
            )
            
            positions = client.futures_position_information()
            
            total_pnl = 0
            count = 0
            
            for pos in positions:
                amt = float(pos['positionAmt'])
                if abs(amt) > 0.0001:
                    symbol = pos['symbol']
                    entry = float(pos['entryPrice'])
                    mark = float(pos['markPrice'])
                    pnl = float(pos['unRealizedProfit'])
                    side = '📈 LONG' if amt > 0 else '📉 SHORT'
                    pnl_emoji = '💰' if pnl > 0 else '📉'
                    
                    logger.info(f"{side} {symbol}")
                    logger.info(f"   Entry: ${entry:,.4f} | Mark: ${mark:,.4f}")
                    logger.info(f"   Size: {abs(amt):.4f} | {pnl_emoji} PnL: ${pnl:+.2f}\n")
                    
                    total_pnl += pnl
                    count += 1
            
            logger.info("="*80)
            if count > 0:
                emoji = '💰' if total_pnl > 0 else '📉'
                logger.info(f"Позицій: {count} | {emoji} Загальний PnL: ${total_pnl:+.2f}")
            else:
                logger.info("✅ Немає відкритих позицій")
            logger.info("="*80 + "\n")
            
            return {'count': count, 'total_pnl': total_pnl}
            
        except Exception as e:
            logger.error(f"❌ Помилка моніторингу: {e}")
            return None
    
    async def run_bot(self, symbols: List[str] = None, testnet: bool = True, enable_trading: bool = False):
        """Запуск торгового бота"""
        logger.info("\n" + "="*80)
        logger.info("🤖 ЗАПУСК ТОРГОВОГО БОТА")
        logger.info("="*80)
        
        symbols = symbols or DEFAULT_SYMBOLS[:6]  # Обмежуємо до 6
        
        logger.info(f"📊 Символи: {', '.join(symbols)}")
        logger.info(f"🧪 Testnet: {testnet}")
        logger.info(f"⚡ Live Trading: {enable_trading}")
        logger.info("="*80 + "\n")
        
        try:
            from simple_trading_bot import SimpleTradingBot
            
            bot = SimpleTradingBot(
                symbols=symbols,
                testnet=testnet,
                enable_trading=enable_trading
            )
            
            await bot.run()
            
        except KeyboardInterrupt:
            logger.info("\n⚠️  Бот зупинено користувачем")
        except Exception as e:
            logger.error(f"❌ Помилка бота: {e}")
            raise

async def main():
    parser = argparse.ArgumentParser(
        description='🎯 MASTER CONTROL - Торгова система',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Приклади використання:

  # Перевірка моделей
  python master_control.py check
  
  # Тренування моделей (за замовчуванням 10 валют)
  python master_control.py train
  
  # Тренування конкретних валют
  python master_control.py train --symbols BTCUSDT ETHUSDT
  
  # Перетренування існуючих моделей
  python master_control.py train --force
  
  # Моніторинг позицій
  python master_control.py monitor
  
  # Запуск бота (demo mode)
  python master_control.py bot --symbols BTCUSDT ETHUSDT
  
  # Запуск бота з live trading
  python master_control.py bot --symbols BTCUSDT ETHUSDT --live
  
  # Повний цикл: train → check → monitor → bot
  python master_control.py all
        """
    )
    
    parser.add_argument(
        'command',
        choices=['train', 'check', 'monitor', 'bot', 'all'],
        help='Команда для виконання'
    )
    
    parser.add_argument(
        '--symbols',
        nargs='+',
        default=None,
        help='Список символів (за замовчуванням: топ-10)'
    )
    
    parser.add_argument(
        '--days',
        type=int,
        default=730,
        help='Днів історії для тренування (default: 730 = 2 роки)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Перетренувати навіть якщо модель існує'
    )
    
    parser.add_argument(
        '--testnet',
        action='store_true',
        default=True,
        help='Використовувати Binance Testnet (default: True)'
    )
    
    parser.add_argument(
        '--live',
        action='store_true',
        help='Увімкнути live trading (ОБЕРЕЖНО!)'
    )
    
    args = parser.parse_args()
    
    control = MasterControl()
    
    try:
        if args.command == 'train':
            await control.train_models(args.symbols, args.days, args.force)
        
        elif args.command == 'check':
            control.check_models(args.symbols)
        
        elif args.command == 'monitor':
            await control.monitor_positions()
        
        elif args.command == 'bot':
            await control.run_bot(args.symbols, args.testnet, args.live)
        
        elif args.command == 'all':
            logger.info("🎯 ПОВНИЙ ЦИКЛ: TRAIN → CHECK → MONITOR → BOT\n")
            
            # 1. Train
            await control.train_models(args.symbols, args.days, args.force)
            
            # 2. Check
            result = control.check_models(args.symbols)
            
            # 3. Monitor
            await control.monitor_positions()
            
            # 4. Bot (якщо є моделі)
            if result['existing']:
                logger.info(f"\n✅ Готово {len(result['existing'])} моделей, запускаю бота...\n")
                await control.run_bot(result['existing'][:6], args.testnet, args.live)
            else:
                logger.warning("⚠️  Немає натренованих моделей для запуску бота")
    
    except KeyboardInterrupt:
        logger.info("\n⚠️  Програму зупинено користувачем")
    except Exception as e:
        logger.error(f"❌ Критична помилка: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
