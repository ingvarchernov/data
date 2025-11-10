#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔧 ЦЕНТРАЛІЗОВАНА КОНФІГУРАЦІЯ СИСТЕМИ
Всі налаштування в одному місці
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# BINANCE API
# ============================================================================
# MAINNET API - для історичних даних та backtest
MAINNET_CONFIG = {
    'api_key': os.getenv('API_KEY'),
    'api_secret': os.getenv('API_SECRET'),
    'testnet': False,  # MAINNET для реальних історичних даних
}

# TESTNET API - для торгівлі та тестування
TESTNET_CONFIG = {
    'api_key': os.getenv('FUTURES_API_KEY'),
    'api_secret': os.getenv('FUTURES_API_SECRET'),
    'testnet': True,  # TESTNET для безпечної торгівлі
}

# За замовчуванням для торгівлі використовуємо TESTNET
BINANCE_CONFIG = TESTNET_CONFIG

# ============================================================================
# АВТОМАТИЧНЕ ВИЯВЛЕННЯ МОДЕЛЕЙ (ВИПРАВЛЕНО після 0% WR!)
# ============================================================================
# Backtest results (30 днів, жовтень):
# 🥇 ETHUSDT + SwingML: +$49.41 (Sharpe 0.16, WR 44.7%)
# 🥈 SOLUSDT + SwingML: +$19.18 (WR 40%)
# 🥉 XRPUSDT + SwingML: +$17.91 (WR 40.2%)
# 💰 BTCUSDT + Breakout: +$6.08 (Sharpe 0.11, WR 40.8%)
# 💰 BNBUSDT + Breakout: +$3.18 (стабільний)
#
# ВИПРАВЛЕННЯ після аналізу (04.11.2025):
# • ПРОБЛЕМА: Win Rate 33%, R/R = 0.79 (середній виграш $15, збиток $19)
# • ПРИЧИНА: SL 1.2% / TP 1.0% = негативний профіль ризику
# • РІШЕННЯ:
#   - SL: 1.2% → 1.0% (щільніший стоп, less slippage)
#   - TP: 1.0% → 2.5% (більший таргет, R/R = 2.5)
#   - Мінімальний профіт: $15 (не відкривати дрібні угоди)
#   - Максимальний збиток: $25 (обмеження розміру позиції)
# • ОЧІКУВАННЯ: R/R 2.5 + WR 35-40% = прибутковість (EV > 0)

# ⚠️ КРИТИЧНЕ ВИПРАВЛЕННЯ (04.11.2025 - 13:30):
# Після аналізу реальної торгівлі: WR 16.7%, PnL -$97.85
# ПРОБЛЕМА: Занадто багато збиткових символів (5 з 6)
# РІШЕННЯ: Залишити тільки BTCUSDT (єдиний прибутковий +$32)
#          + ETHUSDT/SOLUSDT (кращі за backtest)

# 🚀 MEGA UPDATE (04.11.2025 - 20:00):
# Протестовано 60 пар на 350 днів з leverage backtest!
# РЕЗУЛЬТАТ: 5x leverage = $500 → $25,404 за 11.6 місяців (+4,981% ROI)
# ТОП-10 ПАР з найкращими результатами (350 днів backtest):

FUTURES_SYMBOLS = [
    '1INCHUSDT',   # 🥇 46.6% WR, $4.39/day, 307% ROI
    'BNBUSDT',     # 🥈 45.3% WR, $3.75/day, 263% ROI
    'IMXUSDT',     # 🥉 39.7% WR, $3.57/day, 250% ROI
    'RENDERUSDT',  # 38.0% WR, $3.00/day, 210% ROI
    'ARBUSDT',     # 35.8% WR, $2.71/day, 190% ROI
    'XTZUSDT',     # 38.1% WR, $2.68/day, 188% ROI
    'TRXUSDT',     # 36.5% WR, $2.29/day, 160% ROI
    'SOLUSDT',     # 35.1% WR, $2.16/day, 151% ROI
    'SNXUSDT',     # 34.8% WR, $2.04/day, 143% ROI
    'ICPUSDT',     # 33.8% WR, $1.89/day, 133% ROI
]

# Старі SPOT symbols (deprecated - використовувати FUTURES!)
TRADING_SYMBOLS = [
    'BTCUSDT',   # ✅ +$32.17 в реальній торгівлі (єдиний прибутковий!)
    'ETHUSDT',   # 🥇 НАЙКРАЩИЙ за backtest: +$49.41/міс SwingML (Sharpe 0.16)
    'SOLUSDT',   # 🥈 Прибутковий за backtest: +$19.18/міс SwingML
]

# ВИДАЛЕНО (збиткові в реальній торгівлі):
# 'XRPUSDT',   # � -$41.41
# 'BNBUSDT',   # � -$13.50
# 'ADAUSDT',   # � -$12.75

# ============================================================================
# TRADING PARAMETERS (ОПТИМІЗОВАНО ПІСЛЯ BACKTEST)
# ============================================================================

# 🚀 FUTURES TRADING CONFIG (NEW - 5x Leverage Strategy)
# Based on 350-day backtest: $500 → $25,404 (+4,981% ROI)
FUTURES_CONFIG = {
    # Режим торгівлі
    'enabled': True,                    # Використовувати Futures
    'use_testnet': True,                # TRUE = Testnet, FALSE = Mainnet
    
    # Символи та leverage
    'symbols': FUTURES_SYMBOLS,         # TOP-10 пар з backtest
    'leverage': 5,                      # 5x - оптимальний баланс (3x=good, 5x=best, 10x+=margin call)
    'margin_mode': 'ISOLATED',          # ISOLATED margin для кожної пари окремо
    
    # Position sizing
    'initial_capital': 500.0,           # Початковий капітал $500
    'position_size_usd': 250.0,         # $250 per trade (з 5x leverage = $50 margin)
    'max_positions': 10,                # Максимум 10 одночасних позицій
    'position_size_percent': None,      # None = фіксований розмір $250
    
    # Risk Management (з backtest)
    'stop_loss_pct': 0.015,             # SL 1.5% (з backtest)
    'take_profit_pct': 0.040,           # TP 4.0% (з backtest, R/R = 2.67)
    'max_daily_loss': 100.0,            # Max $100 втрат за день
    'max_daily_loss_pct': 0.20,         # Max 20% капіталу за день
    
    # Trading filters (з backtest)
    'min_adx': 25,                      # ADX > 25 (strong trend only)
    'min_volume_multiplier': 0.8,       # Volume > 0.8x average (ПОСЛАБЛЕНО: було 1.2x)
    'require_ema_alignment': True,      # Price aligned with EMA50/200
    'min_confidence': 0.60,             # 60% ML впевненості
    
    # EMA filters
    'ema_short': 9,
    'ema_medium': 21,
    'ema_long_50': 50,
    'ema_long_200': 200,
    
    # Timing
    'check_interval': 300,              # 5 хвилин між перевірками
    'cooldown_after_loss': 3600,        # 1 година після loss
    'cooldown_after_win': 1800,         # 30 хвилин після win
    
    # Funding rates (Binance)
    'funding_rate_limit': 0.0005,       # Skip якщо funding > 0.05%
    'check_funding': True,              # Перевіряти funding перед входом
}

# SPOT TRADING CONFIG (Old - для порівняння)
TRADING_CONFIG = {
    # Основні параметри - оптимізований портфель (3 SYMBOLS - ВИПРАВЛЕНО!)
    'symbols': TRADING_SYMBOLS,
    
    # Ризик-менеджмент (КРИТИЧНО ВИПРАВЛЕНО після WR 16.7%!)
    'leverage': 25,                     # Плече 25x
    'position_size_usd': 400,           # $400 на позицію (ЗМЕНШЕНО з $800 через overleverage!)
    'stop_loss_pct': 0.010,             # SL 1.0% = $10 втрати на $400 (було $20 на $800)
    'take_profit_pct': 0.025,           # TP 2.5% = $25 прибутку на $400 (R/R = 2.5)
    'min_profit_dollars': 15.0,         # Мінімальний очікуваний профіт в $
    'max_loss_dollars': 15.0,           # Максимальний збиток на угоду (ЗМЕНШЕНО з $25)
    'max_positions': 3,                 # 3 позиції (ЗМЕНШЕНО з 8 - тільки 3 символи!)
    'reserve_slot_for_reversal': False, # Без резерву
    
    # 🔄 РОЗВОРОТ ПОЗИЦІЙ: агресивніші налаштування
    'reverse_on_strong_signal': True,
    'reverse_min_confidence': 0.70,        # 70% впевненості
    'reverse_profit_threshold': -0.005,    # якщо PnL < -0.5%
    
    # Впевненість та фільтри (КРИТИЧНО ПІДВИЩЕНО!)
    'min_confidence': 0.55,             # 55% (ПІДВИЩЕНО з 40% через низький WR!)
    'min_consensus': 0.90,              # 90% консенсус (ПІДВИЩЕНО з 80% - сильніша узгодженість!)
    
    # Захист (КРИТИЧНО ПОСИЛЕНО!)
    'max_daily_losses_per_symbol': 1,  # Максимум 1 програш на день (було 2!)
    'cooldown_after_sl': 7200,         # 2 год після SL (було 40 хв - КРИТИЧНО!)
    'cooldown_after_tp': 1800,         # 30 хвилин після TP (було 20 хв)
    'cooldown_after_force_close': 7200, # 2 години після force close
    'max_loss_per_symbol_pct': 0.03,   # Максимум -3% за день
    
    # Таймінги (ШВИДКА РЕАКЦІЯ)
    'check_interval': 240,             # 4 хвилини між перевірками (знижено з 5 хв)
    
    # Адаптація до волатильності (нічний режим - ПОСЛАБЛЕНО)
    'night_mode': {
        'enabled': True,
        'start_hour': 2,                   # 02:00 (пізніше - більше активність вночі)
        'end_hour': 7,                     # 07:00 (раніше - раніше починаємо)
        'interval_multiplier': 1.3,        # Збільшити інтервал на 30% (було 50%)
        'min_confidence_boost': 0.03,      # Додати 3% до впевненості (було 5%)
    },
    
    # Волатильність (динамічний вибір монет)
    'volatility_scan': {
        'enabled': True,                   # Використовувати сканер волатильності
        'scan_interval': 3600,             # Сканувати кожну годину
        'min_score': 35.0,                 # Мінімальний скор волатильності (30-50)
        'top_count': 8,                    # Торгувати топ 8 монет
        'require_volume': 5_000_000,       # Мінімум $5M об'єму за 24h
    },
    'min_volatility_score': 25.0,         # Мінімальний скор для торгівлі (фільтр мертвих пар)
    
    # Trailing stop для захисту прибутку (АГРЕСИВНИЙ - швидше захоплювати прибуток)
    'trailing_stop': {
        'enabled': True,                   # Використовувати trailing stop
        'activation_profit': 0.010,        # Активувати при +1.0% прибутку (25% на депозит)
        'trail_distance': 0.40,            # Закривати при відкаті на 40% від піку (тісніше)
    }
}

# ============================================================================
# TRADING STRATEGIES (НОВА СИСТЕМА)
# ============================================================================
STRATEGY_CONFIG = {
    'enabled': True,  # Використовувати нові стратегії
    'fallback_to_ml': True,  # Якщо стратегії не дають сигналу - використати ML
    
    # Налаштування для Trend Following
    'trend_following': {
        'min_confidence': 65,
        'sma_period': 50,
        'ema_period': 20,
        'rsi_period': 14,
        'rsi_overbought': 70,
        'rsi_oversold': 30,
        'atr_multiplier_sl': 1.5,
        'atr_multiplier_tp': 2.5,
        'trend_strength_threshold': 2.0,
    },
    
    # Налаштування для Mean Reversion
    'mean_reversion': {
        'min_confidence': 60,
        'bb_period': 20,
        'bb_std': 2,
        'rsi_period': 14,
        'rsi_overbought': 70,
        'rsi_oversold': 30,
        'volume_multiplier': 1.5,
        'sl_percent': 1.5,
        'tp_percent': 2.5,
    },
}

# ============================================================================
# MULTI-TIMEFRAME CONFIGURATION
# ============================================================================
# ОНОВЛЕНО: Використовуємо тільки 15m + 1h (БЕЗ 4h - мало даних)
MTF_CONFIG = {
    '15m': {
        'weight': 0.40,      # 40% - точки входу/виходу (найкращі моделі!)
        'interval': '15m',
        'periods': 96        # 24 години історії
    },
    '1h': {
        'weight': 0.60,      # 60% - основний тренд (головний таймфрейм)
        'interval': '1h',
        'periods': 24        # 24 години історії
    },
}

# ============================================================================
# PATTERN DETECTION (RUST)
# ============================================================================
PATTERN_CONFIG = {
    'enabled': True,
    'window': 10,               # Розмір вікна для пошуку патернів
    'min_confidence': 70.0,     # Мінімальна впевненість патерна
    'weight': 0.15,             # Вага в загальному рішенні (15%)
    'patterns': {
        'double_top_bottom': True,
        'head_shoulders': True,
        'triangle': True,
        'flag': True,
        'wedge': True,
        'three_soldiers_crows': True,
    }
}

# ============================================================================
# SIGNAL WEIGHTS (ML + MTF + PATTERNS)
# ============================================================================
SIGNAL_WEIGHTS = {
    'ml_predictor': 0.55,       # 55% - ML прогноз
    'mtf_consensus': 0.30,      # 30% - Multi-timeframe consensus
    'pattern_detector': 0.15,   # 15% - Pattern detection
}

# ============================================================================
# DATABASE
# ============================================================================
DATABASE_CONFIG = {
    'enabled': True,
    'url': os.getenv('DATABASE_URL'),
    'pool_size': 20,
    'max_overflow': 30,
    'pool_recycle': 3600,
}

# ============================================================================
# TELEGRAM
# ============================================================================
TELEGRAM_CONFIG = {
    'bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
    'chat_id': os.getenv('TELEGRAM_CHAT_ID'),
    'enabled': True,
}

# ============================================================================
# LOGGING
# ============================================================================
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_dir': 'logs',
}

# ============================================================================
# MODELS
# ============================================================================
MODEL_CONFIG = {
    'base_dir': 'models',
    'strategy': 'simple_trend',  # simple_trend_SYMBOL
    'n_features': 82,
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def get_model_path(symbol: str) -> str:
    """Отримати шлях до моделі для символа"""
    return f"{MODEL_CONFIG['base_dir']}/{MODEL_CONFIG['strategy']}_{symbol}"


def validate_config():
    """Валідація конфігурації"""
    errors = []
    
    # Перевірка API ключів
    if not BINANCE_CONFIG['api_key']:
        errors.append("❌ FUTURES_API_KEY not set")
    if not BINANCE_CONFIG['api_secret']:
        errors.append("❌ FUTURES_API_SECRET not set")
    
    # Перевірка Telegram
    if TELEGRAM_CONFIG['enabled']:
        if not TELEGRAM_CONFIG['bot_token']:
            errors.append("❌ TELEGRAM_BOT_TOKEN not set")
        if not TELEGRAM_CONFIG['chat_id']:
            errors.append("❌ TELEGRAM_CHAT_ID not set")
    
    # Перевірка Database
    if DATABASE_CONFIG['enabled']:
        if not DATABASE_CONFIG['url']:
            errors.append("❌ DATABASE_URL not set")
    
    if errors:
        for error in errors:
            print(error)
        raise ValueError("Configuration validation failed")
    
    return True


if __name__ == '__main__':
    # Тест конфігурації
    print("="*70)
    print("🔧 КОНФІГУРАЦІЯ СИСТЕМИ")
    print("="*70)
    
    try:
        validate_config()
        print("\n✅ Всі налаштування валідні")
        
        print(f"\n📊 Trading:")
        print(f"   Symbols: {', '.join(TRADING_CONFIG['symbols'])}")
        print(f"   Position: ${TRADING_CONFIG['position_size_usd']} ({TRADING_CONFIG['leverage']}x)")
        print(f"   Min Confidence: {TRADING_CONFIG['min_confidence']:.0%}")
        
        print(f"\n🔄 MTF: {'✅' if MTF_CONFIG['enabled'] else '❌'}")
        print(f"\n💾 Database: {'✅' if DATABASE_CONFIG['enabled'] else '❌'}")
        print(f"\n📱 Telegram: {'✅' if TELEGRAM_CONFIG['enabled'] else '❌'}")
        
    except ValueError as e:
        print(f"\n❌ {e}")
        exit(1)
