#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔧 ЦЕНТРАЛІЗОВАНА КОНФІГУРАЦІЯ СИСТЕМИ
Всі налаштування в одному місці
"""
import os
from dotenv import load_dotenv
from model_scanner import get_available_models

load_dotenv()

# ============================================================================
# BINANCE API
# ============================================================================
BINANCE_CONFIG = {
    'api_key': os.getenv('FUTURES_API_KEY'),
    'api_secret': os.getenv('FUTURES_API_SECRET'),
    'testnet': True,  # Тільки testnet для безпеки
}

# ============================================================================
# АВТОМАТИЧНЕ ВИЯВЛЕННЯ МОДЕЛЕЙ
# ============================================================================
# Тільки пари з хорошими моделями на 15m+1h (БЕЗ BNBUSDT - нестабільна в testnet)
TRADING_SYMBOLS = [
    # ТОП-2: Відмінні на всіх таймфреймах
    'BTCUSDT',   # 93% (15m), 86% (1h) - НАЙКРАЩА
    'TRXUSDT',   # 86% (15m), 79% (1h) - СТАБІЛЬНА
    
    # Додаткові: Хороші моделі
    'SOLUSDT',   # 76% (15m), 67% (1h) - волатільна
    'XRPUSDT',   # 70% (15m), 65% (1h) - банківська
    'ETHUSDT',   # 67% (15m), 65% (1h) - смарт-контракти
    # 'BNBUSDT' ВИДАЛЕНО - в testnet жахливо скаче, створює phantom shorts
]

# ============================================================================
# TRADING PARAMETERS
# ============================================================================
TRADING_CONFIG = {
    # Основні параметри - збалансований портфель
    'symbols': TRADING_SYMBOLS,  # 10 пар (5 стабільних + 5 волатільних)
    
    # Ризик-менеджмент (ОПТИМІЗОВАНО ДЛЯ 15m-1h-4h)
    'leverage': 25,                     # Плече 25x
    'position_size_usd': 1250,          # $1250 на позицію
    'stop_loss_pct': 0.020,             # SL 2.0% = 50% втрати (дати рухатись!)
    'take_profit_pct': 0.025,           # TP 2.5% = 62.5% прибутку (більший потенціал)
    'max_positions': 9,                 # 9 звичайних + 1 резерв для розвороту
    'reserve_slot_for_reversal': True,  # 🔄 Завжди залишати 1 слот вільним для розвороту
    
    # 🔄 РОЗВОРОТ ПОЗИЦІЙ: агресивніші налаштування для швидшої реакції
    'reverse_on_strong_signal': True,      # Закривати + відкривати зворотню угоду
    'reverse_min_confidence': 0.70,        # 70% впевненості
    'reverse_profit_threshold': -0.005,    # якщо PnL < -0.5% (раніше реагуємо)
    
    # Впевненість та фільтри (ПІДВИЩЕНО для якісних сигналів)
    'min_confidence': 0.70,          # 70% (підвищено - фільтруємо слабкі сигнали)
    'min_consensus': 0.85,           # 85% консенсус (15m+1h повинні погоджуватись)
    
    # Захист
    'max_daily_losses_per_symbol': 2,  # Максимум 2 програші на день (було 3)
    'cooldown_after_sl': 3600,         # 1 година після SL
    'cooldown_after_tp': 1800,         # 30 хвилин після TP
    'cooldown_after_force_close': 10800, # 3 години після force close (було 2 години!)
    'max_loss_per_symbol_pct': 0.03,   # Максимум -3% за день (було -5%)
    
    # Таймінги (ШВИДКА РЕАКЦІЯ)
    'check_interval': 300,             # 5 хвилин між перевірками (було 15хв)
    
    # Адаптація до волатильності (нічний режим)
    'night_mode': {
        'enabled': True,
        'start_hour': 0,                   # 00:00
        'end_hour': 8,                     # 08:00
        'interval_multiplier': 1.5,        # Збільшити інтервал на 50%
        'min_confidence_boost': 0.05,      # Додати 5% до впевненості
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
    
    # Trailing stop для захисту прибутку (ЗБАЛАНСОВАНИЙ - дати прибутку зрости)
    'trailing_stop': {
        'enabled': True,                   # Використовувати trailing stop
        'activation_profit': 0.015,        # Активувати при +1.5% прибутку (37.5% на депозит)
        'trail_distance': 0.50,            # Закривати при відкаті на 50% від піку (більше простору)
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
# WEBSOCKET
# ============================================================================
WEBSOCKET_CONFIG = {
    'enabled': True,
    'reconnect_delay': 5,
    'ping_interval': 60,
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
        print(f"\n🔌 WebSocket: {'✅' if WEBSOCKET_CONFIG['enabled'] else '❌'}")
        
    except ValueError as e:
        print(f"\n❌ {e}")
        exit(1)
