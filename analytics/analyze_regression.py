#!/usr/bin/env python3
"""
Аналіз існуючої моделі та генерація торгових сигналів
"""
import sys
import os
import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.rust_features import RustFeatureEngineer
from selected_features import SELECTED_FEATURES
from gpu_config import configure_gpu
from unified_binance_loader import UnifiedBinanceLoader
from dotenv import load_dotenv
import joblib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Налаштування GPU
configure_gpu()
load_dotenv()


class ModelAnalyzer:
    """Аналізатор моделі та генератор торгових сигналів"""
    
    def __init__(self, symbol: str = 'BTCUSDT', model_dir: str = 'models/optimized_BTC'):
        self.symbol = symbol
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.feature_engineer = RustFeatureEngineer(use_rust=True)
        
        # Binance loader
        api_key = os.getenv('FUTURES_API_KEY')
        api_secret = os.getenv('FUTURES_API_SECRET')
        use_testnet = os.getenv('USE_TESTNET', 'false').lower() in ('true', '1', 'yes')
        
        self.data_loader = UnifiedBinanceLoader(
            api_key=api_key,
            api_secret=api_secret,
            testnet=use_testnet,
            use_public_data=True  # Використовувати публічні дані якщо API не доступне
        )
        
    async def load_model(self):
        """Завантаження моделі та scaler"""
        logger.info(f"📦 Завантаження моделі з {self.model_dir}")
        
        # Завантаження моделі
        model_path = os.path.join(self.model_dir, 'best_model.h5')
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            logger.info(f"✅ Модель завантажено: {self.model.count_params():,} параметрів")
        else:
            raise FileNotFoundError(f"Модель не знайдена: {model_path}")
        
        # Завантаження scaler
        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            logger.info(f"✅ Scaler завантажено")
        else:
            logger.warning("⚠️ Scaler не знайдено, нормалізація буде пропущена")
    
    async def load_recent_data(self, days: int = 7):
        """Завантаження останніх даних"""
        logger.info(f"📥 Завантаження даних за останні {days} днів для {self.symbol}")
        
        df = await self.data_loader.get_historical_data(
            symbol=self.symbol,
            interval='1h',
            days_back=days
        )
        
        if df is None or df.empty:
            raise ValueError("Не вдалося завантажити дані")
        
        logger.info(f"✅ Завантажено {len(df)} записів")
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Підготовка features"""
        logger.info("🔧 Розрахунок features...")
        
        # Розрахунок індикаторів
        df = self.feature_engineer.calculate_all(
            df,
            sma_periods=[10, 20, 50],
            ema_periods=[12, 20, 26, 50],
            rsi_periods=[7, 14, 28],
            atr_periods=[7, 14, 21]
        )
        
        # Визначити скільки features очікує scaler
        expected_features = self.scaler.n_features_in_ if self.scaler else len(SELECTED_FEATURES)
        
        # Використати тільки перші N features які очікує модель
        features_to_use = SELECTED_FEATURES[:expected_features]
        
        # Відбір features
        available_features = [f for f in features_to_use if f in df.columns]
        missing_features = [f for f in features_to_use if f not in df.columns]
        
        if missing_features:
            logger.warning(f"⚠️ Відсутні features: {missing_features[:5]}...")
        
        logger.info(f"✅ Використовується {len(available_features)}/{expected_features} features")
        
        # Зберігаємо close для аналізу
        df_features = df[available_features].copy()
        df_features = df_features.dropna()
        
        return df_features, df['close'].iloc[-len(df_features):]
    
    def create_sequences(self, X: np.ndarray, sequence_length: int = 60):
        """Створення послідовностей для LSTM"""
        if len(X) < sequence_length:
            raise ValueError(f"Недостатньо даних: {len(X)} < {sequence_length}")
        
        # Нормалізація
        if self.scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        # Створення послідовностей
        sequences = []
        for i in range(len(X_scaled) - sequence_length + 1):
            sequences.append(X_scaled[i:i + sequence_length])
        
        return np.array(sequences)
    
    async def predict(self, X_sequences: np.ndarray) -> np.ndarray:
        """Прогнозування"""
        logger.info(f"🔮 Прогнозування для {len(X_sequences)} послідовностей...")
        
        predictions = self.model.predict(X_sequences, verbose=0)
        
        logger.info(f"✅ Прогнози отримані")
        return predictions.flatten()
    
    def generate_signals(self, predictions: np.ndarray, prices: pd.Series, threshold: float = 0.5):
        """Генерація торгових сигналів"""
        logger.info("📊 Генерація торгових сигналів...")
        
        signals = []
        sequence_length = 60
        
        # predictions відповідають останнім даним
        actual_prices = prices.iloc[-len(predictions):].values
        
        for i, (pred, price) in enumerate(zip(predictions, actual_prices)):
            # Прогноз відносної зміни ціни
            predicted_change = pred  # Модель прогнозує відносну зміну
            
            # Визначення сигналу
            if predicted_change > threshold:
                signal = "BUY"
                strength = min(predicted_change / threshold, 3.0)
            elif predicted_change < -threshold:
                signal = "SELL"
                strength = min(abs(predicted_change) / threshold, 3.0)
            else:
                signal = "HOLD"
                strength = 0.0
            
            signals.append({
                'index': i,
                'current_price': price,
                'predicted_change': predicted_change,
                'predicted_change_pct': predicted_change * 100,
                'signal': signal,
                'strength': strength,
                'confidence': min(abs(predicted_change), 1.0)
            })
        
        return pd.DataFrame(signals)
    
    def print_analysis(self, signals_df: pd.DataFrame, recent_n: int = 20):
        """Вивести аналіз"""
        logger.info("\n" + "="*80)
        logger.info("📈 АНАЛІЗ МОДЕЛІ ТА ТОРГОВІ СИГНАЛИ")
        logger.info("="*80)
        
        logger.info(f"\n🎯 Останні {recent_n} прогнозів:")
        logger.info("-" * 80)
        
        recent = signals_df.tail(recent_n)
        
        for _, row in recent.iterrows():
            signal_emoji = {
                'BUY': '🟢',
                'SELL': '🔴',
                'HOLD': '⚪'
            }[row['signal']]
            
            logger.info(
                f"{signal_emoji} {row['signal']:4s} | "
                f"Ціна: ${row['current_price']:.2f} | "
                f"Прогноз: {row['predicted_change_pct']:+.2f}% | "
                f"Сила: {'█' * int(row['strength'])} {row['strength']:.2f}"
            )
        
        # Статистика
        logger.info("\n" + "="*80)
        logger.info("📊 СТАТИСТИКА")
        logger.info("="*80)
        
        buy_signals = len(signals_df[signals_df['signal'] == 'BUY'])
        sell_signals = len(signals_df[signals_df['signal'] == 'SELL'])
        hold_signals = len(signals_df[signals_df['signal'] == 'HOLD'])
        
        logger.info(f"🟢 BUY сигналів:  {buy_signals:4d} ({buy_signals/len(signals_df)*100:.1f}%)")
        logger.info(f"🔴 SELL сигналів: {sell_signals:4d} ({sell_signals/len(signals_df)*100:.1f}%)")
        logger.info(f"⚪ HOLD сигналів: {hold_signals:4d} ({hold_signals/len(signals_df)*100:.1f}%)")
        
        avg_prediction = signals_df['predicted_change_pct'].mean()
        logger.info(f"\n📈 Середній прогноз: {avg_prediction:+.2f}%")
        logger.info(f"📊 Мін прогноз: {signals_df['predicted_change_pct'].min():+.2f}%")
        logger.info(f"📊 Макс прогноз: {signals_df['predicted_change_pct'].max():+.2f}%")
        
        # Поточний сигнал
        logger.info("\n" + "="*80)
        logger.info("🎯 ПОТОЧНИЙ СИГНАЛ")
        logger.info("="*80)
        
        current = signals_df.iloc[-1]
        signal_emoji = {
            'BUY': '🟢',
            'SELL': '🔴',
            'HOLD': '⚪'
        }[current['signal']]
        
        logger.info(f"{signal_emoji} {current['signal']}")
        logger.info(f"Поточна ціна: ${current['current_price']:.2f}")
        logger.info(f"Прогнозована зміна: {current['predicted_change_pct']:+.2f}%")
        logger.info(f"Впевненість: {current['confidence']*100:.1f}%")
        logger.info(f"Сила сигналу: {current['strength']:.2f}/3.0")
        
        logger.info("\n" + "="*80)


async def main():
    """Головна функція"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Аналіз моделі та генерація сигналів')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Торговий символ')
    parser.add_argument('--model-dir', type=str, default='models/optimized_BTC', help='Папка з моделлю')
    parser.add_argument('--days', type=int, default=7, help='Скільки днів даних завантажити')
    parser.add_argument('--threshold', type=float, default=0.005, help='Поріг для сигналів (0.005 = 0.5%)')
    parser.add_argument('--recent', type=int, default=20, help='Скільки останніх сигналів показати')
    
    args = parser.parse_args()
    
    try:
        # Створення аналізатора
        analyzer = ModelAnalyzer(symbol=args.symbol, model_dir=args.model_dir)
        
        # Завантаження моделі
        await analyzer.load_model()
        
        # Завантаження даних
        df = await analyzer.load_recent_data(days=args.days)
        
        # Підготовка features
        df_features, prices = analyzer.prepare_features(df)
        
        # Створення послідовностей
        X_sequences = analyzer.create_sequences(df_features.values)
        
        # Прогнозування
        predictions = await analyzer.predict(X_sequences)
        
        # Генерація сигналів
        signals_df = analyzer.generate_signals(predictions, prices, threshold=args.threshold)
        
        # Вивести аналіз
        analyzer.print_analysis(signals_df, recent_n=args.recent)
        
        # Зберегти результати
        output_file = f'analysis_{args.symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        signals_df.to_csv(output_file, index=False)
        logger.info(f"\n💾 Результати збережено: {output_file}")
        
    except Exception as e:
        logger.error(f"❌ Помилка: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
