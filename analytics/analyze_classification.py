#!/usr/bin/env python3
"""
Аналіз класифікаційної моделі та генерація торгових сигналів
"""
import sys
import os
import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.rust_features import RustFeatureEngineer
from unified_binance_loader import UnifiedBinanceLoader
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Вимкнути GPU для стабільності
os.environ['CUDA_VISIBLE_DEVICES'] = ''
load_dotenv()


class ClassificationAnalyzer:
    """Аналізатор класифікаційної моделі"""
    
    CLASSES = ['DOWN', 'NEUTRAL', 'UP']
    
    def __init__(self, symbol: str = 'BTCUSDT', model_path: str = None):
        self.symbol = symbol
        self.model_path = model_path or 'models/classification_BTC/model_resumed_20251021_125248.keras'
        self.model = None
        self.feature_engineer = RustFeatureEngineer(use_rust=True)
        
        # Binance loader
        api_key = os.getenv('FUTURES_API_KEY')
        api_secret = os.getenv('FUTURES_API_SECRET')
        use_testnet = os.getenv('USE_TESTNET', 'false').lower() in ('true', '1', 'yes')
        
        self.data_loader = UnifiedBinanceLoader(
            api_key=api_key,
            api_secret=api_secret,
            testnet=use_testnet,
            use_public_data=True
        )
        
    async def load_model(self):
        """Завантаження моделі"""
        logger.info(f"📦 Завантаження класифікаційної моделі...")
        logger.info(f"   {self.model_path}")
        
        self.model = tf.keras.models.load_model(self.model_path)
        
        logger.info(f"✅ Модель завантажено:")
        logger.info(f"   Параметрів: {self.model.count_params():,}")
        logger.info(f"   Input shape: {self.model.input_shape}")
        logger.info(f"   Output: {self.model.output_shape[-1]} класів (DOWN, NEUTRAL, UP)")
    
    async def load_data(self, days: int = 14):
        """Завантаження даних"""
        logger.info(f"📥 Завантаження даних за {days} днів...")
        
        df = await self.data_loader.get_historical_data(
            symbol=self.symbol,
            interval='1h',
            days_back=days
        )
        
        if df is None or df.empty:
            raise ValueError("Не вдалося завантажити дані")
        
        logger.info(f"✅ Завантажено {len(df)} записів")
        return df
    
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Розрахунок features"""
        logger.info("🔧 Розрахунок features...")
        
        # Базові features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Rust індикатори
        df = self.feature_engineer.calculate_all(
            df,
            sma_periods=[10, 20, 50],
            ema_periods=[12, 20, 26, 50],
            rsi_periods=[7, 14, 28],
            atr_periods=[7, 14, 21]
        )
        
        # Volatility
        df['close_std_5'] = df['close'].rolling(5).std()
        df['close_std_10'] = df['close'].rolling(10).std()
        df['close_std_20'] = df['close'].rolling(20).std()
        
        # Volume features
        df['volume_mean_5'] = df['volume'].rolling(5).mean()
        df['volume_mean_10'] = df['volume'].rolling(10).mean()
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Price distance from SMA
        df['dist_from_sma_10'] = (df['close'] - df['sma_10']) / df['sma_10']
        df['dist_from_sma_20'] = (df['close'] - df['sma_20']) / df['sma_20']
        
        df = df.dropna()
        
        # Вибрати потрібні 41 features
        feature_cols = [
            'open', 'high', 'low', 'volume',
            'returns', 'log_returns',
            'sma_10', 'sma_20', 'sma_50',
            'ema_12', 'ema_20', 'ema_26', 'ema_50',
            'rsi_7', 'rsi_14', 'rsi_28',
            'roc_5', 'roc_10', 'roc_20',
            'price_momentum', 'acceleration',
            'atr_7', 'atr_14', 'atr_21',
            'close_std_5', 'close_std_10', 'close_std_20',
            'hvol_10', 'hvol_20', 'hvol_30',
            'bb_width_20', 'bb_percent_20',
            'obv', 'vwap',
            'volume_mean_5', 'volume_mean_10', 'volume_ratio',
            'dist_from_sma_10', 'dist_from_sma_20',
            'body', 'body_ratio'
        ]
        
        available = [f for f in feature_cols if f in df.columns]
        logger.info(f"✅ Використовується {len(available)}/41 features")
        
        return df[['close'] + available]
    
    def create_sequences(self, df: pd.DataFrame, sequence_length: int = 120):
        """Створення послідовностей"""
        feature_cols = [c for c in df.columns if c != 'close']
        X = df[feature_cols].values
        prices = df['close'].values
        
        # Нормалізація
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Послідовності
        sequences = []
        sequence_prices = []
        
        for i in range(len(X_scaled) - sequence_length + 1):
            sequences.append(X_scaled[i:i + sequence_length])
            sequence_prices.append(prices[i + sequence_length - 1])
        
        return np.array(sequences), np.array(sequence_prices)
    
    async def predict(self, X: np.ndarray):
        """Прогнозування"""
        logger.info(f"🔮 Прогнозування для {len(X)} послідовностей...")
        
        predictions = self.model.predict(X, verbose=0)
        
        # 🔄 ІНВЕРСІЯ СИГНАЛІВ: модель навчена інвертовано!
        # Міняємо місцями ймовірності UP (індекс 2) та DOWN (індекс 0)
        predictions_inverted = predictions.copy()
        predictions_inverted[:, 0] = predictions[:, 2]  # DOWN <- UP
        predictions_inverted[:, 2] = predictions[:, 0]  # UP <- DOWN
        # NEUTRAL (індекс 1) залишається без змін
        
        predicted_classes = np.argmax(predictions_inverted, axis=1)
        confidences = np.max(predictions_inverted, axis=1)
        
        logger.info(f"✅ Прогнози отримані (з інверсією UP↔DOWN)")
        
        return predicted_classes, confidences, predictions_inverted
    
    def generate_signals(self, predicted_classes, confidences, probabilities, prices):
        """Генерація сигналів з помірно-агресивною інтерпретацією"""
        logger.info("📊 Генерація торгових сигналів (помірно-агресивний режим)...")
        
        signals = []
        
        for i, (pred_class, conf, probs, price) in enumerate(
            zip(predicted_classes, confidences, probabilities, prices)
        ):
            prob_down = probs[0]
            prob_neutral = probs[1]
            prob_up = probs[2]
            
            # ПОМІРНО-АГРЕСИВНА ІНТЕРПРЕТАЦІЯ:
            # Головний принцип: NEUTRAL допускається тільки при дуже високій впевненості (>55%)
            # або коли UP і DOWN майже рівні (різниця <2%)
            
            up_down_diff = prob_up - prob_down
            max_prob = max(prob_down, prob_neutral, prob_up)
            
            # Крок 1: Перевіряємо чи UP/DOWN мають достатню перевагу
            if abs(up_down_diff) > 0.04:  # Різниця >4% - вже приймаємо рішення
                if up_down_diff > 0:
                    signal_name = 'UP'
                    effective_conf = prob_up
                else:
                    signal_name = 'DOWN'
                    effective_conf = prob_down
            
            # Крок 2: Якщо різниця мала, дивимось на абсолютні значення
            elif prob_neutral > 0.55:  # Дуже висока впевненість в NEUTRAL
                signal_name = 'NEUTRAL'
                effective_conf = prob_neutral
            
            elif prob_up > 0.35 or prob_down > 0.35:  # Хтось має >35% - приймаємо рішення
                if prob_up > prob_down:
                    signal_name = 'UP'
                    effective_conf = prob_up
                else:
                    signal_name = 'DOWN'
                    effective_conf = prob_down
            
            # Крок 3: Останній варіант - якщо все дуже невизначено
            else:
                # Навіть при малій різниці - вибираємо сторону
                if abs(up_down_diff) > 0.01:  # Хоча б 1% різниці
                    if up_down_diff > 0:
                        signal_name = 'UP'
                        effective_conf = prob_up
                    else:
                        signal_name = 'DOWN'
                        effective_conf = prob_down
                else:
                    # Повна невизначеність - тільки тоді NEUTRAL
                    signal_name = 'NEUTRAL'
                    effective_conf = prob_neutral
            
            # Рекомендація на основі впевненості та різниці
            if signal_name == 'UP':
                if prob_up > 0.50 or up_down_diff > 0.10:
                    recommendation = 'STRONG BUY'
                elif prob_up > 0.40 or up_down_diff > 0.06:
                    recommendation = 'BUY'
                else:
                    recommendation = 'WEAK BUY'
            elif signal_name == 'DOWN':
                if prob_down > 0.50 or up_down_diff < -0.10:
                    recommendation = 'STRONG SELL'
                elif prob_down > 0.40 or up_down_diff < -0.06:
                    recommendation = 'SELL'
                else:
                    recommendation = 'WEAK SELL'
            else:
                recommendation = 'HOLD'
            
            signals.append({
                'index': i,
                'price': price,
                'prediction': signal_name,
                'confidence': effective_conf,
                'up_down_diff': up_down_diff,
                'recommendation': recommendation,
                'prob_down': prob_down,
                'prob_neutral': prob_neutral,
                'prob_up': prob_up
            })
        
        return pd.DataFrame(signals)
    
    def print_analysis(self, signals_df: pd.DataFrame, recent_n: int = 25):
        """Аналіз та вивід"""
        logger.info("\n" + "="*90)
        logger.info("📈 КЛАСИФІКАЦІЙНИЙ АНАЛІЗ МОДЕЛІ")
        logger.info("="*90)
        
        logger.info(f"\n🎯 Останні {recent_n} прогнозів:")
        logger.info("-" * 90)
        
        recent = signals_df.tail(recent_n)
        
        for _, row in recent.iterrows():
            emoji = {'DOWN': '🔴', 'NEUTRAL': '⚪', 'UP': '🟢'}[row['prediction']]
            
            # Стрілка тренду
            diff = row['up_down_diff']
            if diff > 0.05:
                trend = '⬆️⬆️'
            elif diff > 0.02:
                trend = '⬆️'
            elif diff < -0.05:
                trend = '⬇️⬇️'
            elif diff < -0.02:
                trend = '⬇️'
            else:
                trend = '➡️'
            
            logger.info(
                f"{emoji} {row['prediction']:7s} {trend} | "
                f"${row['price']:,.2f} | "
                f"Різниця: {diff*100:+5.1f}% | "
                f"{row['recommendation']:13s} | "
                f"P(↓):{row['prob_down']*100:4.1f}% P(→):{row['prob_neutral']*100:4.1f}% P(↑):{row['prob_up']*100:4.1f}%"
            )
        
        # Статистика
        logger.info("\n" + "="*90)
        logger.info("📊 СТАТИСТИКА ПРОГНОЗІВ")
        logger.info("="*90)
        
        down_count = len(signals_df[signals_df['prediction'] == 'DOWN'])
        neutral_count = len(signals_df[signals_df['prediction'] == 'NEUTRAL'])
        up_count = len(signals_df[signals_df['prediction'] == 'UP'])
        total = len(signals_df)
        
        logger.info(f"🔴 DOWN прогнозів:    {down_count:4d} ({down_count/total*100:.1f}%)")
        logger.info(f"⚪ NEUTRAL прогнозів: {neutral_count:4d} ({neutral_count/total*100:.1f}%)")
        logger.info(f"🟢 UP прогнозів:      {up_count:4d} ({up_count/total*100:.1f}%)")
        
        avg_conf = signals_df['confidence'].mean()
        logger.info(f"\n📈 Середня впевненість: {avg_conf*100:.1f}%")
        logger.info(f"📊 Мінімальна впевненість: {signals_df['confidence'].min()*100:.1f}%")
        logger.info(f"📊 Максимальна впевненість: {signals_df['confidence'].max()*100:.1f}%")
        
        # Поточний сигнал
        logger.info("\n" + "="*90)
        logger.info("🎯 ПОТОЧНИЙ ТОРГОВИЙ СИГНАЛ")
        logger.info("="*90)
        
        current = signals_df.iloc[-1]
        emoji = {'DOWN': '🔴', 'NEUTRAL': '⚪', 'UP': '🟢'}[current['prediction']]
        
        # Аналіз тренду
        diff = current['up_down_diff']
        if diff > 0.10:
            trend_desc = "Сильний бичачий тренд 🐂"
        elif diff > 0.05:
            trend_desc = "Помірний бичачий тренд ↗️"
        elif diff > 0.02:
            trend_desc = "Слабкий бичачий тренд ↗"
        elif diff < -0.10:
            trend_desc = "Сильний ведмежий тренд 🐻"
        elif diff < -0.05:
            trend_desc = "Помірний ведмежий тренд ↘️"
        elif diff < -0.02:
            trend_desc = "Слабкий ведмежий тренд ↘"
        else:
            trend_desc = "Флет, невизначеність ➡️"
        
        logger.info(f"\n{emoji} {current['prediction']}")
        logger.info(f"💰 Поточна ціна: ${current['price']:,.2f}")
        logger.info(f"📊 Різниця UP-DOWN: {diff*100:+.1f}%")
        logger.info(f"📈 Тренд: {trend_desc}")
        logger.info(f"💡 Рекомендація: {current['recommendation']}")
        logger.info(f"\n📉 Ймовірності:")
        logger.info(f"   🔴 Падіння (DOWN):    {current['prob_down']*100:5.1f}%")
        logger.info(f"   ⚪ Флет (NEUTRAL):    {current['prob_neutral']*100:5.1f}%")
        logger.info(f"   🟢 Зростання (UP):    {current['prob_up']*100:5.1f}%")
        
        # Торгові рекомендації
        logger.info(f"\n💼 ТОРГОВА СТРАТЕГІЯ:")
        if current['recommendation'] in ['STRONG BUY', 'BUY']:
            logger.info(f"   ✅ Відкрити LONG позицію")
            logger.info(f"   🎯 Target: +1-2%")
            logger.info(f"   🛑 Stop-loss: -0.5%")
        elif current['recommendation'] in ['STRONG SELL', 'SELL']:
            logger.info(f"   ✅ Відкрити SHORT позицію")
            logger.info(f"   🎯 Target: -1-2%")
            logger.info(f"   🛑 Stop-loss: +0.5%")
        elif current['recommendation'] in ['WEAK BUY']:
            logger.info(f"   ⚠️ Можна розглянути малий LONG (обережно)")
            logger.info(f"   🎯 Target: +0.5-1%")
            logger.info(f"   🛑 Stop-loss: -0.3%")
        elif current['recommendation'] in ['WEAK SELL']:
            logger.info(f"   ⚠️ Можна розглянути малий SHORT (обережно)")
            logger.info(f"   🎯 Target: -0.5-1%")
            logger.info(f"   🛑 Stop-loss: +0.3%")
        else:
            logger.info(f"   💤 Чекати кращого моменту")
            logger.info(f"   👀 Моніторити ситуацію")
        
        logger.info("\n" + "="*90)


async def main():
    """Головна функція"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Аналіз класифікаційної моделі')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Символ')
    parser.add_argument('--model', type=str, help='Шлях до моделі')
    parser.add_argument('--days', type=int, default=14, help='Днів даних')
    parser.add_argument('--recent', type=int, default=25, help='Скільки показати')
    
    args = parser.parse_args()
    
    try:
        analyzer = ClassificationAnalyzer(symbol=args.symbol, model_path=args.model)
        
        await analyzer.load_model()
        df = await analyzer.load_data(days=args.days)
        df_features = analyzer.calculate_features(df)
        X, prices = analyzer.create_sequences(df_features)
        
        predicted_classes, confidences, probabilities = await analyzer.predict(X)
        signals_df = analyzer.generate_signals(predicted_classes, confidences, probabilities, prices)
        
        analyzer.print_analysis(signals_df, recent_n=args.recent)
        
        # Збереження в graphics/csv/
        os.makedirs('graphics/csv', exist_ok=True)
        output = f'graphics/csv/classification_analysis_{args.symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        signals_df.to_csv(output, index=False)
        logger.info(f"\n💾 Результати збережено: {output}")
        
    except Exception as e:
        logger.error(f"❌ Помилка: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
