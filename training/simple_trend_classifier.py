#!/usr/bin/env python3
"""
Проста модель класифікації тренду з accuracy 70%+
Стратегія: Daily timeframe + trend indicators + Random Forest
"""

import os
import sys
import asyncio
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, Dict, List
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from binance.client import Client
from training.rust_features import RustFeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleTrendClassifier:
    """
    Проста класифікація тренду з високою точністю
    
    Підхід:
    - Daily timeframe (менше шуму)
    - Trend labels: Strong Up (2), Weak Up (1), Weak Down (-1), Strong Down (-2)
    - Random Forest (не overfitting як GRU)
    - Top 20 найважливіших features
    """
    
    def __init__(self, symbol: str, timeframe: str = '1d'):
        self.symbol = symbol
        self.timeframe = timeframe
        self.client = Client()  # Public API, no keys needed
        self.feature_engineer = RustFeatureEngineer()
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = []
        
        # Model configs (менше параметрів щоб уникнути overfitting)
        self.n_estimators = 50  # Менше дерев
        self.max_depth = 5      # Менша глибина
        self.min_samples_split = 50  # Більше samples на split
        
    def _create_trend_labels(self, df: pd.DataFrame, lookback: int = 3) -> pd.Series:
        """
        Створюємо BINARY trend labels (простіше = краще)
        
        Binary classification:
        - UP (1): +1.5%+ за 3 періоди (4h: 12h, 1d: 3 дні)
        - DOWN (0): інше
        """
        # Відсоткова зміна за lookback період
        pct_change = (df['close'].shift(-lookback) / df['close'] - 1) * 100
        
        # Просто UP/DOWN з порогом 1.5%
        labels = (pct_change >= 1.5).astype(int)
        
        return labels
    
    def _create_simple_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Створюємо прості, але ефективні features
        
        Focus на trend indicators, не на noise
        """
        df = df.copy()
        
        # Trend indicators
        df = self.feature_engineer.calculate_all(
            df,
            sma_periods=[5, 10, 20, 50, 100, 200],
            ema_periods=[9, 12, 21, 26, 50],
            rsi_periods=[7, 14, 21, 28],
            atr_periods=[14, 21],
        )
        
        # Price relative to MAs (trend strength)
        for period in [5, 10, 20, 50, 100, 200]:
            ma_col = f'sma_{period}'
            if ma_col in df.columns:
                df[f'price_vs_sma{period}'] = (df['close'] / df[ma_col] - 1) * 100
        
        # MA crossovers (trend changes)
        if 'sma_50' in df.columns and 'sma_200' in df.columns:
            df['golden_cross'] = (df['sma_50'] > df['sma_200']).astype(int)
            
        if 'ema_12' in df.columns and 'ema_26' in df.columns:
            df['macd_cross'] = (df['ema_12'] > df['ema_26']).astype(int)
        
        # Volume trend
        if 'volume' in df.columns:
            df['volume_sma20'] = df['volume'].rolling(20).mean()
            df['volume_trend'] = df['volume'] / df['volume_sma20']
        
        # RSI levels (overbought/oversold)
        if 'rsi_14' in df.columns:
            df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
            df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
        
        # Momentum
        df['momentum_5'] = df['close'].pct_change(5) * 100
        df['momentum_10'] = df['close'].pct_change(10) * 100
        df['momentum_20'] = df['close'].pct_change(20) * 100
        
        # Volatility
        df['volatility_20'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean() * 100
        
        return df
    
    async def prepare_data(self, days: int = 730) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Підготовка даних для тренування
        
        Args:
            days: Кількість днів історії (2 роки для daily)
        """
        logger.info(f"📥 Завантаження {self.timeframe} даних для {self.symbol}...")
        
        # Завантажуємо через python-binance
        start_str = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        klines = self.client.get_historical_klines(
            symbol=self.symbol,
            interval=self.timeframe,
            start_str=start_str
        )
        
        df = pd.DataFrame(
            klines,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignore']
        )
        
        # Конвертуємо типи
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        # Залишаємо тільки потрібні колонки
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        if df is None or len(df) < 100:
            raise ValueError(f"Недостатньо даних: {len(df) if df is not None else 0} рядків")
        
        logger.info(f"✅ Завантажено {len(df)} рядків")
        
        # Створюємо features
        logger.info("🔧 Розрахунок features...")
        df = self._create_simple_features(df)
        
        # Створюємо labels
        logger.info("🏷️ Створення trend labels...")
        df['trend_label'] = self._create_trend_labels(df, lookback=5)
        
        # Видаляємо NaN
        df = df.dropna()
        
        logger.info(f"✅ Після очистки: {len(df)} рядків")
        
        # Label distribution
        label_counts = df['trend_label'].value_counts().sort_index()
        logger.info(f"📊 Label distribution:")
        for label, count in label_counts.items():
            pct = count / len(df) * 100
            label_name = {0: 'DOWN', 1: 'UP'}[label]
            logger.info(f"   {label_name:15s}: {count:4d} ({pct:5.1f}%)")
        
        # Відбір features
        feature_cols = [col for col in df.columns 
                       if col not in ['open', 'high', 'low', 'close', 'volume', 
                                     'timestamp', 'trend_label']]
        
        # Перевірка на infinity/NaN
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=feature_cols)
        
        X = df[feature_cols].values
        y = df['trend_label'].values
        
        logger.info(f"✅ Features: {len(feature_cols)}, Samples: {len(X)}")
        
        # Feature importance з попереднього тренування (якщо є)
        self.feature_names = feature_cols
        
        return X, y, feature_cols
    
    async def train(self, days: int = 730) -> Dict:
        """
        Тренування моделі
        
        Returns:
            Dict з результатами
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"🚀 SIMPLE TREND CLASSIFIER: {self.symbol}")
        logger.info(f"{'='*80}\n")
        
        start_time = datetime.now()
        
        # Підготовка даних
        X, y, feature_names = await self.prepare_data(days=days)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False  # Time series - no shuffle!
        )
        
        logger.info(f"📊 Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Scaling
        logger.info("⚖️ Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Тренування Random Forest (найкраща одиночна модель)
        logger.info(f"🌲 Тренування Random Forest (n_estimators={self.n_estimators})...")
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'  # Handle imbalanced classes
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Оцінка
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 РЕЗУЛЬТАТИ")
        logger.info(f"{'='*80}")
        logger.info(f"Train Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
        logger.info(f"Test Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
        logger.info(f"\nTest Classification Report:")
        print(classification_report(y_test, y_pred_test, 
                                   target_names=['DOWN', 'UP'],
                                   zero_division=0))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"\n🏆 Top 15 Most Important Features:")
        for idx, row in feature_importance.head(15).iterrows():
            logger.info(f"   {row['feature']:30s}: {row['importance']:.4f}")
        
        # Збереження моделі
        model_dir = Path(f'models/simple_trend_{self.symbol}')
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / f'model_{self.symbol}_{self.timeframe}.pkl'
        scaler_path = model_dir / f'scaler_{self.symbol}_{self.timeframe}.pkl'
        features_path = model_dir / f'features_{self.symbol}_{self.timeframe}.pkl'
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(feature_names, features_path)
        
        logger.info(f"\n💾 Модель збережено: {model_path}")
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"⏱️ Час тренування: {duration:.1f}s ({duration/60:.1f}m)")
        
        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'n_features': len(feature_names),
            'duration': duration,
            'model_path': str(model_path),
            'feature_importance': feature_importance.to_dict('records')
        }


async def main():
    """Тестування на BTCUSDT"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default='BTCUSDT', help='Trading symbol')
    parser.add_argument('--timeframe', default='1d', help='Timeframe (4h, 1d)')
    parser.add_argument('--days', type=int, default=730, help='Days of history')
    args = parser.parse_args()
    
    trainer = SimpleTrendClassifier(
        symbol=args.symbol,
        timeframe=args.timeframe
    )
    
    results = await trainer.train(days=args.days)
    
    if results['test_accuracy'] >= 0.70:
        logger.info(f"\n🎉 SUCCESS! Accuracy {results['test_accuracy']*100:.2f}% >= 70%")
    else:
        logger.warning(f"\n⚠️ Accuracy {results['test_accuracy']*100:.2f}% < 70% - потрібно покращити")


if __name__ == '__main__':
    asyncio.run(main())
