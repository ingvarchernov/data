"""
ML Predictor - прогнозування з Random Forest моделями
"""
import asyncio
from pathlib import Path
from datetime import datetime
import joblib
import pandas as pd
import numpy as np
import logging

from training.rust_features import RustFeatureEngineer
from mtf_analyzer import MultiTimeframeAnalyzer

logger = logging.getLogger(__name__)


class MLPredictor:
    """ML прогнози для торгівлі"""
    
    def __init__(self, symbols: list, use_mtf: bool = True):
        self.symbols = symbols
        self.use_mtf = use_mtf
        
        # ML компоненти
        self.models = {}
        self.scalers = {}
        self.feature_names_dict = {}
        self.feature_engineer = RustFeatureEngineer()
        
        # MTF аналізатор
        self.mtf_analyzer = MultiTimeframeAnalyzer()
    
    def load_models(self):
        """Завантаження ML моделей для всіх символів"""
        for symbol in self.symbols:
            try:
                model_dir = Path(f'models/simple_trend_{symbol}')
                if not model_dir.exists():
                    logger.warning(f"⚠️ Модель для {symbol} не знайдено")
                    continue
                
                pkl_files = list(model_dir.glob('model_*.pkl'))
                if not pkl_files:
                    logger.warning(f"⚠️ Файли моделі для {symbol} не знайдено")
                    continue
                
                model_path = str(pkl_files[0])
                timeframe = pkl_files[0].stem.split('_')[-1]
                scaler_path = str(model_dir / f'scaler_{symbol}_{timeframe}.pkl')
                features_path = str(model_dir / f'features_{symbol}_{timeframe}.pkl')
                
                self.models[symbol] = joblib.load(model_path)
                self.scalers[symbol] = joblib.load(scaler_path)
                self.feature_names_dict[symbol] = joblib.load(features_path)
                
                logger.info(f"✅ {symbol}: модель завантажено ({len(self.feature_names_dict[symbol])} features)")
            
            except Exception as e:
                logger.error(f"❌ Помилка завантаження моделі {symbol}: {e}")
        
        if not self.models:
            raise RuntimeError("❌ Жодна модель не завантажена!")
        
        logger.info(f"✅ Завантажено моделей: {len(self.models)}/{len(self.symbols)}")
    
    async def predict(self, symbol: str, df: pd.DataFrame) -> dict:
        """Прогноз напрямку руху для одного символу"""
        try:
            if symbol not in self.models:
                logger.warning(f"⚠️ Модель для {symbol} не завантажена")
                return None
            
            # Розрахунок features через Rust
            df_features = self.feature_engineer.calculate_all(
                df,
                sma_periods=[5, 10, 20, 50, 100, 200],
                ema_periods=[9, 12, 21, 26, 50],
                rsi_periods=[7, 14, 21, 28],
                atr_periods=[14, 21],
            )
            
            # Додаткові features
            for period in [5, 10, 20, 50, 100, 200]:
                ma_col = f'sma_{period}'
                if ma_col in df_features.columns:
                    df_features[f'price_vs_sma{period}'] = (df['close'] / df_features[ma_col] - 1) * 100
            
            # MA crossovers
            if 'sma_50' in df_features.columns and 'sma_200' in df_features.columns:
                df_features['golden_cross'] = (df_features['sma_50'] > df_features['sma_200']).astype(int)
            
            if 'ema_12' in df_features.columns and 'ema_26' in df_features.columns:
                df_features['macd_cross'] = (df_features['ema_12'] > df_features['ema_26']).astype(int)
            
            # Volume features
            if 'volume' in df.columns:
                df_features['volume_sma20'] = df['volume'].rolling(20).mean()
                df_features['volume_trend'] = df['volume'] / df_features['volume_sma20']
            
            # RSI levels
            if 'rsi_14' in df_features.columns:
                df_features['rsi_overbought'] = (df_features['rsi_14'] > 70).astype(int)
                df_features['rsi_oversold'] = (df_features['rsi_14'] < 30).astype(int)
            
            # Momentum
            df_features['momentum_5'] = df['close'].pct_change(5) * 100
            df_features['momentum_10'] = df['close'].pct_change(10) * 100
            df_features['momentum_20'] = df['close'].pct_change(20) * 100
            
            # Volatility
            df_features['volatility_20'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean() * 100
            
            # Очистка NaN
            df_features = df_features.dropna()
            
            if len(df_features) < 10:
                logger.warning("⚠️ Недостатньо даних для прогнозу")
                return None
            
            # Використовуємо тільки потрібні features
            feature_names = self.feature_names_dict[symbol]
            missing_features = [f for f in feature_names if f not in df_features.columns]
            if missing_features:
                logger.warning(f"⚠️ Відсутні features для {symbol}: {missing_features[:5]}...")
                return None
            
            # Останній рядок з потрібними features
            X = df_features[feature_names].iloc[-1:].values
            
            # Скалювання
            X_scaled = self.scalers[symbol].transform(X)
            
            # Прогноз
            prediction = self.models[symbol].predict(X_scaled)[0]
            proba = self.models[symbol].predict_proba(X_scaled)[0]
            
            current_price = df['close'].iloc[-1]
            
            result = {
                'symbol': symbol,
                'prediction': 'UP' if prediction == 1 else 'DOWN',
                'confidence': max(proba),
                'proba_down': proba[0],
                'proba_up': proba[1] if len(proba) > 1 else 0,
                'current_price': current_price,
                'timestamp': datetime.now()
            }
            
            logger.info(
                f"🤖 {symbol}: {result['prediction']} "
                f"(confidence: {result['confidence']:.2%}, price: ${current_price:.2f})"
            )
            
            return result
        
        except Exception as e:
            logger.error(f"❌ Помилка прогнозу {symbol}: {e}", exc_info=True)
            return None
    
    async def predict_mtf(self, symbol: str, get_market_data_func) -> dict:
        """
        Multi-timeframe прогноз
        Аналізує 4h, 1h, 15m і комбінує сигнали
        
        Args:
            get_market_data_func: async функція для завантаження даних
        """
        try:
            if not self.use_mtf:
                # Якщо MTF вимкнено, використовуємо стандартний прогноз
                df = await get_market_data_func(symbol, interval='4h')
                return await self.predict(symbol, df)
            
            # Завантаження даних для всіх таймфреймів
            timeframes = ['4h', '1h', '15m']
            predictions = {}
            
            for tf in timeframes:
                df = await get_market_data_func(symbol, interval=tf, limit=1000)
                
                if df.empty:
                    logger.warning(f"⚠️ {symbol} {tf}: немає даних")
                    continue
                
                pred = await self.predict(symbol, df)
                if pred:
                    predictions[tf] = pred
            
            # Перевірка наявності всіх прогнозів
            if len(predictions) < 3:
                logger.warning(f"⚠️ {symbol}: недостатньо MTF даних ({len(predictions)}/3)")
                # Fallback на 4h якщо є
                return predictions.get('4h')
            
            # Комбінування через MTF аналізатор
            mtf_result = self.mtf_analyzer.analyze(predictions, require_alignment=True)
            
            if not mtf_result:
                logger.info(f"⚠️ {symbol}: MTF не дав чіткого сигналу")
                return None
            
            return mtf_result
        
        except Exception as e:
            logger.error(f"❌ Помилка MTF прогнозу {symbol}: {e}", exc_info=True)
            return None
