"""
Трендова стратегія на 4h таймфреймі
Використовує Random Forest для визначення трендів
"""
import asyncio
import logging
from pathlib import Path
from typing import Dict, Optional, List
import pandas as pd
import joblib
from datetime import datetime

from .base_strategy import BaseStrategy, Signal
from training.simple_trend_classifier import SimpleTrendClassifier

logger = logging.getLogger(__name__)


class TrendStrategy4h(BaseStrategy):
    """4h трендова стратегія з Random Forest"""
    
    def __init__(
        self,
        symbols: List[str],
        testnet: bool = True,
        min_confidence: float = 0.70,
        risk_per_trade: float = 0.01
    ):
        super().__init__(
            name="TrendStrategy4h",
            timeframe="4h",
            symbols=symbols,
            min_confidence=min_confidence,
            risk_per_trade=risk_per_trade,
            stop_loss_pct=0.02,
            take_profit_pct=0.05
        )
        self.testnet = testnet
        # Feature calculator буде створюватись для кожного символу окремо
        self.feature_calculators = {}
        
    async def load_models(self):
        """Завантаження моделей для всіх символів"""
        models_dir = Path('models')
        loaded = 0
        
        for symbol in self.symbols:
            try:
                model_dir = models_dir / f'simple_trend_{symbol}'
                
                if not model_dir.exists():
                    logger.warning(f"⚠️ Модель для {symbol} не знайдена: {model_dir}")
                    continue
                
                # Знаходимо файли моделі
                pkl_files = list(model_dir.glob(f'model_{symbol}_*.pkl'))
                if not pkl_files:
                    logger.warning(f"⚠️ Не знайдено файлів моделі для {symbol}")
                    continue
                
                model_path = str(pkl_files[0])
                timeframe = pkl_files[0].stem.split('_')[-1]
                scaler_path = str(model_dir / f'scaler_{symbol}_{timeframe}.pkl')
                features_path = str(model_dir / f'features_{symbol}_{timeframe}.pkl')
                
                # Завантаження
                self.models[symbol] = joblib.load(model_path)
                self.scalers[symbol] = joblib.load(scaler_path)
                self.feature_names_dict[symbol] = joblib.load(features_path)
                
                loaded += 1
                logger.info(f"✅ {symbol}: модель завантажено ({len(self.feature_names_dict[symbol])} features)")
                
            except Exception as e:
                logger.error(f"❌ Помилка завантаження моделі {symbol}: {e}")
                continue
        
        logger.info(f"✅ Завантажено моделей: {loaded}/{len(self.symbols)}")
        
    async def generate_signal(self, symbol: str, df: pd.DataFrame) -> Optional[Signal]:
        """Генерація сигналу на основі ML прогнозу"""
        try:
            if symbol not in self.models:
                return None
            
            # Створюємо feature calculator для символу якщо потрібно
            if symbol not in self.feature_calculators:
                self.feature_calculators[symbol] = SimpleTrendClassifier(symbol=symbol, timeframe='4h')
            
            # Розрахунок features
            df_features = self.feature_calculators[symbol]._create_simple_features(df)
            
            if df_features.empty or len(df_features) == 0:
                logger.warning(f"⚠️ {symbol}: немає features")
                return None
            
            # Перевірка features
            expected_features = self.feature_names_dict[symbol]
            missing = set(expected_features) - set(df_features.columns)
            
            if missing:
                logger.error(f"❌ {symbol}: відсутні features: {missing}")
                return None
            
            # Prediction
            X = df_features[expected_features].iloc[-1:].values
            X_scaled = self.scalers[symbol].transform(X)
            
            prediction = self.models[symbol].predict(X_scaled)[0]
            probas = self.models[symbol].predict_proba(X_scaled)[0]
            
            proba_down = probas[0]
            proba_up = probas[1]
            confidence = max(proba_down, proba_up)
            
            current_price = df['close'].iloc[-1]
            direction = 'UP' if prediction == 1 else 'DOWN'
            
            # Визначення action
            if confidence >= self.min_confidence:
                action = 'BUY' if direction == 'UP' else 'SELL'
            else:
                action = 'HOLD'
            
            signal = Signal(
                symbol=symbol,
                strategy=self.name,
                action=action,
                confidence=confidence,
                price=current_price,
                timestamp=datetime.now(),
                metadata={
                    'direction': direction,
                    'proba_up': proba_up,
                    'proba_down': proba_down,
                    'timeframe': self.timeframe
                }
            )
            
            logger.info(f"🤖 {symbol}: {direction} (confidence: {confidence:.2%}, price: ${current_price:.2f})")
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Помилка прогнозу {symbol}: {e}", exc_info=True)
            return None
    
    async def should_close(self, symbol: str, position: Dict, current_price: float) -> tuple[bool, str]:
        """Перевірка SL/TP"""
        try:
            entry_price = position['entry_price']
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
            
            # Stop-loss
            if pnl_pct <= -self.stop_loss_pct * 100:
                return True, "SL"
            
            # Take-profit
            if pnl_pct >= self.take_profit_pct * 100:
                return True, "TP"
            
            return False, ""
            
        except Exception as e:
            logger.error(f"❌ Помилка перевірки закриття {symbol}: {e}")
            return False, ""
