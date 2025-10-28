"""
Свінг-стратегія на 1h таймфреймі
Для середньострокових угод
"""
import logging
from pathlib import Path
from typing import Dict, Optional, List
import pandas as pd
import joblib
from datetime import datetime

from .base_strategy import BaseStrategy, Signal
from training.rust_features import RustFeatureEngineer

logger = logging.getLogger(__name__)


class SwingStrategy1h(BaseStrategy):
    """1h свінг-стратегія з Random Forest"""
    
    def __init__(
        self,
        symbols: List[str],
        testnet: bool = True,
        min_confidence: float = 0.65,
        risk_per_trade: float = 0.005  # Менший ризик для коротших таймфреймів
    ):
        super().__init__(
            name="SwingStrategy1h",
            timeframe="1h",
            symbols=symbols,
            min_confidence=min_confidence,
            risk_per_trade=risk_per_trade,
            stop_loss_pct=0.015,  # 1.5% SL
            take_profit_pct=0.03  # 3% TP
        )
        self.testnet = testnet
        self.feature_engineer = RustFeatureEngineer()
        
    async def load_models(self):
        """Завантаження моделей для 1h таймфрейму"""
        models_dir = Path('models')
        loaded = 0
        
        for symbol in self.symbols:
            try:
                # Шукаємо моделі з 1h в назві
                model_dir = models_dir / f'swing_1h_{symbol}'
                
                if not model_dir.exists():
                    logger.warning(f"⚠️ 1h модель для {symbol} не знайдена (треба натренувати)")
                    continue
                
                pkl_files = list(model_dir.glob(f'model_{symbol}_1h.pkl'))
                if not pkl_files:
                    logger.warning(f"⚠️ Не знайдено 1h моделі для {symbol}")
                    continue
                
                model_path = str(pkl_files[0])
                scaler_path = str(model_dir / f'scaler_{symbol}_1h.pkl')
                features_path = str(model_dir / f'features_{symbol}_1h.pkl')
                
                # Завантаження
                self.models[symbol] = joblib.load(model_path)
                self.scalers[symbol] = joblib.load(scaler_path)
                self.feature_names_dict[symbol] = joblib.load(features_path)
                
                loaded += 1
                logger.info(f"✅ {symbol} [1h]: модель завантажено ({len(self.feature_names_dict[symbol])} features)")
                
            except Exception as e:
                logger.error(f"❌ Помилка завантаження 1h моделі {symbol}: {e}")
                continue
        
        if loaded == 0:
            logger.warning("⚠️ Жодної 1h моделі не завантажено - треба натренувати!")
        else:
            logger.info(f"✅ Завантажено 1h моделей: {loaded}/{len(self.symbols)}")
        
    async def generate_signal(self, symbol: str, df: pd.DataFrame) -> Optional[Signal]:
        """Генерація свінг-сигналу"""
        try:
            if symbol not in self.models:
                return None
            
            # Розрахунок features
            df_features = self.feature_engineer.calculate_all(df)
            
            if df_features.empty or len(df_features) == 0:
                return None
            
            # Перевірка features
            expected_features = self.feature_names_dict[symbol]
            missing = set(expected_features) - set(df_features.columns)
            
            if missing:
                logger.error(f"❌ {symbol} [1h]: відсутні features: {missing}")
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
            
            logger.info(f"🔄 {symbol} [1h]: {direction} (confidence: {confidence:.2%}, price: ${current_price:.2f})")
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Помилка 1h прогнозу {symbol}: {e}", exc_info=True)
            return None
    
    async def should_close(self, symbol: str, position: Dict, current_price: float) -> tuple[bool, str]:
        """Перевірка SL/TP для 1h стратегії"""
        try:
            entry_price = position['entry_price']
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
            
            # Більш тугий SL/TP для 1h
            if pnl_pct <= -self.stop_loss_pct * 100:
                return True, "SL"
            
            if pnl_pct >= self.take_profit_pct * 100:
                return True, "TP"
            
            return False, ""
            
        except Exception as e:
            logger.error(f"❌ Помилка перевірки закриття {symbol} [1h]: {e}")
            return False, ""
