"""
Custom Keras Callbacks

Callback'и для тренування моделі:
- DatabaseHistoryCallback: збереження історії в БД
- DenormalizedMetricsCallback: вивід реальних метрик
"""

import logging
import numpy as np
from tensorflow import keras
from keras import callbacks

logger = logging.getLogger(__name__)


class DatabaseHistoryCallback(callbacks.Callback):
    """Callback для збереження історії тренування в базу даних"""
    
    def __init__(self, db_engine, symbol_id: int, interval_id: int, fold: int = 1):
        """
        Ініціалізація
        
        Args:
            db_engine: SQLAlchemy engine
            symbol_id: ID символу в БД
            interval_id: ID інтервалу в БД
            fold: Номер fold (для cross-validation)
        """
        super().__init__()
        self.db_engine = db_engine
        self.symbol_id = symbol_id
        self.interval_id = interval_id
        self.fold = fold
    
    def on_epoch_end(self, epoch, logs=None):
        """Збереження метрик епохи в БД"""
        if logs is None:
            return
        
        try:
            from sqlalchemy import text
            with self.db_engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO training_history 
                    (symbol_id, interval_id, fold, epoch, loss, mae, val_loss, val_mae, 
                     mape, val_mape, directional_accuracy, real_mae, real_mape)
                    VALUES (:symbol_id, :interval_id, :fold, :epoch, :loss, :mae, 
                            :val_loss, :val_mae, :mape, :val_mape, :directional_accuracy, 
                            :real_mae, :real_mape)
                    ON CONFLICT (symbol_id, interval_id, fold, epoch)
                    DO UPDATE SET
                        loss = EXCLUDED.loss,
                        mae = EXCLUDED.mae,
                        val_loss = EXCLUDED.val_loss,
                        val_mae = EXCLUDED.val_mae,
                        mape = EXCLUDED.mape,
                        val_mape = EXCLUDED.val_mape,
                        directional_accuracy = EXCLUDED.directional_accuracy,
                        real_mae = EXCLUDED.real_mae,
                        real_mape = EXCLUDED.real_mape
                """), {
                    'symbol_id': self.symbol_id,
                    'interval_id': self.interval_id,
                    'fold': self.fold,
                    'epoch': epoch + 1,  # epoch починається з 0, але зберігаємо з 1
                    'loss': float(logs.get('loss', 0)),
                    'mae': float(logs.get('mae', 0)),
                    'val_loss': float(logs.get('val_loss', 0)),
                    'val_mae': float(logs.get('val_mae', 0)),
                    'mape': float(logs.get('mape', 0)),
                    'val_mape': float(logs.get('val_mape', 0)),
                    'directional_accuracy': float(logs.get('directional_accuracy', 0)),
                    'real_mae': float(logs.get('real_mae', logs.get('mae', 0))),
                    'real_mape': float(logs.get('real_mape', logs.get('mape', 0)))
                })
                conn.commit()
        except Exception as e:
            logger.error(f"❌ Помилка збереження в БД: {e}")


class DenormalizedMetricsCallback(callbacks.Callback):
    """
    Callback для виводу спрощених метрик
    
    Розраховує та виводить метрики як у відсоткових змінах, так і в реальних цінах
    """
    
    def __init__(self, scaler=None, feature_index: int = None, X_val: np.ndarray = None, y_val: np.ndarray = None):
        """
        Ініціалізація
        
        Args:
            scaler: Scaler для денормалізації
            feature_index: Індекс фічі ціни close
            X_val: Validation X data
            y_val: Validation y data
        """
        super().__init__()
        self.scaler = scaler
        self.feature_index = feature_index
        self.X_val = X_val
        self.y_val = y_val
    
    def on_epoch_end(self, epoch, logs=None):
        """Вивід метрик після епохи"""
        if logs is None:
            return
        
        try:
            # Отримуємо передбачення на валідаційних даних
            if self.X_val is not None and self.y_val is not None and self.scaler is not None:
                y_pred = self.model.predict(self.X_val, verbose=0).flatten()
                y_true = self.y_val  # відсоткові зміни
                
                # Розраховуємо метрики на рівні відсоткових змін
                mae_norm = np.mean(np.abs(y_true - y_pred))
                mape_norm = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-6))) * 100
                
                # Directional accuracy: знак відсоткових змін
                dir_acc = np.mean(np.sign(y_true) == np.sign(y_pred)) * 100
                
                logger.info(f"💰 Метрики відсоткових змін (Epoch {epoch + 1}): "
                           f"MAE={mae_norm:.4f}, MAPE={mape_norm:.2f}%, Напрямок={dir_acc:.1f}%")
                
                # Реальні метрики: конвертуємо назад до абсолютних цін
                current_prices = self.X_val[:, -1, self.feature_index].flatten()
                
                y_true_pct = y_true  # відсоткові зміни
                y_pred_pct = y_pred  # відсоткові зміни
                
                true_next_prices = current_prices * (1 + y_true_pct / 100)
                pred_next_prices = current_prices * (1 + y_pred_pct / 100)
                
                mae_real = np.mean(np.abs(true_next_prices - pred_next_prices))
                mape_real = np.mean(np.abs((true_next_prices - pred_next_prices) / (np.abs(true_next_prices) + 1e-6))) * 100
                
                logger.info(f"💵 Реальні метрики цін (Epoch {epoch + 1}): "
                           f"MAE={mae_real:.2f}, MAPE={mape_real:.2f}%, Напрямок={dir_acc:.1f}%")
                
                # Додаємо метрики до logs для збереження в БД
                logs['mape'] = mape_real
                logs['val_mape'] = mape_real
                logs['directional_accuracy'] = dir_acc
                logs['real_mae'] = mae_real
                logs['real_mape'] = mape_real
            
        except Exception as e:
            logger.error(f"❌ Помилка розрахунку метрик: {e}")


__all__ = [
    'DatabaseHistoryCallback',
    'DenormalizedMetricsCallback',
]
