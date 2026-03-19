#!/usr/bin/env python3
"""
LSTM Model Integration & Synchronization
Bridges the gap between new LSTM training system and existing strategy/prediction modules
Provides unified interface for model management and predictions
"""
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd

from config import LSTM_MODEL_CONFIG

logger = logging.getLogger(__name__)


class LSTMModelManager:
    """Central manager for LSTM model operations and synchronization"""
    
    def __init__(self, model_path: Optional[str] = None):
        resolved_model_path = model_path or LSTM_MODEL_CONFIG.get('model_path', 'models/lstm_model.pt')
        self.model_path = Path(resolved_model_path)
        self.predictor = None
        self.ensemble = None
        self.initialized = False
        
    def initialize(self) -> bool:
        """Initialize LSTM predictor and ensemble"""
        try:
            # Try to import and initialize LSTM predictor
            try:
                from ml_predictor_lstm import LSTMPricePredictor
                self.predictor = LSTMPricePredictor(str(self.model_path))
                logger.info("✅ LSTMPricePredictor initialized")
            except ImportError:
                logger.warning("⚠️ Could not import LSTMPricePredictor from ml_predictor_lstm")
                return False
            
            # Initialize ensemble
            try:
                from ensemble_predictor_lstm import LSTMEnsemblePredictor
                self.ensemble = LSTMEnsemblePredictor(str(self.model_path))
                logger.info("✅ LSTMEnsemblePredictor initialized")
            except ImportError:
                logger.warning("⚠️ Could not import LSTMEnsemblePredictor")
            
            # Check if model is loaded
            if self.predictor and hasattr(self.predictor, 'is_ready'):
                if self.predictor.is_ready():
                    self.initialized = True
                    logger.info("✅ LSTM model ready for predictions")
                    return True
                else:
                    logger.error("❌ LSTM model not ready after initialization")
                    return False
            else:
                logger.error("❌ Predictor does not have is_ready method")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize LSTM models: {e}")
            return False
    
    def predict_lstm(self, df: pd.DataFrame) -> Optional[Dict]:
        """Get LSTM price prediction"""
        if not self.initialized or not self.predictor:
            return None
        
        try:
            return self.predictor.predict(df)
        except Exception as e:
            logger.error(f"LSTM prediction error: {e}")
            return None
    
    def predict_ensemble(
        self,
        df: pd.DataFrame,
        pattern_confidence: float,
        pattern_direction: str
    ) -> Optional[Dict]:
        """Get ensemble prediction combining LSTM and other signals"""
        if not self.ensemble:
            return None
        
        try:
            return self.ensemble.predict_ensemble(df, pattern_confidence, pattern_direction)
        except Exception as e:
            logger.error(f"Ensemble prediction error: {e}")
            return None
    
    def get_model_info(self) -> Dict:
        """Get information about loaded LSTM model"""
        info = {
            'initialized': self.initialized,
            'model_path': str(self.model_path),
            'model_exists': self.model_path.exists(),
            'predictor_status': 'ready' if self.predictor and hasattr(self.predictor, 'is_ready') and self.predictor.is_ready() else 'not_ready',
            'ensemble_status': 'loaded' if self.ensemble else 'not_loaded'
        }
        
        if self.predictor and hasattr(self.predictor, 'get_model_info'):
            info['predictor_info'] = self.predictor.get_model_info()
        
        if self.ensemble and hasattr(self.ensemble, 'get_ensemble_info'):
            info['ensemble_info'] = self.ensemble.get_ensemble_info()
        
        return info


# Global singleton instance
_lstm_manager: Optional[LSTMModelManager] = None


def get_lstm_manager(model_path: Optional[str] = None) -> LSTMModelManager:
    """Get or create global LSTM model manager"""
    global _lstm_manager
    if _lstm_manager is None:
        _lstm_manager = LSTMModelManager(model_path)
    return _lstm_manager


async def initialize_lstm_models(model_path: Optional[str] = None) -> bool:
    """Initialize LSTM models asynchronously"""
    manager = get_lstm_manager(model_path)
    return manager.initialize()


# Backward compatibility functions
def load_lstm_predictor(model_path: Optional[str] = None):
    """Load LSTM predictor (backward compatible)"""
    try:
        from ml_predictor_lstm import LSTMPricePredictor
        return LSTMPricePredictor(model_path or LSTM_MODEL_CONFIG.get('model_path', 'models/lstm_model.pt'))
    except ImportError:
        logger.error("Could not import LSTMPricePredictor")
        return None


def load_ensemble_predictor(model_path: Optional[str] = None):
    """Load ensemble predictor (backward compatible)"""
    try:
        from ensemble_predictor_lstm import LSTMEnsemblePredictor
        return LSTMEnsemblePredictor(model_path or LSTM_MODEL_CONFIG.get('model_path', 'models/lstm_model.pt'))
    except ImportError:
        logger.error("Could not import LSTMEnsemblePredictor")
        return None


if __name__ == "__main__":
    # Test initialization
    logging.basicConfig(level=logging.INFO)
    
    manager = get_lstm_manager()
    print(f"Initialization result: {manager.initialize()}")
    print(f"Model info: {manager.get_model_info()}")
