#!/usr/bin/env python3
"""
LSTM Model Synchronization Test & Validation
Verifies all components are properly synchronized with new LSTM training system
"""
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class SyncValidator:
    """Validates synchronization of all LSTM components"""
    
    def __init__(self):
        self.results = {}
        self.errors = []
    
    def check_model_files(self) -> bool:
        """Check if trained LSTM model files exist"""
        logger.info("🔍 Checking LSTM model files...")
        
        model_path = Path("models/lstm_model.pt")
        scaler_path = Path("models/lstm_scaler.pkl")
        metadata_path = Path("models/lstm_metadata.json")
        
        checks = {
            'model': model_path.exists(),
            'scaler': scaler_path.exists(),
            'metadata': metadata_path.exists()
        }
        
        for name, exists in checks.items():
            status = "✅" if exists else "❌"
            logger.info(f"  {status} {name}: {Path(name) / 'models'}")
        
        self.results['model_files'] = all(checks.values())
        return self.results['model_files']
    
    def check_predictor_imports(self) -> bool:
        """Check if LSTM predictor can be imported"""
        logger.info("🔍 Checking LSTM predictor imports...")
        
        try:
            from ml_predictor_lstm import LSTMPricePredictor
            logger.info("  ✅ LSTMPricePredictor imported successfully")
            self.results['predictor_import'] = True
            return True
        except ImportError as e:
            logger.error(f"  ❌ Failed to import LSTMPricePredictor: {e}")
            self.errors.append(f"Predictor import: {e}")
            self.results['predictor_import'] = False
            return False
    
    def check_ensemble_imports(self) -> bool:
        """Check if LSTM ensemble can be imported"""
        logger.info("🔍 Checking LSTM ensemble imports...")
        
        try:
            from ensemble_predictor_lstm import LSTMEnsemblePredictor
            logger.info("  ✅ LSTMEnsemblePredictor imported successfully")
            self.results['ensemble_import'] = True
            return True
        except ImportError as e:
            logger.error(f"  ❌ Failed to import LSTMEnsemblePredictor: {e}")
            self.errors.append(f"Ensemble import: {e}")
            self.results['ensemble_import'] = False
            return False
    
    def check_model_loading(self) -> bool:
        """Test if LSTM model can be loaded"""
        logger.info("🔍 Checking LSTM model loading...")
        
        try:
            from ml_predictor_lstm import LSTMPricePredictor
            predictor = LSTMPricePredictor("models/lstm_model.pt")
            
            if predictor.is_ready():
                logger.info("  ✅ LSTM model loaded successfully")
                info = predictor.get_model_info()
                logger.info(f"     Device: {info.get('device')}")
                logger.info(f"     Sequence length: {info.get('sequence_length')}")
                logger.info(f"     Features: {info.get('feature_count')}")
                self.results['model_loading'] = True
                return True
            else:
                logger.error("  ❌ LSTM model loaded but not ready")
                self.results['model_loading'] = False
                return False
                
        except Exception as e:
            logger.error(f"  ❌ Failed to load LSTM model: {e}")
            self.errors.append(f"Model loading: {e}")
            self.results['model_loading'] = False
            return False
    
    def check_strategy_compatibility(self) -> bool:
        """Check if strategies are compatible with new system"""
        logger.info("🔍 Checking strategy compatibility...")
        
        try:
            from strategies.base import BaseStrategy, Signal
            from strategies.mean_reversion import MeanReversionStrategy
            from strategies.trend_following import TrendFollowingStrategy
            
            logger.info("  ✅ All strategies imported successfully")
            
            # Verify signal structure
            test_signal = Signal(
                direction='LONG',
                confidence=75.0,
                entry_price=100.0,
                stop_loss=99.0,
                take_profit=101.5,
                reason="Test signal"
            )
            
            if test_signal.is_valid():
                logger.info("  ✅ Signal structure is valid")
                self.results['strategy_compatibility'] = True
                return True
            else:
                logger.error("  ❌ Signal validation failed")
                self.results['strategy_compatibility'] = False
                return False
                
        except Exception as e:
            logger.error(f"  ❌ Strategy compatibility check failed: {e}")
            self.errors.append(f"Strategy compatibility: {e}")
            self.results['strategy_compatibility'] = False
            return False
    
    def check_visualizer_compatibility(self) -> bool:
        """Check if visualizer is compatible"""
        logger.info("🔍 Checking visualizer compatibility...")
        
        try:
            from pattern_visualizer_debug import PatternVisualizer
            visualizer = PatternVisualizer()
            logger.info("  ✅ PatternVisualizer imported successfully")
            self.results['visualizer_compatibility'] = True
            return True
        except Exception as e:
            logger.error(f"  ❌ Visualizer compatibility check failed: {e}")
            self.errors.append(f"Visualizer compatibility: {e}")
            self.results['visualizer_compatibility'] = False
            return False
    
    def check_integration_module(self) -> bool:
        """Check if LSTM integration module is available"""
        logger.info("🔍 Checking LSTM integration module...")
        
        try:
            from lstm_model_integration import LSTMModelManager, get_lstm_manager
            logger.info("  ✅ Integration module imported successfully")
            self.results['integration_module'] = True
            return True
        except Exception as e:
            logger.error(f"  ❌ Integration module check failed: {e}")
            self.errors.append(f"Integration module: {e}")
            self.results['integration_module'] = False
            return False
    
    def run_full_validation(self) -> bool:
        """Run complete validation suite"""
        logger.info("\n" + "="*60)
        logger.info("LSTM SYNCHRONIZATION VALIDATION")
        logger.info("="*60 + "\n")
        
        # Run all checks
        checks = [
            ("Model files", self.check_model_files),
            ("Predictor imports", self.check_predictor_imports),
            ("Ensemble imports", self.check_ensemble_imports),
            ("Model loading", self.check_model_loading),
            ("Strategy compatibility", self.check_strategy_compatibility),
            ("Visualizer compatibility", self.check_visualizer_compatibility),
            ("Integration module", self.check_integration_module),
        ]
        
        all_passed = True
        for name, check_func in checks:
            try:
                if not check_func():
                    all_passed = False
            except Exception as e:
                logger.error(f"❌ {name} check crashed: {e}")
                all_passed = False
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("VALIDATION SUMMARY")
        logger.info("="*60)
        
        for check_name, result in self.results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{status}: {check_name}")
        
        if self.errors:
            logger.info("\nErrors encountered:")
            for error in self.errors:
                logger.info(f"  - {error}")
        
        logger.info("="*60 + "\n")
        
        if all_passed:
            logger.info("✅ ALL CHECKS PASSED - System is synchronized!")
        else:
            logger.error("❌ SOME CHECKS FAILED - See errors above")
        
        return all_passed


if __name__ == "__main__":
    validator = SyncValidator()
    success = validator.run_full_validation()
    sys.exit(0 if success else 1)
