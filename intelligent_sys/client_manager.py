"""
Управління API клієнтами (Binance, CCXT)
"""
import logging
import os
from typing import Optional
from dotenv import load_dotenv

from .base import DataSource, LoaderConfig

# Спроба імпорту бібліотек
try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException, BinanceRequestException
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False
    Client = None
    BinanceAPIException = Exception
    BinanceRequestException = Exception

try:
    import ccxt.async_support as ccxt_async
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    ccxt_async = None

logger = logging.getLogger(__name__)
load_dotenv()


class ClientManager:
    """
    Менеджер API клієнтів
    """
    
    def __init__(self, config: LoaderConfig):
        self.config = config
        self.binance_client = None
        self.ccxt_client = None
        
        # Визначення джерела даних
        self._determine_data_source()
        
        # Отримання API ключів
        self._setup_credentials()
        
        # Ініціалізація клієнтів
        self._initialize_clients()

    @staticmethod
    def _sanitize_credential(value: Optional[str]) -> Optional[str]:
        """Прибирає зайві пробіли з облікових даних."""
        if value is None:
            return None
        return value.strip()

    @staticmethod
    def _mask_credential(value: Optional[str]) -> str:
        if not value:
            return "<відсутній>"
        if len(value) <= 8:
            return f"{value[0]}***{value[-1]}"
        return f"{value[:4]}***{value[-4:]}"
    
    def _determine_data_source(self):
        """Визначення джерела даних"""
        if self.config.data_source == DataSource.AUTO:
            if self.config.testnet and BINANCE_AVAILABLE:
                # CCXT не підтримує testnet для більшості endpoint'ів Binance
                self.config.data_source = DataSource.PYTHON_BINANCE
                logger.info("📡 Обрано python-binance (testnet режим)")
            elif CCXT_AVAILABLE:
                self.config.data_source = DataSource.CCXT
                logger.info("📡 Автоматично обрано CCXT")
            elif BINANCE_AVAILABLE:
                self.config.data_source = DataSource.PYTHON_BINANCE
                logger.info("📡 Автоматично обрано python-binance")
            else:
                raise ImportError(
                    "Потрібно встановити python-binance або ccxt:\n"
                    "  pip install python-binance\n"
                    "  або\n"
                    "  pip install ccxt"
                )
    
    def _setup_credentials(self):
        """Налаштування облікових даних"""
        if not self.config.use_public_data:
            # Пріоритет FUTURES_API_KEY / FUTURES_API_SECRET для всіх операцій
            self.api_key = (
                self.config.api_key or
                os.getenv('FUTURES_API_KEY') or
                os.getenv('BINANCE_TEST_API_KEY' if self.config.testnet else 'API_KEY')
            )
            self.api_secret = (
                self.config.api_secret or
                os.getenv('FUTURES_API_SECRET') or
                os.getenv('BINANCE_TEST_API_SECRET' if self.config.testnet else 'API_SECRET')
            )

            self.api_key = self._sanitize_credential(self.api_key)
            self.api_secret = self._sanitize_credential(self.api_secret)

            if not self.api_key or not self.api_secret:
                logger.warning(
                    "🔓 Відсутні API ключі для автентифікованого доступу – переносимося на публічні дані"
                )
                self.config.use_public_data = True

        if self.config.use_public_data:
            self.api_key = "public"
            self.api_secret = "public"

        logger.info(
            "🔑 ClientManager key (masked): %s",
            self._mask_credential(self.api_key if self.api_key != "public" else None)
        )
    
    def _initialize_clients(self):
        """Ініціалізація API клієнтів"""
        try:
            # Скидаємо попередні інスタції перед переініціалізацією
            self.binance_client = None
            self.ccxt_client = None

            if self.config.data_source == DataSource.PYTHON_BINANCE:
                self._init_binance_client()
            elif self.config.data_source == DataSource.CCXT:
                self._init_ccxt_client()
            
            logger.info(
                f"✅ ClientManager ініціалізовано: "
                f"source={self.config.data_source.value}, "
                f"testnet={self.config.testnet}"
            )
        
        except Exception as e:
            logger.error(f"❌ Помилка ініціалізації клієнта: {e}")
            raise
    
    def _init_binance_client(self):
        """Ініціалізація python-binance клієнта"""
        if not BINANCE_AVAILABLE:
            raise ImportError("python-binance не встановлено")
        
        if self.config.use_public_data:
            self.binance_client = Client("", "", testnet=self.config.testnet)
        else:
            self.binance_client = Client(
                self.api_key,
                self.api_secret,
                testnet=self.config.testnet
            )
    
    def _init_ccxt_client(self):
        """Ініціалізація CCXT клієнта"""
        if not CCXT_AVAILABLE:
            raise ImportError("ccxt не встановлено")
        
        config = {
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        }
        
        # Binance не підтримує testnet для публічних endpoint'ів, тому
        # використовуємо testnet URL тільки коли працюємо з автентифікованими запитами.
        if self.config.testnet and not self.config.use_public_data:
            config['urls'] = {'api': 'https://testnet.binance.vision'}
        
        if not self.config.use_public_data:
            config['apiKey'] = self.api_key
            config['secret'] = self.api_secret
        
        self.ccxt_client = ccxt_async.binance(config)
    
    async def close(self):
        """Закриття з'єднань"""
        if self.ccxt_client:
            await self.ccxt_client.close()
            logger.info("🔒 CCXT клієнт закрито")
            self.ccxt_client = None

    async def enable_public_data(self, reason: Optional[str] = None):
        """Примусове переключення на публічні дані"""
        if self.config.use_public_data:
            return

        logger.warning(
            "🔄 %s – переключаємося на публічні дані",
            reason or "Помилка автентифікації Binance"
        )

        await self.close()
        self.config.use_public_data = True
        self._setup_credentials()
        self._initialize_clients()
    
    def get_client(self):
        """Отримання активного клієнта"""
        if self.config.data_source == DataSource.CCXT:
            return self.ccxt_client
        else:
            return self.binance_client