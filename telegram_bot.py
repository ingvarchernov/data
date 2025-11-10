"""
Telegram бот для повідомлень про торгові сигнали
"""
import os
import asyncio
from datetime import datetime
from typing import Optional
from telegram import Bot
from telegram.error import TelegramError
import logging
from dotenv import load_dotenv

# Завантаження .env при імпорті модуля
load_dotenv()

logger = logging.getLogger(__name__)

# Вимикаємо зайві HTTP логи
logging.getLogger('httpx').setLevel(logging.WARNING)


class TelegramNotifier:
    """
    Клас для відправки повідомлень через Telegram бота
    """

    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None):
        self.token = token or os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')
        self.bot: Optional[Bot] = None

        if self.token:
            self.bot = Bot(token=self.token)
            logger.info("✅ Telegram бот ініціалізований")
        else:
            logger.warning("⚠️ TELEGRAM_BOT_TOKEN не знайдено")

    async def send_message(self, message: str, parse_mode: str = 'Markdown') -> bool:
        """
        Відправити повідомлення в Telegram

        Args:
            message: Текст повідомлення
            parse_mode: Форматування (Markdown, HTML)

        Returns:
            bool: True якщо повідомлення відправлено успішно
        """
        if not self.bot or not self.chat_id:
            logger.warning("⚠️ Telegram бот не налаштований")
            return False

        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=parse_mode,
                disable_web_page_preview=True
            )
            # Прибрали зайвий лог - тільки помилки
            return True
        except TelegramError as e:
            logger.error(f"❌ Помилка відправки в Telegram: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Неочікувана помилка Telegram: {e}")
            return False

    async def send_trade_signal(self, symbol: str, action: str, quantity: float,
                               price: float, confidence: float) -> bool:
        """
        Відправити повідомлення про торговий сигнал
        """
        emoji = "🟢" if action.upper() == "BUY" else "🔴"
        message = f"""
{emoji} **Торговий сигнал**

📊 **{symbol}**
🎯 **{action.upper()}**
💰 Кількість: {quantity:.6f}
💵 Ціна: ${price:,.2f}
🎚️ Впевненість: {confidence:.1%}
⏰ Час: {datetime.now().strftime('%H:%M:%S')}
        """.strip()

        return await self.send_message(message)

    async def send_trade_execution(self, symbol: str, action: str, quantity: float,
                                  price: float, cost: float, balance: float, 
                                  is_paper_trading: bool = False) -> bool:
        """
        Відправити повідомлення про виконання угоди
        """
        emoji = "✅" if action.upper() == "BUY" else "💰"
        trade_type = "📝 PAPER TRADING" if is_paper_trading else "🔴 LIVE TRADING"
        message = f"""
{emoji} **{trade_type}**

📊 **{symbol}**
🎯 **{action.upper()}**
💰 Кількість: {quantity:.6f}
💵 Ціна: ${price:,.2f}
💸 Вартість: ${cost:,.2f}
🏦 Баланс: ${balance:,.2f}
        """.strip()

        return await self.send_message(message)

    async def send_pnl_update(self, symbol: str, pnl: float, pnl_percent: float,
                             current_price: float) -> bool:
        """
        Відправити повідомлення про P&L
        """
        emoji = "📈" if pnl >= 0 else "📉"
        message = f"""
{emoji} **P&L оновлення**

📊 **{symbol}**
💰 P&L: ${pnl:,.2f} ({pnl_percent:+.2f}%)
💵 Поточна ціна: ${current_price:,.2f}
        """.strip()

        return await self.send_message(message)

    async def send_system_status(self, status: str, details: str = "") -> bool:
        """
        Відправити повідомлення про статус системи
        """
        emoji = "🟢" if "запущена" in status.lower() else "🔴"
        message = f"""
{emoji} **Система: {status}**

{details}
        """.strip()

        return await self.send_message(message)


# Глобальний екземпляр для використання в усьому проекті
telegram_notifier = TelegramNotifier()