#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebSocket Manager для Binance Futures
Реалізує user data stream для миттєвих оновлень ордерів та позицій
"""
import asyncio
import json
import logging
from typing import Callable, Dict, Optional
import websockets
from binance.client import Client

logger = logging.getLogger(__name__)


class BinanceFuturesWebSocket:
    """WebSocket клієнт для Binance Futures User Data Stream"""
    
    def __init__(self, client: Client, testnet: bool = True):
        self.client = client
        self.testnet = testnet
        self.listen_key = None
        self.ws = None
        self.is_running = False
        
        # Колбеки для подій
        self.on_order_update: Optional[Callable] = None
        self.on_account_update: Optional[Callable] = None
        self.on_position_update: Optional[Callable] = None
        
        # WebSocket URL
        if testnet:
            self.ws_base = "wss://stream.binancefuture.com/ws"
        else:
            self.ws_base = "wss://fstream.binance.com/ws"
    
    async def start(self):
        """Запуск WebSocket з'єднання"""
        if self.is_running:
            logger.warning("WebSocket вже запущено")
            return
        
        try:
            # Отримання listen key
            await self._create_listen_key()
            
            # Запуск WebSocket
            self.is_running = True
            asyncio.create_task(self._ws_loop())
            asyncio.create_task(self._keepalive_loop())
            
            logger.info("✅ WebSocket user data stream запущено")
            
        except Exception as e:
            logger.error(f"❌ Помилка запуску WebSocket: {e}")
            self.is_running = False
    
    async def stop(self):
        """Зупинка WebSocket"""
        self.is_running = False
        
        if self.ws:
            await self.ws.close()
        
        if self.listen_key:
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.futures_close_listen_key(listenKey=self.listen_key)
                )
            except:
                pass
        
        logger.info("👋 WebSocket зупинено")
    
    async def _create_listen_key(self):
        """Створення listen key для user data stream"""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.futures_stream_get_listen_key()
            )
            
            # Перевірка формату відповіді
            if isinstance(response, dict) and 'listenKey' in response:
                self.listen_key = response['listenKey']
            elif isinstance(response, str):
                # Іноді API повертає просто строку
                self.listen_key = response
            else:
                raise ValueError(f"Неочікуваний формат listen key: {type(response)}")
            
            logger.info(f"🔑 Listen key отримано: {self.listen_key[:10]}...")
            
        except Exception as e:
            logger.error(f"❌ Помилка створення listen key: {e}", exc_info=True)
            raise
    
    async def _keepalive_loop(self):
        """Підтримка listen key активним (кожні 30 хв)"""
        while self.is_running:
            await asyncio.sleep(30 * 60)  # 30 хвилин
            
            if not self.listen_key:
                continue
            
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.futures_stream_keepalive(listenKey=self.listen_key)
                )
                logger.debug("✅ Listen key keepalive")
                
            except Exception as e:
                logger.error(f"❌ Keepalive помилка: {e}")
                # Спроба пересоздати listen key
                await self._create_listen_key()
    
    async def _ws_loop(self):
        """Основний цикл WebSocket"""
        while self.is_running:
            try:
                url = f"{self.ws_base}/{self.listen_key}"
                
                async with websockets.connect(url) as ws:
                    self.ws = ws
                    logger.info("🔗 WebSocket підключено")
                    
                    async for message in ws:
                        if not self.is_running:
                            break
                        
                        await self._handle_message(message)
                        
            except websockets.exceptions.ConnectionClosed:
                logger.warning("⚠️ WebSocket з'єднання закрито, переподключення...")
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"❌ WebSocket помилка: {e}")
                await asyncio.sleep(5)
    
    async def _handle_message(self, message: str):
        """Обробка повідомлення з WebSocket"""
        try:
            data = json.loads(message)
            event_type = data.get('e')
            
            if event_type == 'ORDER_TRADE_UPDATE':
                await self._handle_order_update(data)
                
            elif event_type == 'ACCOUNT_UPDATE':
                await self._handle_account_update(data)
                
            else:
                logger.debug(f"Невідомий event: {event_type}")
                
        except Exception as e:
            logger.error(f"❌ Помилка обробки повідомлення: {e}")
    
    async def _handle_order_update(self, data: Dict):
        """
        Обробка ORDER_TRADE_UPDATE події
        
        Приклад даних:
        {
            "e": "ORDER_TRADE_UPDATE",
            "o": {
                "s": "BTCUSDT",           # Symbol
                "c": "client_order_id",   # Client order ID
                "S": "BUY",               # Side
                "o": "MARKET",            # Order type
                "f": "GTC",               # Time in force
                "q": "0.001",             # Original quantity
                "p": "0",                 # Price
                "ap": "50000",            # Average price
                "sp": "0",                # Stop price
                "x": "TRADE",             # Execution type
                "X": "FILLED",            # Order status
                "i": 12345678,            # Order ID
                "l": "0.001",             # Last filled quantity
                "z": "0.001",             # Cumulative filled quantity
                "L": "50000",             # Last filled price
                "n": "0",                 # Commission
                "N": "USDT",              # Commission asset
                "T": 1234567890123,       # Transaction time
                "wt": "CONTRACT_PRICE"    # Working type
            }
        }
        """
        try:
            order = data['o']
            
            event_info = {
                'symbol': order['s'],
                'order_id': order['i'],
                'client_order_id': order.get('c'),
                'side': order['S'],
                'order_type': order['o'],
                'status': order['X'],
                'execution_type': order['x'],
                'quantity': float(order['q']),
                'filled_quantity': float(order['z']),
                'price': float(order['p']) if order['p'] != '0' else None,
                'avg_price': float(order['ap']) if order.get('ap') and order['ap'] != '0' else None,
                'last_filled_price': float(order['L']) if order.get('L') and order['L'] != '0' else None,
                'stop_price': float(order['sp']) if order.get('sp') and order['sp'] != '0' else None,
                'time': order['T']
            }
            
            # Логування важливих подій
            if event_info['execution_type'] == 'TRADE':
                logger.info(
                    f"🔔 ORDER FILLED: {event_info['symbol']} "
                    f"{event_info['side']} {event_info['filled_quantity']} @ "
                    f"${event_info['avg_price']:.2f}"
                )
            elif event_info['status'] in ['CANCELED', 'EXPIRED', 'REJECTED']:
                logger.warning(
                    f"⚠️ ORDER {event_info['status']}: {event_info['symbol']} "
                    f"{event_info['side']} {event_info['order_id']}"
                )
            
            # Виклик колбека якщо встановлено
            if self.on_order_update:
                await self.on_order_update(event_info)
                
        except Exception as e:
            logger.error(f"❌ Помилка обробки order update: {e}")
    
    async def _handle_account_update(self, data: Dict):
        """
        Обробка ACCOUNT_UPDATE події
        
        Включає оновлення балансу та позицій
        """
        try:
            update_data = data.get('a', {})
            
            # Оновлення балансів
            balances = update_data.get('B', [])
            for balance in balances:
                asset = balance['a']
                free = float(balance['wb'])  # Wallet balance
                if asset == 'USDT':
                    logger.info(f"💰 Balance update: ${free:.2f} USDT")
            
            # Оновлення позицій
            positions = update_data.get('P', [])
            for pos in positions:
                symbol = pos['s']
                amount = float(pos['pa'])  # Position amount
                entry_price = float(pos['ep'])
                unrealized_pnl = float(pos['up'])
                
                if amount != 0:
                    side = 'LONG' if amount > 0 else 'SHORT'
                    logger.info(
                        f"📊 Position update: {symbol} {side} "
                        f"{abs(amount)} @ ${entry_price:.2f} "
                        f"(PnL: ${unrealized_pnl:.2f})"
                    )
            
            # Виклик колбека
            if self.on_account_update:
                await self.on_account_update(update_data)
            
            if self.on_position_update and positions:
                await self.on_position_update(positions)
                
        except Exception as e:
            logger.error(f"❌ Помилка обробки account update: {e}")


# Приклад використання
async def example_usage():
    """Приклад використання WebSocket manager"""
    from binance.client import Client
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    client = Client(
        os.getenv('FUTURES_API_KEY'),
        os.getenv('FUTURES_API_SECRET'),
        testnet=True
    )
    
    ws = BinanceFuturesWebSocket(client, testnet=True)
    
    # Встановлення колбеків
    async def on_order(order_info):
        print(f"Order callback: {order_info}")
    
    async def on_account(account_data):
        print(f"Account callback: {account_data}")
    
    ws.on_order_update = on_order
    ws.on_account_update = on_account
    
    # Запуск
    await ws.start()
    
    # Чекати 60 секунд
    await asyncio.sleep(60)
    
    # Зупинка
    await ws.stop()


if __name__ == '__main__':
    asyncio.run(example_usage())
