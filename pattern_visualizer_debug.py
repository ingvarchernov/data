"""
Pattern Visualizer Debug - Interactive candlestick charts with technical indicators
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas_ta as ta
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class PatternVisualizer:
    """Interactive visualization of patterns with technical indicators"""
    
    def __init__(self):
        # Modern color scheme
        self.colors = {
            'compression': 'rgba(30, 144, 255, 0.3)',  # Dodger blue with transparency
            'resistance': '#FF6B6B',  # Coral red
            'support': '#4ECDC4',    # Teal
            'entry': '#9B59B6',      # Purple
            'sl': '#E74C3C',         # Red
            'tp': '#27AE60',         # Green
            'win': '#2ECC71',        # Bright green
            'loss': '#E74C3C',       # Bright red
            'background': '#1a1a1a', # Dark background
            'grid': '#333333'        # Dark grid
        }
    
    async def visualize_from_backtest(
        self,
        backtest_file: str,
        pattern_types: List[str],
        num_samples: int = 10,
        win_only: Optional[bool] = None
    ):
        """
        Візуалізація трейдів з бектесту
        
        Args:
            backtest_file: Шлях до JSON файлу з результатами
            pattern_types: Список типів паттернів для візуалізації
            num_samples: Кількість семплів для візуалізації
            win_only: True - тільки wins, False - тільки losses, None - все
        """
        print(f"🔍 Завантажую дані з {backtest_file}")
        # Завантажуємо дані
        with open(backtest_file) as f:
            data = json.load(f)
        
        # Отримуємо symbol і interval з верхнього рівня
        symbol = data.get('symbol')
        interval = data.get('interval')
        
        trades = data.get('trades', [])
        print(f"📊 Знайдено {len(trades)} трейдів для {symbol} {interval}")
        
        # Фільтруємо трейди
        filtered_trades = []
        for trade in trades:
            # Підтримка обох форматів: 'pattern' і 'pattern_type'
            pattern = trade.get('pattern') or trade.get('pattern_type')
            if pattern_types:
                # Перевіряємо чи pattern містить будь-який з pattern_types
                if not any(pt in pattern for pt in pattern_types):
                    continue
            if win_only is True and not trade.get('win'):
                continue
            if win_only is False and trade.get('win'):
                continue
            filtered_trades.append(trade)
        
        print(f"📊 Після фільтрації: {len(filtered_trades)} трейдів")
        logger.info(f"📊 Візуалізую {min(num_samples, len(filtered_trades))} трейдів з {len(filtered_trades)}")
        
        # Візуалізуємо кожен трейд
        for i, trade in enumerate(filtered_trades[:num_samples]):
            print(f"🔨 Обробка трейду {i + 1}/{min(num_samples, len(filtered_trades))}")
            await self._visualize_trade(trade, i + 1, symbol, interval)
    
    async def _visualize_trade(self, trade: Dict, index: int, symbol: str, timeframe: str):
        """Interactive visualization of a single trade with technical indicators"""
        try:
            print(f"  📈 Візуалізація трейду {index}")
            entry_time = pd.Timestamp(trade['entry_time'], tz='UTC')
            exit_time = pd.Timestamp(trade['exit_time'], tz='UTC')
            
            print(f"  🔄 Завантаження даних для {symbol} {timeframe}")
            # Load historical data
            from unified_binance_loader import UnifiedBinanceLoader
            loader = UnifiedBinanceLoader(testnet=False)
            
            # Load data around entry (±100 candles)
            start_date = entry_time - timedelta(hours=100)
            end_date = exit_time + timedelta(hours=50)
            
            df = await loader.get_historical_data(
                symbol=symbol,
                interval=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            
            if df is None or len(df) < 50:  # Need more data for indicators
                logger.warning(f"⚠️ Insufficient data for {symbol} {timeframe}")
                print(f"  ⚠️ Недостатньо даних для {symbol} {timeframe}")
                return
            
            print(f"  ✅ Завантажено {len(df)} свічок")
            
            # Clean and prepare data
            df = df.dropna(subset=['open', 'high', 'low', 'close'])
            if 'volume' not in df.columns or df['volume'].isna().all():
                df['volume'] = 1000.0
            else:
                df['volume'] = df['volume'].fillna(1000.0)
            
            # Convert index to timezone-naive for Plotly compatibility
            df.index = df.index.tz_localize(None)
            
            print(f"  📊 Обчислення індикаторів...")
            
            # Calculate technical indicators - try Rust first, fallback to pandas_ta
            try:
                from pattern_detector import calculate_indicators
                indicators = calculate_indicators(df['close'].values.tolist())
                df['ema_20'] = indicators['ema20']
                df['ema_50'] = indicators['ema50']
                df['rsi'] = indicators['rsi']
                df['macd'] = indicators['macd_line']
                df['macd_signal'] = indicators['macd_signal']
                df['macd_hist'] = indicators['macd_histogram']
            except (ImportError, RuntimeError, KeyError, TypeError) as e:
                logger.warning(f"⚠️ Pattern detector unavailable ({e}), using pandas_ta")
                df['ema_20'] = ta.ema(df['close'], length=20)
                df['ema_50'] = ta.ema(df['close'], length=50)
                df['rsi'] = ta.rsi(df['close'], length=14)
                macd_result = ta.macd(df['close'], fast=12, slow=26, signal=9)
                if macd_result is not None and not macd_result.empty:
                    df['macd'] = macd_result.iloc[:, 0]
                    df['macd_signal'] = macd_result.iloc[:, 1] if len(macd_result.columns) > 1 else 0
                    df['macd_hist'] = macd_result.iloc[:, 2] if len(macd_result.columns) > 2 else 0
                else:
                    df['macd'] = 0
                    df['macd_signal'] = 0
                    df['macd_hist'] = 0
            
            # Trade details
            entry_price = trade['entry_price']
            exit_price = trade['exit_price']
            # Підтримка різних форматів JSON
            sl_price = trade.get('stop_loss') or trade.get('sl_price')
            tp_price = trade.get('take_profit') or trade.get('tp_price')
            direction = trade.get('direction') or trade.get('side')
            win = trade['win']
            pnl = trade.get('pnl') or trade.get('pnl_pct', 0)
            
            # Compression zone
            compression_start_idx = trade.get('compression_start_idx')
            compression_end_idx = trade.get('compression_end_idx')
            comp_start_time = None
            comp_end_time = None
            comp_high = None
            comp_low = None
            
            if compression_start_idx is not None and compression_end_idx is not None:
                try:
                    # Convert entry_time to timezone-naive for comparison
                    entry_time_naive = entry_time.tz_localize(None) if entry_time.tz else entry_time
                    comp_start_time = entry_time_naive - timedelta(hours=(compression_end_idx - compression_start_idx + 5))
                    comp_end_time = entry_time_naive
                    mask = (df.index >= comp_start_time) & (df.index <= comp_end_time)
                    if mask.any():
                        comp_high = df.loc[mask, 'high'].max()
                        comp_low = df.loc[mask, 'low'].min()
                except Exception as e:
                    logger.warning(f"Compression zone calculation error: {e}")
            
            # Create subplots: main chart, volume, RSI, MACD
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=('Price', 'Volume', 'RSI (14)', 'MACD'),
                row_heights=[0.5, 0.15, 0.15, 0.2]
            )
            
            # Candlestick chart
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Candles',
                increasing_line_color=self.colors['win'],
                decreasing_line_color=self.colors['loss']
            ), row=1, col=1)
            
            # EMAs
            fig.add_trace(go.Scatter(
                x=df.index, y=df['ema_20'],
                line=dict(color='orange', width=1.5),
                name='EMA 20'
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=df.index, y=df['ema_50'],
                line=dict(color='blue', width=1.5),
                name='EMA 50'
            ), row=1, col=1)
            
            # Compression zone as filled area
            if comp_start_time and comp_end_time and comp_high and comp_low:
                fig.add_shape(
                    type="rect",
                    x0=comp_start_time, x1=comp_end_time,
                    y0=comp_low, y1=comp_high,
                    fillcolor=self.colors['compression'],
                    line=dict(width=0),
                    layer="below"
                )
            
            # Horizontal lines for SL/TP
            fig.add_hline(y=sl_price, line_dash="dash", line_color=self.colors['sl'], 
                         annotation_text="SL", row=1, col=1)
            fig.add_hline(y=tp_price, line_dash="dash", line_color=self.colors['tp'], 
                         annotation_text="TP", row=1, col=1)
            
            # Entry/Exit markers
            entry_time_naive = entry_time.tz_localize(None) if entry_time.tz else entry_time
            exit_time_naive = exit_time.tz_localize(None) if exit_time.tz else exit_time
            
            entry_results = df.index.get_indexer([entry_time_naive], method='nearest')
            entry_idx = entry_results[0] if len(entry_results) > 0 else -1
            
            exit_results = df.index.get_indexer([exit_time_naive], method='nearest')
            exit_idx = exit_results[0] if len(exit_results) > 0 else -1
            
            if entry_idx >= 0 and entry_idx < len(df):
                fig.add_trace(go.Scatter(
                    x=[df.index[entry_idx]],
                    y=[entry_price],
                    mode='markers',
                    marker=dict(symbol='triangle-up' if direction == 'LONG' else 'triangle-down',
                              size=12, color=self.colors['entry']),
                    name='Entry'
                ), row=1, col=1)
            
            if exit_idx >= 0 and exit_idx < len(df):
                fig.add_trace(go.Scatter(
                    x=[df.index[exit_idx]],
                    y=[exit_price],
                    mode='markers',
                    marker=dict(symbol='triangle-down' if direction == 'LONG' else 'triangle-up',
                              size=12, color=self.colors['win'] if win else self.colors['loss']),
                    name='Exit'
                ), row=1, col=1)
            
            # Volume bar chart
            fig.add_trace(go.Bar(
                x=df.index, y=df['volume'],
                name='Volume',
                marker_color='rgba(158,158,158,0.8)'
            ), row=2, col=1)
            
            # RSI
            fig.add_trace(go.Scatter(
                x=df.index, y=df['rsi'],
                line=dict(color='purple', width=1.5),
                name='RSI'
            ), row=3, col=1)
            fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)
            
            # MACD
            fig.add_trace(go.Scatter(
                x=df.index, y=df['macd'],
                line=dict(color='blue', width=1.5),
                name='MACD'
            ), row=4, col=1)
            fig.add_trace(go.Scatter(
                x=df.index, y=df['macd_signal'],
                line=dict(color='red', width=1.5),
                name='Signal'
            ), row=4, col=1)
            fig.add_trace(go.Bar(
                x=df.index, y=df['macd_hist'],
                name='Histogram',
                marker_color='rgba(0,255,0,0.5)' if df['macd_hist'].iloc[-1] > 0 else 'rgba(255,0,0,0.5)'
            ), row=4, col=1)
            
            # Update layout
            pnl_sign = '+' if pnl >= 0 else ''
            # Компактний заголовок для PNG
            title = f"{symbol} {timeframe} | {direction} | "
            title += f"{'✅ WIN' if win else '❌ LOSS'} {pnl_sign}{pnl:.2f}%<br>"
            title += f"Entry: ${entry_price:.2f} → Exit: ${exit_price:.2f}"
            
            fig.update_layout(
                title=dict(
                    text=title,
                    font=dict(size=16, color='white')
                ),
                xaxis_rangeslider_visible=False,
                template='plotly_dark',
                paper_bgcolor=self.colors['background'],
                plot_bgcolor=self.colors['background'],
                font=dict(color='white', size=10),
                height=900,  # Оптимально для Telegram
                width=1200,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font=dict(size=9)
                ),
                margin=dict(l=60, r=30, t=80, b=40)
            )
            
            # Update y-axes
            fig.update_yaxes(title_text="Price", row=1, col=1, title_font=dict(size=11))
            fig.update_yaxes(title_text="Volume", row=2, col=1, title_font=dict(size=11))
            fig.update_yaxes(title_text="RSI", row=3, col=1, title_font=dict(size=11))
            fig.update_yaxes(title_text="MACD", row=4, col=1, title_font=dict(size=11))
            
            # Save as PNG for Telegram
            output_dir = Path('charts')
            output_dir.mkdir(exist_ok=True)
            entry_ts = int(entry_time_naive.timestamp()) if hasattr(entry_time_naive, 'timestamp') else int(entry_time.timestamp())
            filename = output_dir / f"trade_{index}_{symbol}_{timeframe}_{entry_ts}.png"
            
            # Зберігаємо як PNG з високою якістю
            fig.write_image(str(filename), format='png', scale=2)
            
            logger.info(f"✅ Chart {index} saved: {filename.name}")
            print(f"  ✅ Збережено: {filename.name}")
            
        except Exception as e:
            logger.error(f"❌ Error visualizing trade {index}: {e}", exc_info=True)
