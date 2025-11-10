#!/usr/bin/env python3
"""
Pattern Visualizer - Charts with pattern markers
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, FancyBboxPatch
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️ matplotlib not installed. Run: pip install matplotlib")

logger = logging.getLogger(__name__)


class PatternVisualizer:
    """Візуалізує графіки з позначеними паттернами"""
    
    def __init__(self, output_dir='charts'):
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib required for visualization")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Style settings
        plt.style.use('dark_background')
        self.colors = {
            'LONG': '#00ff00',    # Green
            'SHORT': '#ff0000',   # Red
            'NEUTRAL': '#ffff00', # Yellow
            'candle_up': '#00ff88',
            'candle_down': '#ff4444',
            'ema9': '#00ffff',
            'ema21': '#ff00ff',
            'ema50': '#ffaa00',
            'volume': '#444444'
        }
    
    def plot_pattern(
        self,
        df: pd.DataFrame,
        pattern,
        symbol: str,
        timeframe: str,
        show_indicators: bool = True,
        save: bool = True
    ):
        """
        Малює графік з паттерном
        
        Args:
            df: DataFrame з OHLCV + indicators
            pattern: Pattern object
            symbol: Назва символу
            timeframe: Таймфрейм
            show_indicators: Показувати індикатори
            save: Зберегти в файл
        """
        if len(df) < 50:
            logger.warning(f"Not enough data to plot: {len(df)} candles")
            return None
        
        # Prepare data
        df = df.copy()
        if 'timestamp' not in df.columns and df.index.name != 'timestamp':
            df = df.reset_index()
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.05)
        
        ax_price = fig.add_subplot(gs[0])
        ax_volume = fig.add_subplot(gs[1], sharex=ax_price)
        ax_rsi = fig.add_subplot(gs[2], sharex=ax_price)
        
        # Plot candlesticks
        self._plot_candlesticks(ax_price, df)
        
        # Plot indicators
        if show_indicators:
            self._plot_moving_averages(ax_price, df)
        
        # Plot volume
        self._plot_volume(ax_volume, df)
        
        # Plot RSI
        self._plot_rsi(ax_rsi, df)
        
        # Mark pattern
        self._mark_pattern(ax_price, df, pattern)
        
        # Add pattern info
        self._add_pattern_info(fig, ax_price, pattern, symbol, timeframe)
        
        # Format axes
        self._format_axes(ax_price, ax_volume, ax_rsi, df)
        
        # Title
        title = (f"{symbol} - {timeframe} | "
                f"{pattern.pattern_type.value} ({pattern.direction}) | "
                f"Confidence: {pattern.confidence:.1f}%")
        fig.suptitle(title, fontsize=16, fontweight='bold', color='white')
        
        plt.tight_layout()
        
        # Save
        if save:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{symbol}_{timeframe}_{pattern.pattern_type.value.replace(' ', '_')}_{timestamp}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
            logger.info(f"📊 Chart saved: {filepath}")
            plt.close()
            return filepath
        
        return fig
    
    def _plot_candlesticks(self, ax, df):
        """Малює свічки"""
        for idx in range(len(df)):
            row = df.iloc[idx]
            
            # Colors
            color = self.colors['candle_up'] if row['close'] >= row['open'] else self.colors['candle_down']
            
            # Body
            body_bottom = min(row['open'], row['close'])
            body_height = abs(row['close'] - row['open'])
            
            rect = Rectangle(
                (idx - 0.3, body_bottom),
                0.6, body_height,
                facecolor=color,
                edgecolor=color,
                alpha=0.8
            )
            ax.add_patch(rect)
            
            # Wicks
            ax.plot([idx, idx], [row['low'], row['high']], 
                   color=color, linewidth=1, alpha=0.6)
    
    def _plot_moving_averages(self, ax, df):
        """Малює ковзні середні"""
        x = range(len(df))
        
        if 'ema9' in df.columns:
            ax.plot(x, df['ema9'], color=self.colors['ema9'], 
                   linewidth=1.5, alpha=0.8, label='EMA 9')
        
        if 'ema21' in df.columns:
            ax.plot(x, df['ema21'], color=self.colors['ema21'],
                   linewidth=1.5, alpha=0.8, label='EMA 21')
        
        if 'ema50' in df.columns:
            ax.plot(x, df['ema50'], color=self.colors['ema50'],
                   linewidth=1.5, alpha=0.8, label='EMA 50')
        
        ax.legend(loc='upper left', framealpha=0.3)
    
    def _plot_volume(self, ax, df):
        """Малює об'єм"""
        colors = [self.colors['candle_up'] if row['close'] >= row['open'] 
                 else self.colors['candle_down'] for _, row in df.iterrows()]
        
        ax.bar(range(len(df)), df['volume'], color=colors, alpha=0.5, width=0.8)
        ax.set_ylabel('Volume', color='white', fontsize=10)
        ax.tick_params(axis='y', labelcolor='white', labelsize=8)
        ax.grid(True, alpha=0.2)
    
    def _plot_rsi(self, ax, df):
        """Малює RSI"""
        if 'rsi' not in df.columns:
            return
        
        x = range(len(df))
        ax.plot(x, df['rsi'], color='#00ffff', linewidth=1.5)
        
        # Overbought/Oversold levels
        ax.axhline(y=70, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(y=30, color='green', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(y=50, color='gray', linestyle=':', alpha=0.3, linewidth=1)
        
        # Fill
        ax.fill_between(x, df['rsi'], 50, where=(df['rsi'] >= 50), 
                        color='green', alpha=0.2, interpolate=True)
        ax.fill_between(x, df['rsi'], 50, where=(df['rsi'] < 50),
                        color='red', alpha=0.2, interpolate=True)
        
        ax.set_ylabel('RSI', color='white', fontsize=10)
        ax.set_ylim(0, 100)
        ax.tick_params(axis='y', labelcolor='white', labelsize=8)
        ax.grid(True, alpha=0.2)
    
    def _mark_pattern(self, ax, df, pattern):
        """Позначає паттерн на графіку"""
        # Pattern occurs at the end (last candle)
        pattern_idx = len(df) - 1
        
        # Get pattern price
        pattern_price = pattern.price
        
        # Color based on direction
        color = self.colors.get(pattern.direction, self.colors['NEUTRAL'])
        
        # Draw marker
        marker_style = '^' if pattern.direction == 'LONG' else 'v'
        marker_size = 200 if pattern.confidence >= 75 else 150
        
        ax.scatter(pattern_idx, pattern_price, 
                  marker=marker_style, s=marker_size,
                  color=color, edgecolors='white', linewidths=2,
                  zorder=10, alpha=0.9)
        
        # Draw box around pattern area (last 10 candles)
        pattern_start = max(0, pattern_idx - 10)
        pattern_window = df.iloc[pattern_start:pattern_idx+1]
        
        box_left = pattern_start - 0.5
        box_width = pattern_idx - pattern_start + 1
        box_bottom = pattern_window['low'].min()
        box_height = pattern_window['high'].max() - box_bottom
        
        fancy_box = FancyBboxPatch(
            (box_left, box_bottom),
            box_width, box_height,
            boxstyle="round,pad=0.1",
            edgecolor=color,
            facecolor='none',
            linewidth=2,
            linestyle='--',
            alpha=0.6,
            zorder=5
        )
        ax.add_patch(fancy_box)
        
        # Add text label
        label_y = pattern_window['high'].max() * 1.01
        ax.text(pattern_idx, label_y,
               f"{pattern.pattern_type.value}\n{pattern.confidence:.0f}%",
               color=color, fontsize=10, fontweight='bold',
               ha='center', va='bottom',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='black', 
                        edgecolor=color, alpha=0.8))
    
    def _add_pattern_info(self, fig, ax, pattern, symbol, timeframe):
        """Додає інформаційний блок про паттерн"""
        info_text = (
            f"Symbol: {symbol}\n"
            f"Timeframe: {timeframe}\n"
            f"Pattern: {pattern.pattern_type.value}\n"
            f"Direction: {pattern.direction}\n"
            f"Strength: {pattern.strength.value}\n"
            f"Confidence: {pattern.confidence:.1f}%\n"
            f"Price: ${pattern.price:.4f}\n"
            f"Time: {pattern.timestamp.strftime('%Y-%m-%d %H:%M')}"
        )
        
        # Add text box
        props = dict(boxstyle='round', facecolor='#2a2a2a', edgecolor='white', 
                    alpha=0.9, linewidth=1)
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', fontfamily='monospace',
               bbox=props, color='white')
    
    def _format_axes(self, ax_price, ax_volume, ax_rsi, df):
        """Форматує осі"""
        # Hide x labels except bottom
        plt.setp(ax_price.get_xticklabels(), visible=False)
        plt.setp(ax_volume.get_xticklabels(), visible=False)
        
        # Price axis
        ax_price.set_ylabel('Price (USDT)', color='white', fontsize=12, fontweight='bold')
        ax_price.tick_params(axis='y', labelcolor='white', labelsize=10)
        ax_price.grid(True, alpha=0.2, linestyle='--')
        
        # X axis (bottom only)
        ax_rsi.set_xlabel('Candle Index', color='white', fontsize=10)
        ax_rsi.tick_params(axis='x', labelcolor='white', labelsize=8)
        
        # Set limits
        ax_price.set_xlim(-1, len(df))
        
        # Add minor gridlines
        ax_price.minorticks_on()
        ax_price.grid(which='minor', alpha=0.1, linestyle=':')


async def visualize_top_patterns(scanner, results, top_n=5):
    """Async версія візуалізації"""
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping visualization")
        return []
    
    visualizer = PatternVisualizer()
    chart_files = []
    
    logger.info(f"\n📊 Generating charts for top {top_n} MTF signals...")
    
    for i, result in enumerate(results[:top_n], 1):
        symbol = result['symbol']
        mtf_score = result['mtf_score']
        
        logger.info(f"\n{i}. {symbol} (MTF Score: {mtf_score:.1f})")
        
        # Візуалізуємо паттерни з кожного таймфрейму
        for tf, patterns in result['timeframes'].items():
            if not patterns:
                continue
            
            # Візуалізуємо найкращий паттерн з цього ТФ
            pattern = patterns[0]
            
            try:
                # Завантажуємо дані знову (з індикаторами)
                from multi_timeframe_scanner import TF_DAYS_BACK
                
                df = await scanner.loader.get_historical_data(
                    symbol=symbol,
                    interval=tf,
                    days_back=TF_DAYS_BACK.get(tf, 7)
                )
                
                if df is None or len(df) < 50:
                    continue
                
                # Розраховуємо індикатори
                df = df.reset_index()
                df = scanner.pattern_scanner._calculate_indicators(df)
                
                # Малюємо
                filepath = visualizer.plot_pattern(
                    df, pattern, symbol, tf
                )
                
                if filepath:
                    chart_files.append(filepath)
                    logger.info(f"  ✅ [{tf}] {pattern.pattern_type.value} - {filepath.name}")
                
            except Exception as e:
                logger.warning(f"  ⚠️ [{tf}] Error: {e}")
                continue
    
    logger.info(f"\n✅ Generated {len(chart_files)} charts in {visualizer.output_dir}/")
    return chart_files


if __name__ == "__main__":
    print("Pattern Visualizer - use from multi_timeframe_scanner.py")
