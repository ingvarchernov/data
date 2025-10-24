#!/usr/bin/env python3
"""
Візуалізація торгових сигналів класифікаційної моделі
"""
import sys
import os
import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Вимкнути GPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from analyze_classification import ClassificationAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_visualization(df_full, signals_df, symbol='BTCUSDT'):
    """Створення графіка з сигналами"""
    
    # Створення фігури з 3 підграфіками
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1.5], hspace=0.3)
    
    ax1 = fig.add_subplot(gs[0])  # Ціна + сигнали
    ax2 = fig.add_subplot(gs[1], sharex=ax1)  # Різниця UP-DOWN
    ax3 = fig.add_subplot(gs[2], sharex=ax1)  # Ймовірності
    ax4 = fig.add_subplot(gs[3])  # Статистика
    
    # Підготовка даних
    prices = signals_df['price'].values
    timestamps = df_full.index[-len(signals_df):]
    
    # ========== ГРАФІК 1: ЦІНА + СИГНАЛИ ==========
    ax1.plot(timestamps, prices, label='BTC Price', color='black', linewidth=2, alpha=0.7)
    
    # Позначки сигналів
    buy_mask = signals_df['prediction'] == 'UP'
    sell_mask = signals_df['prediction'] == 'DOWN'
    neutral_mask = signals_df['prediction'] == 'NEUTRAL'
    
    # Strong сигнали (різниця > 5%)
    strong_buy = (buy_mask) & (signals_df['up_down_diff'] > 0.05)
    strong_sell = (sell_mask) & (signals_df['up_down_diff'] < -0.05)
    
    # BUY сигнали
    if buy_mask.any():
        ax1.scatter(timestamps[buy_mask], prices[buy_mask], 
                   color='green', marker='^', s=100, alpha=0.6, 
                   label='BUY Signal', zorder=5)
    
    if strong_buy.any():
        ax1.scatter(timestamps[strong_buy], prices[strong_buy], 
                   color='darkgreen', marker='^', s=200, alpha=0.9, 
                   label='STRONG BUY', zorder=6, edgecolors='yellow', linewidths=2)
    
    # SELL сигнали
    if sell_mask.any():
        ax1.scatter(timestamps[sell_mask], prices[sell_mask], 
                   color='red', marker='v', s=100, alpha=0.6, 
                   label='SELL Signal', zorder=5)
    
    if strong_sell.any():
        ax1.scatter(timestamps[strong_sell], prices[strong_sell], 
                   color='darkred', marker='v', s=200, alpha=0.9, 
                   label='STRONG SELL', zorder=6, edgecolors='yellow', linewidths=2)
    
    # Поточна ціна
    current_price = prices[-1]
    current_signal = signals_df.iloc[-1]
    ax1.axhline(y=current_price, color='blue', linestyle='--', linewidth=1, alpha=0.5)
    ax1.text(timestamps[-1], current_price, f' ${current_price:,.0f}', 
            verticalalignment='center', fontsize=10, color='blue', fontweight='bold')
    
    ax1.set_ylabel('Ціна BTC (USDT)', fontsize=12, fontweight='bold')
    ax1.set_title(f'{symbol} - Торгові сигнали класифікаційної моделі\n'
                 f'Поточний сигнал: {current_signal["prediction"]} | '
                 f'Рекомендація: {current_signal["recommendation"]}',
                 fontsize=14, fontweight='bold', pad=20)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # ========== ГРАФІК 2: РІЗНИЦЯ UP-DOWN ==========
    diff_values = signals_df['up_down_diff'].values * 100
    colors = ['green' if d > 0 else 'red' for d in diff_values]
    
    ax2.bar(timestamps, diff_values, color=colors, alpha=0.6, width=0.02)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.axhline(y=5, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Strong BUY threshold')
    ax2.axhline(y=-5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Strong SELL threshold')
    ax2.axhline(y=3, color='lightgreen', linestyle=':', linewidth=1, alpha=0.5)
    ax2.axhline(y=-3, color='lightcoral', linestyle=':', linewidth=1, alpha=0.5)
    
    ax2.set_ylabel('UP-DOWN різниця (%)', fontsize=10, fontweight='bold')
    ax2.set_title('Різниця ймовірностей UP vs DOWN', fontsize=11)
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # ========== ГРАФІК 3: ЙМОВІРНОСТІ ==========
    prob_down = signals_df['prob_down'].values * 100
    prob_neutral = signals_df['prob_neutral'].values * 100
    prob_up = signals_df['prob_up'].values * 100
    
    ax3.fill_between(timestamps, 0, prob_down, color='red', alpha=0.3, label='P(DOWN)')
    ax3.fill_between(timestamps, prob_down, prob_down + prob_neutral, 
                     color='gray', alpha=0.3, label='P(NEUTRAL)')
    ax3.fill_between(timestamps, prob_down + prob_neutral, 100, 
                     color='green', alpha=0.3, label='P(UP)')
    
    ax3.set_ylabel('Ймовірність (%)', fontsize=10, fontweight='bold')
    ax3.set_xlabel('Час', fontsize=10, fontweight='bold')
    ax3.set_title('Розподіл ймовірностей класів', fontsize=11)
    ax3.set_ylim(0, 100)
    ax3.legend(loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # ========== ГРАФІК 4: СТАТИСТИКА ==========
    ax4.axis('off')
    
    # Підрахунок статистики
    total = len(signals_df)
    up_count = (signals_df['prediction'] == 'UP').sum()
    down_count = (signals_df['prediction'] == 'DOWN').sum()
    neutral_count = (signals_df['prediction'] == 'NEUTRAL').sum()
    
    strong_buy_count = ((signals_df['prediction'] == 'UP') & 
                       (signals_df['up_down_diff'] > 0.05)).sum()
    strong_sell_count = ((signals_df['prediction'] == 'DOWN') & 
                         (signals_df['up_down_diff'] < -0.05)).sum()
    
    avg_diff = signals_df['up_down_diff'].mean() * 100
    
    # Текст статистики
    stats_text = f"""
СТАТИСТИКА ПРОГНОЗІВ ({total} точок):

🟢 UP сигналів:       {up_count:3d} ({up_count/total*100:5.1f}%)
   💪 STRONG BUY:     {strong_buy_count:3d}
   
🔴 DOWN сигналів:     {down_count:3d} ({down_count/total*100:5.1f}%)
   💪 STRONG SELL:    {strong_sell_count:3d}
   
⚪ NEUTRAL сигналів:  {neutral_count:3d} ({neutral_count/total*100:5.1f}%)

📊 Середня різниця UP-DOWN: {avg_diff:+.2f}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ПОТОЧНИЙ СИГНАЛ:
{current_signal['prediction']} | ${current_price:,.2f}
Різниця: {current_signal['up_down_diff']*100:+.1f}%
Рекомендація: {current_signal['recommendation']}

Target: {'+0.5-1%' if 'BUY' in current_signal['recommendation'] else '-0.5-1%' if 'SELL' in current_signal['recommendation'] else 'N/A'}
Stop-loss: {'-0.3%' if 'BUY' in current_signal['recommendation'] else '+0.3%' if 'SELL' in current_signal['recommendation'] else 'N/A'}
"""
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Форматування дат на осі X
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m %H:%M'))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    return fig


async def main():
    """Головна функція"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Візуалізація торгових сигналів')
    parser.add_argument('--symbol', type=str, default='BTCUSDT')
    parser.add_argument('--days', type=int, default=14)
    parser.add_argument('--output', type=str, help='Файл для збереження графіка')
    
    args = parser.parse_args()
    
    try:
        # Аналіз
        analyzer = ClassificationAnalyzer(symbol=args.symbol)
        
        logger.info("📦 Завантаження моделі...")
        await analyzer.load_model()
        
        logger.info("📥 Завантаження даних...")
        df = await analyzer.load_data(days=args.days)
        
        logger.info("🔧 Розрахунок features...")
        df_features = analyzer.calculate_features(df)
        
        logger.info("🔮 Прогнозування...")
        X, prices = analyzer.create_sequences(df_features)
        predicted_classes, confidences, probabilities = await analyzer.predict(X)
        
        logger.info("📊 Генерація сигналів...")
        signals_df = analyzer.generate_signals(predicted_classes, confidences, probabilities, prices)
        
        # Створення графіка
        logger.info("📈 Створення візуалізації...")
        fig = create_visualization(df_features, signals_df, args.symbol)
        
        # Збереження
        os.makedirs('graphics', exist_ok=True)
        if args.output:
            output_file = args.output
        else:
            output_file = f'graphics/trading_signals_{args.symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        logger.info(f"✅ Графік збережено: {output_file}")
        
        # Показати
        logger.info("👁️ Відображення графіка...")
        plt.show()
        
    except Exception as e:
        logger.error(f"❌ Помилка: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
