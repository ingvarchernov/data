#!/usr/bin/env python3
"""Main scanner with training, scanning and chart generation."""
import asyncio
import json
import logging
import time
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import TRADING_CONFIG, FUTURES_CONFIG, DASHBOARD_CONFIG, MARKET_DATA_CONFIG
from unified_binance_loader import UnifiedBinanceLoader
from pattern_analytics import PatternAnalytics
from lstm_model_integration import get_lstm_manager


log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"scanner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


MODELS_DIR = Path("models")
CHARTS_DIR = Path("charts")
MIN_MODEL_AGE_DAYS = 7  # Retrain if model older than 7 days


def _check_model_exists() -> bool:
    """Перевірка наявності тренованої моделі."""
    model_file = MODELS_DIR / "lstm_model.pt"
    return model_file.exists()


def _should_retrain() -> bool:
    """Перевірка чи потрібне дотренування моделі."""
    model_file = MODELS_DIR / "lstm_model.pt"
    if not model_file.exists():
        return True
    
    # Перевіряємо вік моделі
    mtime = datetime.fromtimestamp(model_file.stat().st_mtime)
    age_days = (datetime.now() - mtime).days
    
    logger.info(f"Модель вік: {age_days} днів")
    
    if age_days >= MIN_MODEL_AGE_DAYS:
        logger.info(f"Модель застаріла (>{MIN_MODEL_AGE_DAYS} днів), потрібне дотренування")
        return True
    
    return False


async def ensure_model_trained() -> bool:
    """Переконатися що модель тренована, якщо ні - тренувати."""
    if _check_model_exists() and not _should_retrain():
        logger.info("✅ Модель існує та свіжа, будемо використовувати")
        return True
    
    logger.info("🔄 Запускаю тренування моделі...")
    
    try:
        # Імпортуємо тренування з train_ml_models
        from train_ml_models import run_training
        
        # Запускаємо асинхронне тренування
        await run_training()
        
        logger.info("✅ Тренування завершено")
        return True
        
    except Exception as e:
        logger.error(f"❌ Помилка при тренуванні: {e}")
        # Якщо тренування не вдалось але модель існує - продовжуємо з старою
        if _check_model_exists():
            logger.warning("Буду використовувати стару модель")
            return True
        return False


def _detect_pattern(df) -> tuple:
    """Detect actual market direction, confidence and pattern type from OHLCV data.

    Returns (direction, confidence, pattern_type) where:
      direction   – 'LONG' or 'SHORT'
      confidence  – float 50–90
      pattern_type – descriptive string
    """
    try:
        import pandas_ta as ta_lib
        close = df["close"]
        high  = df["high"]
        low   = df["low"]

        # EMA trend alignment
        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        ema_gap = (ema20.iloc[-1] - ema50.iloc[-1]) / ema50.iloc[-1]  # relative gap

        direction = "LONG" if ema_gap > 0 else "SHORT"
        # Scale confidence: 0% gap → 50, 2% gap → 90 (capped)
        conf = min(90.0, 50.0 + abs(ema_gap) * 2000.0)

        # Pattern type from ATR compression ratio
        atr = ta_lib.atr(high, low, close, length=14)
        if atr is not None and len(atr.dropna()) >= 20:
            atr_clean = atr.dropna()
            atr_ratio = float(atr_clean.iloc[-1]) / float(atr_clean.iloc[-20:].mean())
        else:
            atr_ratio = 1.0

        if atr_ratio < 0.7:
            pattern_type = "Compression Breakout"
        elif abs(ema_gap) > 0.008:
            pattern_type = "Trend Continuation"
        else:
            pattern_type = "Range Breakout"

        return direction, conf, pattern_type

    except Exception:
        return "LONG", 60.0, "Compression Breakout"


def _apply_risk_rule(analysis: Dict[str, Any], direction: str, sl_to_tp_ratio: float) -> Dict[str, Any]:
    """Normalize risk so SL distance is a fixed fraction of TP distance.

    Example: ratio=1/3 => R:R ~= 3.0
    """
    entry = analysis.get("entry_price")
    tp = analysis.get("take_profit")
    if entry is None or tp is None:
        return analysis

    entry = float(entry)
    tp = float(tp)
    ratio = max(0.05, min(1.0, float(sl_to_tp_ratio)))

    tp_distance = abs(tp - entry)
    if tp_distance <= 0:
        tp_distance = max(entry * 0.005, 1e-8)  # fallback 0.5%
        tp = entry + tp_distance if direction == "LONG" else entry - tp_distance

    sl_distance = tp_distance * ratio
    stop_loss = entry - sl_distance if direction == "LONG" else entry + sl_distance

    analysis["entry_price"] = entry
    analysis["take_profit"] = tp
    analysis["stop_loss"] = stop_loss
    analysis["risk_reward_ratio"] = round(tp_distance / sl_distance, 2) if sl_distance > 0 else 0.0

    if entry != 0:
        if direction == "LONG":
            tp_pct = ((tp - entry) / entry) * 100
            sl_pct = ((stop_loss - entry) / entry) * 100
        else:
            tp_pct = ((entry - tp) / entry) * 100
            sl_pct = ((entry - stop_loss) / entry) * 100
        analysis["tp_profit_pct"] = round(tp_pct, 2)
        analysis["sl_loss_pct"] = round(sl_pct, 2)

    return analysis


async def scan_symbols() -> List[Dict[str, Any]]:
    """Сканування торгових пар та генерація сигналів."""
    logger.info("🚀 Запускаю сканування пар...")
    
    loader = UnifiedBinanceLoader(testnet=False, use_public_data=True)
    analytics = PatternAnalytics()
    lstm_manager = get_lstm_manager()
    lstm_manager.initialize()
    
    # Отримуємо список символів
    symbols = FUTURES_CONFIG.get("symbols") or TRADING_CONFIG.get("symbols") or ["BTCUSDT"]
    symbols = symbols[:int(DASHBOARD_CONFIG.get("scan_symbols_limit", 20))]
    
    logger.info(f"📊 Сканую {len(symbols)} пар...")
    
    timeframe = DASHBOARD_CONFIG.get("default_timeframe", "1h")
    days_back = int(DASHBOARD_CONFIG.get("default_lookback_days", 7))
    signal_mode = str(DASHBOARD_CONFIG.get("signal_mode", "risk_levels")).lower()
    sl_to_tp_ratio = float(DASHBOARD_CONFIG.get("sl_to_tp_ratio", 0.333333))
    
    t0 = time.time()
    signals: List[Dict[str, Any]] = []
    
    try:
        # Завантажуємо дані для всіх символів
        data = await loader.get_multiple_symbols(symbols, interval=timeframe, days_back=days_back)
        
        for symbol, df in data.items():
            if df is None or df.empty or len(df) < 80:
                continue
            
            logger.debug(f"  Аналізую {symbol}...")
            
            try:
                # Detect actual pattern, direction and confidence from price action
                pattern_dir, pattern_conf, pattern_type = _detect_pattern(df)
                ensemble = lstm_manager.predict_ensemble(
                    df,
                    pattern_confidence=pattern_conf,
                    pattern_direction=pattern_dir
                )
                
                if not ensemble:
                    continue
                
                # Розраховуємо Entry, SL, TP
                direction = "LONG" if ensemble.get("prediction") == "UP" else "SHORT"
                
                analysis = analytics.calculate_entry_sl_tp(
                    df=df,
                    pattern_type=pattern_type,
                    pattern_direction=direction,
                    pattern_data={"symbol": symbol},
                )

                if signal_mode == "risk_levels":
                    analysis = _apply_risk_rule(analysis, direction, sl_to_tp_ratio)
                else:
                    # Direction-only mode: keep only current/entry and direction, no hard SL/TP levels
                    current_px = analysis.get("current_price")
                    entry_px = analysis.get("entry_price", current_px)
                    analysis["entry_price"] = float(entry_px) if entry_px is not None else None
                    analysis["current_price"] = float(current_px) if current_px is not None else None
                    analysis["stop_loss"] = None
                    analysis["take_profit"] = None
                    analysis["risk_reward_ratio"] = 0.0
                    analysis["tp_profit_pct"] = 0.0
                    analysis["sl_loss_pct"] = 0.0
                
                conf = float(ensemble.get("confidence", 0.0))
                # Use conviction (distance from neutral 0.5) as signal quality score
                # ensemble_score > 0.5 → LONG, < 0.5 → SHORT; conf = |score-0.5|*200
                mtf_score = conf
                
                # Require at least 15% model conviction (filters near-neutral noise)
                if mtf_score < 15.0:
                    continue
                
                signal = {
                    "symbol": symbol,
                    "mtf_score": mtf_score,
                    "analytics": {
                        "pattern_type": pattern_type,
                        "timeframe": timeframe,
                        "current_price": analysis.get("current_price"),
                        "entry_price": analysis.get("entry_price"),
                        "stop_loss": analysis.get("stop_loss"),
                        "take_profit": analysis.get("take_profit"),
                        "risk_reward_ratio": analysis.get("risk_reward_ratio", 0.0),
                        "tp_profit_pct": analysis.get("tp_profit_pct", 0.0),
                        "sl_loss_pct": analysis.get("sl_loss_pct", 0.0),
                    },
                    "confluence": {
                        "dominant_direction": direction,
                        "same_direction": max(0.0, min(1.0, conf / 100.0)),
                    },
                }
                signals.append(signal)
                logger.info(f"  ✅ {symbol}: score={mtf_score:.1f}, direction={direction}")
                
            except Exception as e:
                logger.debug(f"  ⚠️  Помилка при обробці {symbol}: {e}")
                continue
        
        # Сортуємо за оцінкою та обмежуємо
        signals.sort(key=lambda x: x.get("mtf_score", 0.0), reverse=True)
        signals = signals[:int(DASHBOARD_CONFIG.get("signal_limit", 100))]
        
        duration = time.time() - t0
        logger.info(f"✅ Сканування завершено: {len(signals)} сигналів за {duration:.2f}с")
        
    finally:
        await loader.close()
    
    return signals


async def generate_charts(signals: List[Dict[str, Any]]) -> None:
    """Генерація графіків для всіх знайдених сигналів."""
    if not signals:
        logger.info("Немає сигналів для відображення")
        return
    
    logger.info(f"📈 Генерую графіки для {len(signals)} сигналів...")
    
    CHARTS_DIR.mkdir(exist_ok=True)
    
    loader = UnifiedBinanceLoader(testnet=False, use_public_data=True)
    
    timeframe = DASHBOARD_CONFIG.get("default_timeframe", "1h")
    days_back = int(DASHBOARD_CONFIG.get("default_lookback_days", 7))
    
    try:
        chart_files = []
        for idx, signal in enumerate(signals[:10], 1):  # Графіки для топ 10
            symbol = signal["symbol"]
            logger.info(f"  Генерую графік {idx}: {symbol}...")
            
            try:
                # Завантажуємо дані для побудови графіка
                df = await loader.get_historical_data(
                    symbol=symbol,
                    interval=timeframe,
                    days_back=days_back
                )
                
                if df is None or df.empty:
                    logger.warning(f"    Немає даних для {symbol}")
                    continue
                
                # Генеруємо графік
                analytics_data = signal["analytics"]
                
                # Додаємо дані для відображення на графіку
                chart_data = {
                    "symbol": symbol,
                    "entry_price": analytics_data["entry_price"],
                    "stop_loss": analytics_data["stop_loss"],
                    "take_profit": analytics_data["take_profit"],
                    "pattern_type": analytics_data["pattern_type"],
                    "direction": signal["confluence"]["dominant_direction"],
                }
                
                filename = _generate_chart_pdf(df, chart_data, idx)
                chart_files.append({
                    "index": idx,
                    "symbol": symbol,
                    "score": signal["mtf_score"],
                    "direction": signal["confluence"]["dominant_direction"],
                    "filename": filename
                })
                logger.info(f"    ✅ Графік збережено: {filename}")
                
            except Exception as e:
                logger.warning(f"    ⚠️  Помилка при генерації графіка {symbol}: {e}")
                continue
        
        # Зберігаємо простий маніфест із PDF файлами
        if chart_files:
            manifest_file = CHARTS_DIR / "charts_manifest.json"
            with open(manifest_file, "w", encoding="utf-8") as f:
                json.dump(chart_files, f, ensure_ascii=False, indent=2)
            logger.info(f"✅ Маніфест графіків: {manifest_file}")
    
    finally:
        await loader.close()


def _generate_chart_pdf(df, chart_data: Dict[str, Any], index: int) -> str:
    """Генерація PDF графіка з candlestick, обсягами та індикаторами."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import mplfinance as mpf
        import pandas_ta as ta
    except ImportError:
        logger.warning("matplotlib/mplfinance/pandas_ta не встановлені, пропускаю графіки")
        return ""
    
    symbol = chart_data["symbol"]
    entry = chart_data.get("entry_price")
    sl = chart_data.get("stop_loss")
    tp = chart_data.get("take_profit")
    direction = chart_data["direction"]
    pattern = chart_data["pattern_type"]
    
    # Останні 100 свічок
    df_plot = df.tail(100).copy()

    # Конвертуємо індекс в timezone-naive
    if df_plot.index.tz is not None:
        df_plot.index = df_plot.index.tz_localize(None)

    # Формат для mplfinance
    df_plot = df_plot.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    })

    for col in ("Open", "High", "Low", "Close", "Volume"):
        df_plot[col] = df_plot[col].astype(float)
    
    # Розраховуємо технічні індикатори
    try:
        df_plot["rsi"] = ta.rsi(df_plot["Close"], length=14)
        df_plot["ema20"] = ta.ema(df_plot["Close"], length=20)
        df_plot["ema50"] = ta.ema(df_plot["Close"], length=50)

        # MACD
        macd = ta.macd(df_plot["Close"], fast=12, slow=26, signal=9)
        if macd is not None:
            df_plot["macd"] = macd.iloc[:, 0]
            df_plot["macd_signal"] = macd.iloc[:, 1]
            df_plot["macd_hist"] = macd.iloc[:, 2]
    except Exception as e:
        logger.debug(f"Помилка при розрахунку індикаторів: {e}")
    # Підготовка технічних оверлеїв
    add_plots = []
    if "ema20" in df_plot.columns:
        add_plots.append(mpf.make_addplot(df_plot["ema20"], panel=0, color="orange", width=1))
    if "ema50" in df_plot.columns:
        add_plots.append(mpf.make_addplot(df_plot["ema50"], panel=0, color="magenta", width=1))
    if "rsi" in df_plot.columns:
        add_plots.append(mpf.make_addplot(df_plot["rsi"], panel=2, color="dodgerblue", ylabel="RSI"))
    if "macd" in df_plot.columns:
        add_plots.append(mpf.make_addplot(df_plot["macd"], panel=3, color="cyan", ylabel="MACD"))
    if "macd_signal" in df_plot.columns:
        add_plots.append(mpf.make_addplot(df_plot["macd_signal"], panel=3, color="tomato"))
    if "macd_hist" in df_plot.columns:
        add_plots.append(mpf.make_addplot(df_plot["macd_hist"], panel=3, type="bar", alpha=0.4, color="gray"))

    # ========== Оформлення ==========
    has_levels = entry is not None and sl is not None and tp is not None
    risk_reward = abs((tp - entry) / (entry - sl)) if has_levels and entry != sl else 0
    if has_levels:
        title = f"{symbol} {direction} | Pattern: {pattern} | R:R = {risk_reward:.2f}"
    else:
        title = f"{symbol} {direction} | Pattern: {pattern} | Signal Only"

    fig, axes = mpf.plot(
        df_plot,
        type="candle",
        style="charles",
        volume=True,
        addplot=add_plots,
        title=title,
        returnfig=True,
        panel_ratios=(4, 1, 1, 1),
        figscale=1.15,
        figratio=(16, 9),
    )

    # Горизонтальні рівні на ціновій панелі
    price_ax = axes[0]
    if has_levels:
        price_ax.axhline(entry, color="#6f42c1", linestyle="--", linewidth=1.2, label=f"Entry {entry:.4f}")
        price_ax.axhline(sl, color="#d62728", linestyle="--", linewidth=1.2, label=f"SL {sl:.4f}")
        price_ax.axhline(tp, color="#2ca02c", linestyle="--", linewidth=1.2, label=f"TP {tp:.4f}")
        price_ax.legend(loc="upper left", fontsize=8)

    # Optional projected future path (non-existing bars shown as dashed line)
    if bool(DASHBOARD_CONFIG.get("project_future_bars", True)):
        future_n = max(1, int(DASHBOARD_CONFIG.get("future_bars_count", 6)))
        last_close = float(df_plot["Close"].iloc[-1])
        target_price = float(tp) if tp is not None else (last_close * (1.01 if direction == "LONG" else 0.99))
        x0 = len(df_plot) - 1
        x_future = [x0 + i for i in range(1, future_n + 1)]
        y_future = np.linspace(last_close, target_price, future_n)
        price_ax.plot(x_future, y_future, linestyle="--", linewidth=1.5, color="#888888", alpha=0.9, label="Projection")
        price_ax.scatter(x_future, y_future, s=10, color="#888888", alpha=0.7)

    # RSI межі якщо панель RSI присутня
    if len(axes) >= 3:
        rsi_ax = axes[2]
        rsi_ax.axhline(30, color="gray", linestyle=":", linewidth=1)
        rsi_ax.axhline(70, color="gray", linestyle=":", linewidth=1)

    # Зберігаємо PDF
    filename = CHARTS_DIR / f"{index:02d}_{symbol}_{direction}.pdf"
    fig.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close(fig)
    
    return str(filename)


async def main() -> None:
    """Основна функція."""
    logger.info("=" * 60)
    logger.info("🚀 Scanner with Training and Chart Generation")
    logger.info("=" * 60)
    
    # 1. Перевіряємо/тренуємо модель
    model_ok = await ensure_model_trained()
    if not model_ok:
        logger.error("❌ Не вдалось підготувати модель!")
        return
    
    # 2. Сканування пар
    signals = await scan_symbols()
    
    if not signals:
        logger.warning("⚠️  Сигналів не знайдено")
        return
    
    # Виводимо результати
    logger.info("\n" + "=" * 60)
    logger.info(f"📊 Знайдено {len(signals)} сигналів:")
    logger.info("=" * 60)
    for idx, signal in enumerate(signals[:10], 1):
        sym = signal["symbol"]
        score = signal["mtf_score"]
        direction = signal["confluence"]["dominant_direction"]
        entry = signal["analytics"].get("entry_price")
        tp = signal["analytics"].get("take_profit")
        sl = signal["analytics"].get("stop_loss")
        rr = signal["analytics"]["risk_reward_ratio"]

        if entry is not None and tp is not None and sl is not None:
            logger.info(f"{idx}. {sym} | score={score:.1f} | {direction} | "
                        f"Entry={entry:.2f} | TP={tp:.2f} | SL={sl:.2f} | R:R={rr:.2f}")
        else:
            logger.info(f"{idx}. {sym} | score={score:.1f} | {direction} | Signal Only")
    
    # 3. Генеруємо графіки
    logger.info("\n" + "=" * 60)
    await generate_charts(signals)
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ Сканування завершено!")
    logger.info(f"📂 Графіки збережено в: {CHARTS_DIR.absolute()}")


if __name__ == "__main__":
    asyncio.run(main())
