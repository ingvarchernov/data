#!/usr/bin/env python3
"""LSTM model training for crypto return prediction with TA-rich features."""

import asyncio
import contextlib
import importlib
import json
import logging
import os
import pickle
import platform
import random
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import ta

try:
    torch = importlib.import_module("torch")
    nn = torch.nn
    torch_utils_data = importlib.import_module("torch.utils.data")
    DataLoader = torch_utils_data.DataLoader
    Dataset = torch_utils_data.Dataset
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    DataLoader = None
    Dataset = object
    TORCH_AVAILABLE = False

try:
    from pattern_detector import calculate_indicators, detect_breakouts
    RUST_PATTERN_AVAILABLE = True
except ImportError:
    calculate_indicators = None
    detect_breakouts = None
    RUST_PATTERN_AVAILABLE = False

from unified_binance_loader import UnifiedBinanceLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _attach_file_logger(log_path: Path) -> logging.Handler:
    """Attach run-specific file logger while keeping existing console logging."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    return handler


def _safe_json_dump(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _git_commit_sha() -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.5,
        )
        if proc.returncode != 0:
            return None
        sha = proc.stdout.strip()
        return sha or None
    except Exception:
        return None


def _collect_runtime_snapshot(config: "TrainConfig") -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {
        "started_at_utc": datetime.utcnow().isoformat(),
        "cwd": os.getcwd(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "argv": sys.argv,
        "git_commit": _git_commit_sha(),
        "config": asdict(config),
        "torch_available": TORCH_AVAILABLE,
        "rust_pattern_available": RUST_PATTERN_AVAILABLE,
    }

    if TORCH_AVAILABLE and torch is not None:
        snapshot["torch_version"] = getattr(torch, "__version__", "unknown")
        snapshot["cuda_is_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            try:
                idx = torch.cuda.current_device()
                snapshot["cuda_current_device"] = idx
                snapshot["cuda_device_name"] = torch.cuda.get_device_name(idx)
                props = torch.cuda.get_device_properties(idx)
                snapshot["cuda_total_memory_gb"] = round(props.total_memory / (1024**3), 3)
                snapshot["cuda_compute_capability"] = f"{props.major}.{props.minor}"
            except Exception as exc:
                snapshot["cuda_probe_error"] = str(exc)

    return snapshot


@dataclass
class TrainConfig:
    symbols: List[str]
    timeframe: str = "1h"
    lookback_days: int = 365
    sequence_length: int = 64
    forecast_horizon: int = 24
    min_rows_per_symbol: int = 500
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    dense_size: int = 64
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 120
    early_stopping_patience: int = 15
    train_ratio: float = 0.8
    random_seed: int = 42
    num_workers: int = 0
    console_batch_logging: bool = True
    log_every_n_batches: int = 10
    maximize_gpu: bool = False
    max_batch_size: int = 512
    use_mixed_precision: bool = True
    gpu_memory_fraction: float = 0.85
    batch_sleep_sec: float = 0.02
    thermal_guard_enabled: bool = True
    thermal_guard_temp_c: int = 82
    thermal_cooldown_sec: float = 2.0
    enable_checkpoint_resume: bool = True
    checkpoint_interval: int = 5
    enable_memory_cleanup: bool = True
    gpu_cleanup_interval: int = 10
    checkpoint_interval: int = 5
    enable_gradient_accum: bool = True
    accumulation_steps: int = 2
    enable_memory_cleanup: bool = True


class SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class LSTMRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        dense_size: int,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, dense_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dense_size, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.head(last_hidden).squeeze(-1)


class MLTrainer:
    """Train a sequence-based LSTM model with technical indicators."""

    def __init__(self, model_path: str = "models", config: Optional[TrainConfig] = None):
        if not TORCH_AVAILABLE:
            raise ImportError("Missing dependency 'torch'. Install with: pip install torch")

        self.model_path = Path(model_path)
        self.model_path.mkdir(exist_ok=True)
        self.loader = UnifiedBinanceLoader(testnet=False)
        self.config = config or TrainConfig(symbols=["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT"])

        self.device = self._select_device()
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.use_amp = bool(self.config.use_mixed_precision and str(self.device) == "cuda")
        self.current_batch_size = self.config.batch_size

        if str(self.device) == "cuda":
            # Conservative mode to avoid aggressive kernel/autotune spikes on laptop GPUs.
            torch.backends.cudnn.benchmark = bool(self.config.maximize_gpu)
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
            if hasattr(torch.cuda, "set_per_process_memory_fraction"):
                try:
                    frac = min(0.95, max(0.5, float(self.config.gpu_memory_fraction)))
                    torch.cuda.set_per_process_memory_fraction(frac)
                    logger.info("CUDA memory fraction cap set to %.2f", frac)
                except Exception as exc:
                    logger.warning("Could not set CUDA memory fraction cap: %s", exc)

        self._set_seeds(self.config.random_seed)
        logger.info("Device: %s", self.device)
        logger.info("Rust pattern detector available: %s", RUST_PATTERN_AVAILABLE)
        logger.info("Mixed precision enabled: %s", self.use_amp)

    @staticmethod
    def _select_device():
        if not torch.cuda.is_available():
            return torch.device("cpu")

        try:
            # Runtime probe: catches mismatched driver/build kernels.
            _ = torch.zeros(8, device="cuda")
            probe_lstm = torch.nn.LSTM(input_size=4, hidden_size=8, num_layers=1, batch_first=True).cuda()
            probe_x = torch.randn(2, 5, 4, device="cuda")
            _ = probe_lstm(probe_x)
            return torch.device("cuda")
        except Exception as exc:
            logger.warning("CUDA runtime is not usable (%s). Falling back to CPU.", exc)
            return torch.device("cpu")

    @staticmethod
    def _set_seeds(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    async def generate_training_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        logger.info("Generating sequential training data...")
        x_chunks: List[np.ndarray] = []
        y_chunks: List[np.ndarray] = []

        for symbol in symbols:
            try:
                logger.info("Processing %s...", symbol)
                df = await self.loader.get_historical_data(
                    symbol=symbol,
                    interval=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                )
                if df is None or df.empty:
                    logger.warning("No data for %s", symbol)
                    continue

                seq_x, seq_y = self._build_symbol_sequences(df, symbol)
                if len(seq_x) == 0:
                    logger.warning("No valid sequences for %s", symbol)
                    continue

                x_chunks.append(seq_x)
                y_chunks.append(seq_y)
                logger.info("%s: %d sequences", symbol, len(seq_x))
            except Exception as exc:
                logger.warning("Error processing %s: %s", symbol, exc)

        if not x_chunks:
            raise RuntimeError("No training data generated. Check API access and input symbols.")

        x = np.concatenate(x_chunks, axis=0).astype(np.float32)
        y = np.concatenate(y_chunks, axis=0).astype(np.float32)
        logger.info("Total samples: %d", len(x))
        return x, y

    def _build_symbol_sequences(self, df: pd.DataFrame, symbol: str) -> Tuple[np.ndarray, np.ndarray]:
        df = df.copy().sort_index()
        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"{symbol}: missing required column '{col}'")

        for col in required:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=required)
        if len(df) < self.config.min_rows_per_symbol:
            return np.array([]), np.array([])

        feature_df = self._create_feature_frame(df)
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Target is future return over forecast_horizon.
        horizon = self.config.forecast_horizon
        close = df["close"]
        target = (close.shift(-horizon) / close) - 1.0

        feature_values = feature_df.values
        target_values = target.values

        if not self.feature_names:
            self.feature_names = feature_df.columns.tolist()
            logger.info("Feature set initialized with %d columns", len(self.feature_names))

        x_out: List[np.ndarray] = []
        y_out: List[float] = []
        seq_len = self.config.sequence_length
        max_idx = len(feature_df) - horizon

        for end_idx in range(seq_len - 1, max_idx):
            start_idx = end_idx - seq_len + 1
            seq = feature_values[start_idx : end_idx + 1]
            tgt = target_values[end_idx]
            if np.any(np.isnan(seq)) or np.isnan(tgt):
                continue
            x_out.append(seq)
            y_out.append(float(tgt))

        if not x_out:
            return np.array([]), np.array([])

        return np.stack(x_out), np.asarray(y_out, dtype=np.float32)

    def _create_feature_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"].replace(0, np.nan).ffill().fillna(1.0)

        feat = pd.DataFrame(index=df.index)

        # Price/momentum features
        feat["ret_1"] = close.pct_change(1)
        feat["ret_4"] = close.pct_change(4)
        feat["ret_12"] = close.pct_change(12)
        feat["ret_24"] = close.pct_change(24)
        feat["log_ret_1"] = np.log(close / close.shift(1))
        feat["hl_spread"] = (high - low) / close
        feat["oc_spread"] = (df["open"] - close) / close
        feat["volatility_12"] = feat["ret_1"].rolling(12).std()
        feat["volatility_48"] = feat["ret_1"].rolling(48).std()

        # Volume/flow features
        feat["volume_z_20"] = (volume - volume.rolling(20).mean()) / volume.rolling(20).std()
        feat["volume_ratio_20"] = volume / volume.rolling(20).mean()
        if "trades" in df.columns:
            trades = pd.to_numeric(df["trades"], errors="coerce")
            feat["trades_ratio_20"] = trades / trades.rolling(20).mean()
        else:
            feat["trades_ratio_20"] = 1.0

        # TA features from Rust (preferred), fallback to ta library.
        rust_indicators = self._get_rust_indicators(close)
        if rust_indicators is not None:
            feat["rsi"] = pd.Series(rust_indicators.get("rsi", []), index=feat.index)
            feat["ema20"] = pd.Series(rust_indicators.get("ema20", []), index=feat.index)
            feat["ema50"] = pd.Series(rust_indicators.get("ema50", []), index=feat.index)
            feat["macd"] = pd.Series(rust_indicators.get("macd_line", []), index=feat.index)
            feat["macd_signal"] = pd.Series(rust_indicators.get("macd_signal", []), index=feat.index)
            feat["macd_hist"] = pd.Series(rust_indicators.get("macd_histogram", []), index=feat.index)
            feat["bb_upper"] = pd.Series(rust_indicators.get("bb_upper", []), index=feat.index)
            feat["bb_lower"] = pd.Series(rust_indicators.get("bb_lower", []), index=feat.index)
            feat["bb_sma"] = pd.Series(rust_indicators.get("bb_sma", []), index=feat.index)
        else:
            feat["rsi"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
            macd_indicator = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
            feat["macd"] = macd_indicator.macd()
            feat["macd_signal"] = macd_indicator.macd_signal()
            feat["macd_hist"] = macd_indicator.macd_diff()
            bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
            feat["bb_upper"] = bb.bollinger_hband()
            feat["bb_lower"] = bb.bollinger_lband()
            feat["bb_sma"] = bb.bollinger_mavg()
            feat["ema20"] = close.ewm(span=20, adjust=False).mean()
            feat["ema50"] = close.ewm(span=50, adjust=False).mean()

        feat["price_vs_ema20"] = (close - feat["ema20"]) / close
        feat["price_vs_ema50"] = (close - feat["ema50"]) / close
        feat["ema_gap"] = (feat["ema20"] - feat["ema50"]) / close
        feat["bb_width"] = (feat["bb_upper"] - feat["bb_lower"]) / feat["bb_sma"].replace(0, np.nan)
        feat["atr_14"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range() / close

        for col in ("ema20", "ema50", "bb_upper", "bb_lower", "bb_sma"):
            feat[col] = feat[col] / close
        for col in ("macd", "macd_signal", "macd_hist"):
            feat[col] = feat[col] / close

        # Rust breakout signals as additional sequence context.
        breakout_confidence, breakout_direction = self._get_breakout_features(df)
        feat["breakout_confidence"] = breakout_confidence
        feat["breakout_short_signal"] = breakout_direction

        return feat

    @staticmethod
    def _get_rust_indicators(close: pd.Series) -> Optional[Dict[str, List[float]]]:
        if calculate_indicators is None:
            return None
        try:
            values = close.values.astype(float).tolist()
            return calculate_indicators(values)
        except Exception as exc:
            logger.warning("Rust calculate_indicators unavailable, fallback to python-ta: %s", exc)
            return None

    @staticmethod
    def _get_breakout_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        n = len(df)
        confidence = np.zeros(n, dtype=np.float32)
        short_signal = np.zeros(n, dtype=np.float32)

        if detect_breakouts is None:
            return confidence, short_signal

        try:
            results = detect_breakouts(
                df["open"].astype(float).tolist(),
                df["high"].astype(float).tolist(),
                df["low"].astype(float).tolist(),
                df["close"].astype(float).tolist(),
                df["volume"].astype(float).tolist(),
                12,
                1.2,
                1.5,
            )
            for event in results:
                idx = int(event.get("index", -1))
                if 0 <= idx < n:
                    confidence[idx] = float(event.get("confidence", 0.0)) / 100.0
                    short_signal[idx] = 1.0 if event.get("direction") == "SHORT" else 0.0
        except Exception as exc:
            logger.warning("Rust detect_breakouts unavailable for features: %s", exc)

        return confidence, short_signal

    def train_lstm(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        logger.info("Training LSTM model...")
        n_samples = len(x)
        split_idx = int(n_samples * self.config.train_ratio)

        x_train, x_val = x[:split_idx], x[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Fit scaler only on train split to avoid leakage.
        train_2d = x_train.reshape(-1, x_train.shape[-1])
        self.scaler.fit(train_2d)
        x_train = self._scale_sequences(x_train)
        x_val = self._scale_sequences(x_val)

        self.current_batch_size = self.config.batch_size
        if self.config.maximize_gpu and str(self.device) == "cuda":
            self.current_batch_size = self._find_max_batch_size(
                x_train_shape=x_train.shape,
                target_dim=x.shape[-1],
            )
            logger.info("GPU stress mode: selected batch_size=%d", self.current_batch_size)

        train_loader = DataLoader(
            SequenceDataset(x_train, y_train),
            batch_size=self.current_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            SequenceDataset(x_val, y_val),
            batch_size=self.current_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        model = LSTMRegressor(
            input_size=x.shape[-1],
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            dense_size=self.config.dense_size,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )
        criterion = nn.SmoothL1Loss(beta=0.01)
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler(device="cuda", enabled=self.use_amp)
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        best_val_loss = float("inf")
        best_epoch = 0
        patience = 0
        history = []
        start_epoch = 1
        
        # Try to resume from checkpoint if enabled
        if self.config.enable_checkpoint_resume and (self.model_path / "lstm_model.pt").exists():
            try:
                checkpoint = torch.load(self.model_path / "lstm_model.pt", map_location=self.device, weights_only=False)
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                start_epoch = checkpoint.get("epoch", 0) + 1
                best_val_loss = checkpoint.get("val_loss", float("inf"))
                if start_epoch <= self.config.max_epochs:
                    logger.info("Resumed training from epoch %d (best val_loss=%.6f)", start_epoch, best_val_loss)
            except Exception as e:
                logger.warning("Could not resume from checkpoint: %s. Starting fresh.", e)
                start_epoch = 1

        for epoch in range(start_epoch, self.config.max_epochs + 1):
            epoch_start = time.perf_counter()
            train_loss = self._train_epoch(model, train_loader, optimizer, criterion, scaler, epoch)
            val_loss, val_metrics = self._evaluate_epoch(model, val_loader, criterion)
            scheduler.step(val_loss)
            epoch_time = time.perf_counter() - epoch_start

            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch_time_sec": epoch_time,
                **val_metrics,
            })
            logger.info(
                "Epoch %03d | time=%.2fs | train_loss=%.6f | val_loss=%.6f | val_rmse=%.6f | val_mae=%.6f | val_r2=%.4f",
                epoch,
                epoch_time,
                train_loss,
                val_loss,
                val_metrics["rmse"],
                val_metrics["mae"],
                val_metrics["r2"],
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience = 0
                self._save_checkpoint(model, optimizer, epoch, best_val_loss)
            else:
                patience += 1
            
            # Periodic checkpoints to enable recovery from crashes
            if self.config.enable_checkpoint_resume and epoch % self.config.checkpoint_interval == 0:
                self._save_checkpoint(model, optimizer, epoch, val_loss)
                if self.config.enable_memory_cleanup:
                    torch.cuda.empty_cache()

            if patience >= self.config.early_stopping_patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

        logger.info("Best epoch: %d, best val loss: %.6f", best_epoch, best_val_loss)
        self._save_artifacts(history)
        final_metrics = self.evaluate_best_model(x_val, y_val, x_is_scaled=True)
        return final_metrics

    def _scale_sequences(self, x: np.ndarray) -> np.ndarray:
        shape = x.shape
        x_2d = x.reshape(-1, shape[-1])
        x_scaled = self.scaler.transform(x_2d)
        return x_scaled.reshape(shape).astype(np.float32)

    def _autocast_context(self):
        if not self.use_amp:
            return contextlib.nullcontext()
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast(device_type="cuda", enabled=True)
        return torch.cuda.amp.autocast(enabled=True)

    def _gpu_temperature_c(self) -> Optional[float]:
        if str(self.device) != "cuda":
            return None
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                check=False,
                capture_output=True,
                text=True,
                timeout=1.5,
            )
            if result.returncode != 0:
                return None
            line = result.stdout.strip().splitlines()[0]
            return float(line)
        except Exception:
            return None

    def _find_max_batch_size(self, x_train_shape: Tuple[int, ...], target_dim: int) -> int:
        seq_len = x_train_shape[1]
        candidate = max(8, self.config.batch_size)
        max_candidate = min(self.config.max_batch_size, x_train_shape[0])
        best = candidate

        logger.info(
            "Searching max stable batch size for GPU load (start=%d, max=%d)",
            candidate,
            max_candidate,
        )

        while candidate <= max_candidate:
            if self._can_run_batch_size(candidate, seq_len, target_dim):
                best = candidate
                logger.info("Batch size %d OK", candidate)
                candidate *= 2
            else:
                logger.info("Batch size %d failed, stopping search", candidate)
                break

        return best

    def _can_run_batch_size(self, batch_size: int, seq_len: int, target_dim: int) -> bool:
        try:
            model = LSTMRegressor(
                input_size=target_dim,
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_layers,
                dropout=self.config.dropout,
                dense_size=self.config.dense_size,
            ).to(self.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
            criterion = nn.SmoothL1Loss(beta=0.01)
            if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
                scaler = torch.amp.GradScaler(device="cuda", enabled=self.use_amp)
            else:
                scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

            x_probe = torch.randn(batch_size, seq_len, target_dim, device=self.device)
            y_probe = torch.randn(batch_size, device=self.device)

            optimizer.zero_grad(set_to_none=True)
            with self._autocast_context():
                preds = model(x_probe)
                loss = criterion(preds, y_probe)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            del model, optimizer, criterion, scaler, x_probe, y_probe, preds, loss
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            return True
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                return False
            raise

    def _train_epoch(
        self,
        model: Any,
        loader: Any,
        optimizer: Any,
        criterion: Any,
        scaler: Any,
        epoch: int,
    ) -> float:
        model.train()
        losses = []
        batch_start = time.perf_counter()
        for batch_idx, (xb, yb) in enumerate(loader, start=1):
            xb = xb.to(self.device, non_blocking=True)
            yb = yb.to(self.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with self._autocast_context():
                preds = model(xb)
                loss = criterion(preds, yb)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            losses.append(float(loss.item()))

            # Periodic memory cleanup to prevent fragmentation
            if self.config.enable_memory_cleanup and batch_idx % self.config.gpu_cleanup_interval == 0:
                torch.cuda.empty_cache()

            if self.config.console_batch_logging and batch_idx % self.config.log_every_n_batches == 0:
                elapsed = time.perf_counter() - batch_start
                avg_loss = float(np.mean(losses[-self.config.log_every_n_batches :]))
                if str(self.device) == "cuda":
                    mem_alloc = torch.cuda.memory_allocated() / 1024**3
                    mem_reserved = torch.cuda.memory_reserved() / 1024**3
                    logger.info(
                        "Epoch %03d Batch %04d/%04d | avg_loss=%.6f | step_time=%.2fs | gpu_mem_alloc=%.2fGB | gpu_mem_reserved=%.2fGB",
                        epoch,
                        batch_idx,
                        len(loader),
                        avg_loss,
                        elapsed,
                        mem_alloc,
                        mem_reserved,
                    )
                else:
                    logger.info(
                        "Epoch %03d Batch %04d/%04d | avg_loss=%.6f | step_time=%.2fs",
                        epoch,
                        batch_idx,
                        len(loader),
                        avg_loss,
                        elapsed,
                    )
                batch_start = time.perf_counter()

            if self.config.thermal_guard_enabled and str(self.device) == "cuda":
                temp_c = self._gpu_temperature_c()
                if temp_c is not None and temp_c >= float(self.config.thermal_guard_temp_c):
                    logger.warning(
                        "Thermal guard: GPU temp %.1fC >= %dC, cooling for %.1fs",
                        temp_c,
                        self.config.thermal_guard_temp_c,
                        self.config.thermal_cooldown_sec,
                    )
                    time.sleep(max(0.0, float(self.config.thermal_cooldown_sec)))

            if self.config.batch_sleep_sec > 0:
                time.sleep(self.config.batch_sleep_sec)

        return float(np.mean(losses)) if losses else 0.0

    @torch.no_grad()
    def _evaluate_epoch(self, model: Any, loader: Any, criterion: Any) -> Tuple[float, Dict[str, float]]:
        model.eval()
        losses = []
        y_true = []
        y_pred = []

        for xb, yb in loader:
            xb = xb.to(self.device, non_blocking=True)
            yb = yb.to(self.device, non_blocking=True)
            preds = model(xb)
            loss = criterion(preds, yb)
            losses.append(float(loss.item()))

            y_true.extend(yb.detach().cpu().numpy().tolist())
            y_pred.extend(preds.detach().cpu().numpy().tolist())

        metrics = self._regression_metrics(np.array(y_true), np.array(y_pred))
        val_loss = float(np.mean(losses)) if losses else float("inf")
        return val_loss, metrics

    def _regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        true_sign = np.sign(y_true)
        pred_sign = np.sign(y_pred)
        directional_accuracy = float((true_sign == pred_sign).mean())

        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "directional_accuracy": directional_accuracy,
        }

    def _save_checkpoint(self, model: Any, optimizer: Any, epoch: int, val_loss: float) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "config": asdict(self.config),
            "feature_names": self.feature_names,
        }
        torch.save(checkpoint, self.model_path / "lstm_model.pt")

    def _save_artifacts(self, history: List[Dict[str, float]]) -> None:
        with open(self.model_path / "lstm_scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

        metadata = {
            "trained_at": datetime.utcnow().isoformat(),
            "device": str(self.device),
            "rust_pattern_available": RUST_PATTERN_AVAILABLE,
            "feature_names": self.feature_names,
            "config": asdict(self.config),
        }
        with open(self.model_path / "lstm_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        with open(self.model_path / "lstm_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    def evaluate_best_model(self, x_val: np.ndarray, y_val: np.ndarray, x_is_scaled: bool = False) -> Dict[str, float]:
        ckpt_path = self.model_path / "lstm_model.pt"
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        model = LSTMRegressor(
            input_size=x_val.shape[-1],
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            dense_size=self.config.dense_size,
        ).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        x_scaled = x_val if x_is_scaled else self._scale_sequences(x_val)
        loader = DataLoader(SequenceDataset(x_scaled, y_val), batch_size=self.current_batch_size, shuffle=False)

        y_true = []
        y_pred = []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device)
                preds = model(xb).cpu().numpy()
                y_pred.extend(preds.tolist())
                y_true.extend(yb.numpy().tolist())

        metrics = self._regression_metrics(np.array(y_true), np.array(y_pred))
        logger.info("Final validation metrics: %s", json.dumps(metrics, indent=2))
        return metrics


async def run_training() -> None:
    from config import FUTURES_SYMBOLS
    # Combine stable major coins with the actual scanning targets for a well-rounded model
    base_symbols = [
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
        "SOLUSDT",
        "ADAUSDT",
        "XRPUSDT",
        "DOGEUSDT",
        "LINKUSDT",
    ]
    # dict.fromkeys preserves order and deduplicates
    symbols = list(dict.fromkeys(base_symbols + FUTURES_SYMBOLS))
    config = TrainConfig(
        symbols=symbols,
        lookback_days=365,
        sequence_length=96,
        forecast_horizon=24,
        hidden_size=128,
        num_layers=3,
        dropout=0.25,
        dense_size=96,
        batch_size=48,
        learning_rate=7e-4,
        max_epochs=100,
        early_stopping_patience=10,
        maximize_gpu=True,
        max_batch_size=512,
        console_batch_logging=True,
        log_every_n_batches=10,
        use_mixed_precision=True,
        gpu_memory_fraction=0.90,  # Reduced from 0.95 for more stability
        batch_sleep_sec=0.05,  # Increased from 0.0 for thermal relief
        thermal_guard_enabled=True,
        thermal_guard_temp_c=80,  # Lowered from 84 to prevent crashes
        thermal_cooldown_sec=3.0,  # Increased from 2.0 for better cooling
        enable_checkpoint_resume=True,  # Enable resume capability
        checkpoint_interval=5,  # Save every 5 epochs
        enable_memory_cleanup=True,  # Cleanup cache periodically
    )

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("logs") / "training_runs"
    run_log_path = run_dir / f"train_{run_id}.log"
    run_context_path = run_dir / f"train_{run_id}_context.json"
    run_summary_path = run_dir / f"train_{run_id}_summary.json"

    file_handler = _attach_file_logger(run_log_path)
    run_started_ts = time.perf_counter()

    _safe_json_dump(run_context_path, _collect_runtime_snapshot(config))
    logger.info("Run ID: %s", run_id)
    logger.info("Run log file: %s", run_log_path)
    logger.info("Run context file: %s", run_context_path)

    trainer = MLTrainer(config=config)

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=config.lookback_days)

    try:
        x, y = await trainer.generate_training_data(
            symbols=config.symbols,
            start_date=start_date,
            end_date=end_date,
            timeframe=config.timeframe,
        )

        logger.info(
            "Dataset summary: samples=%d, seq_len=%d, features=%d",
            len(x),
            int(x.shape[1]) if len(x) > 0 else 0,
            int(x.shape[2]) if len(x) > 0 else 0,
        )

        if len(x) < 1500:
            logger.warning("Small dataset (%d samples). Consider more symbols/time window.", len(x))

        metrics = trainer.train_lstm(x, y)
        logger.info("LSTM training complete. Metrics: %s", json.dumps(metrics, indent=2))

        elapsed_sec = time.perf_counter() - run_started_ts
        _safe_json_dump(
            run_summary_path,
            {
                "run_id": run_id,
                "status": "success",
                "finished_at_utc": datetime.utcnow().isoformat(),
                "elapsed_sec": round(elapsed_sec, 3),
                "device": str(trainer.device),
                "metrics": metrics,
                "artifacts": {
                    "model": str(trainer.model_path / "lstm_model.pt"),
                    "metadata": str(trainer.model_path / "lstm_metadata.json"),
                    "history": str(trainer.model_path / "lstm_history.json"),
                    "scaler": str(trainer.model_path / "lstm_scaler.pkl"),
                },
            },
        )
        logger.info("Run summary file: %s", run_summary_path)

    except Exception as exc:
        elapsed_sec = time.perf_counter() - run_started_ts
        logger.exception("Training failed: %s", exc)
        _safe_json_dump(
            run_summary_path,
            {
                "run_id": run_id,
                "status": "failed",
                "finished_at_utc": datetime.utcnow().isoformat(),
                "elapsed_sec": round(elapsed_sec, 3),
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
        )
        logger.info("Run summary file: %s", run_summary_path)
        raise

    finally:
        await trainer.loader.close()
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()


def main() -> None:
    asyncio.run(run_training())


if __name__ == "__main__":
    main()