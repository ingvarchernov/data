"""
Backtest Database Manager - зберігає результати в SQLite
Замінює генерацію окремих JSON файлів
"""
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class BacktestDB:
    def __init__(self, db_path: str = "database/backtests.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Ініціалізація таблиць"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS backtest_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbols TEXT NOT NULL,
                    timeframes TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    initial_capital REAL NOT NULL,
                    risk_per_trade REAL NOT NULL,
                    max_concurrent_trades INTEGER NOT NULL,
                    total_trades INTEGER NOT NULL,
                    winning_trades INTEGER NOT NULL,
                    losing_trades INTEGER NOT NULL,
                    win_rate REAL NOT NULL,
                    total_pnl REAL NOT NULL,
                    total_pnl_pct REAL NOT NULL,
                    final_capital REAL NOT NULL,
                    max_drawdown REAL NOT NULL,
                    profit_factor REAL,
                    avg_win REAL,
                    avg_loss REAL,
                    skipped_trades INTEGER DEFAULT 0,
                    notes TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS backtest_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    pattern_type TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_time TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    take_profit REAL NOT NULL,
                    exit_time TEXT,
                    exit_price REAL,
                    status TEXT NOT NULL,
                    pnl REAL NOT NULL,
                    pnl_pct REAL NOT NULL,
                    capital_before REAL NOT NULL,
                    capital_after REAL NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES backtest_runs(id)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_run_timestamp 
                ON backtest_runs(timestamp DESC)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trade_run 
                ON backtest_trades(run_id)
            """)
            
            conn.commit()
    
    def save_backtest(self, config: Dict, results: Dict, trades: List[Dict]) -> int:
        """Зберегти результати backtest в DB"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Зберегти run
            cursor.execute("""
                INSERT INTO backtest_runs (
                    timestamp, symbols, timeframes, start_date, end_date,
                    initial_capital, risk_per_trade, max_concurrent_trades,
                    total_trades, winning_trades, losing_trades, win_rate,
                    total_pnl, total_pnl_pct, final_capital, max_drawdown,
                    profit_factor, avg_win, avg_loss, skipped_trades, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                json.dumps(config.get('symbols', [])),
                json.dumps(config.get('timeframes', [])),
                config['start_date'],
                config['end_date'],
                config['initial_capital'],
                config['risk_per_trade'],
                config.get('max_concurrent_trades', 3),
                results['total_trades'],
                results['winning_trades'],
                results['losing_trades'],
                results['win_rate'],
                results['total_pnl'],
                results['total_pnl_pct'],
                results['final_capital'],
                results.get('max_drawdown', 0),
                results.get('profit_factor'),
                results.get('avg_win'),
                results.get('avg_loss'),
                results.get('skipped_trades', 0),
                config.get('notes')
            ))
            
            run_id = cursor.lastrowid
            
            # Зберегти trades
            for trade in trades:
                cursor.execute("""
                    INSERT INTO backtest_trades (
                        run_id, symbol, timeframe, pattern_type, direction,
                        entry_time, entry_price, stop_loss, take_profit,
                        exit_time, exit_price, status, pnl, pnl_pct,
                        capital_before, capital_after
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    run_id,
                    trade['symbol'],
                    trade['timeframe'],
                    trade['pattern_type'],
                    trade['direction'],
                    trade['entry_time'],
                    trade['entry_price'],
                    trade['stop_loss'],
                    trade['take_profit'],
                    trade.get('exit_time'),
                    trade.get('exit_price'),
                    trade['status'],
                    trade['pnl'],
                    trade['pnl_pct'],
                    trade['capital_before'],
                    trade['capital_after']
                ))
            
            conn.commit()
            logger.info(f"✅ Backtest saved to DB: run_id={run_id}, {len(trades)} trades")
            return run_id
    
    def get_recent_runs(self, limit: int = 10) -> List[Dict]:
        """Отримати останні backtest runs"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM backtest_runs 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_run_trades(self, run_id: int) -> List[Dict]:
        """Отримати всі трейди для конкретного run"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM backtest_trades 
                WHERE run_id = ? 
                ORDER BY entry_time
            """, (run_id,))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_best_runs(self, metric: str = 'total_pnl_pct', limit: int = 5) -> List[Dict]:
        """Отримати найкращі runs за метрикою"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(f"""
                SELECT * FROM backtest_runs 
                ORDER BY {metric} DESC 
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]
    
    def compare_runs(self, run_ids: List[int]) -> Dict:
        """Порівняти декілька runs"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            placeholders = ','.join('?' * len(run_ids))
            cursor = conn.execute(f"""
                SELECT * FROM backtest_runs 
                WHERE id IN ({placeholders})
            """, run_ids)
            return [dict(row) for row in cursor.fetchall()]
