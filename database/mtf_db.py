#!/usr/bin/env python3
"""
MTF Pattern Database - PostgreSQL backend
"""
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from psycopg2.pool import ThreadedConnectionPool
from contextlib import contextmanager
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)


class MTFDatabase:
    """PostgreSQL database для зберігання MTF паттернів"""
    
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 5432,
        database: str = 'trading_patterns',
        user: str = 'trader',
        password: str = 'trading123',
        min_conn: int = 1,
        max_conn: int = 10
    ):
        """
        Args:
            host: PostgreSQL host
            port: PostgreSQL port
            database: Database name
            user: Username
            password: Password
            min_conn: Minimum connections in pool
            max_conn: Maximum connections in pool
        """
        self.pool = None
        self.conn_params = {
            'host': host,
            'port': port,
            'database': database,
            'user': user,
            'password': password
        }
        self.min_conn = min_conn
        self.max_conn = max_conn
        
        # Initialize connection pool
        self._init_pool()
    
    def _init_pool(self):
        """Ініціалізує connection pool"""
        try:
            self.pool = ThreadedConnectionPool(
                self.min_conn,
                self.max_conn,
                **self.conn_params
            )
            logger.info(f"✅ Connected to PostgreSQL: {self.conn_params['database']}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to PostgreSQL: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Context manager для connection з pool"""
        conn = self.pool.getconn()
        try:
            yield conn
        finally:
            self.pool.putconn(conn)
    
    def add_pattern(
        self,
        symbol: str,
        pattern_type: str,
        direction: str,
        timeframe: str,
        confidence: float,
        price: float,
        timestamp: datetime,
        strength: str,
        mtf_score: Optional[float] = None
    ) -> int:
        """
        Додає паттерн в базу
        
        Returns:
            pattern_id
        """
        query = """
            INSERT INTO patterns 
            (symbol, pattern_type, direction, timeframe, confidence, 
             price, timestamp, strength, mtf_score)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING pattern_id
        """
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (
                    symbol, pattern_type, direction, timeframe,
                    confidence, price, timestamp, strength, mtf_score
                ))
                pattern_id = cur.fetchone()[0]
                conn.commit()
                
                # Update pattern stats
                self._update_pattern_stats(conn, symbol, pattern_type, direction, timeframe)
                
                return pattern_id
    
    def _update_pattern_stats(self, conn, symbol, pattern_type, direction, timeframe):
        """Оновлює статистику паттерну"""
        query = """
            INSERT INTO pattern_stats 
            (symbol, pattern_type, direction, timeframe, occurrence_count, avg_confidence, last_seen)
            VALUES (%s, %s, %s, %s, 1, 
                    (SELECT confidence FROM patterns 
                     WHERE symbol=%s AND pattern_type=%s AND direction=%s AND timeframe=%s
                     ORDER BY timestamp DESC LIMIT 1),
                    CURRENT_TIMESTAMP)
            ON CONFLICT (symbol, pattern_type, direction, timeframe)
            DO UPDATE SET
                occurrence_count = pattern_stats.occurrence_count + 1,
                avg_confidence = (
                    SELECT AVG(confidence) 
                    FROM patterns 
                    WHERE symbol=%s AND pattern_type=%s AND direction=%s AND timeframe=%s
                ),
                last_seen = CURRENT_TIMESTAMP
        """
        
        with conn.cursor() as cur:
            cur.execute(query, (
                symbol, pattern_type, direction, timeframe,
                symbol, pattern_type, direction, timeframe,
                symbol, pattern_type, direction, timeframe
            ))
            conn.commit()
    
    def add_mtf_signal(
        self,
        symbol: str,
        mtf_score: float,
        confluence_pct: float,
        dominant_direction: str,
        timeframes_count: int,
        signal_data: Dict[str, Any]
    ) -> int:
        """Додає MTF сигнал"""
        query = """
            INSERT INTO mtf_signals
            (symbol, mtf_score, confluence_pct, dominant_direction, 
             timeframes_count, signal_data)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING signal_id
        """
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (
                    symbol, mtf_score, confluence_pct, dominant_direction,
                    timeframes_count, Json(signal_data)
                ))
                signal_id = cur.fetchone()[0]
                conn.commit()
                return signal_id
    
    def save_chart_snapshot(
        self,
        symbol: str,
        timeframe: str,
        pattern_id: int,
        candles_data: List[Dict],
        indicators_data: Dict,
        pattern_coordinates: Dict
    ) -> int:
        """Зберігає дані для відтворення графіку"""
        query = """
            INSERT INTO pattern_chart_snapshots
            (symbol, timeframe, pattern_id, candles_data, 
             indicators_data, pattern_coordinates)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING snapshot_id
        """
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (
                    symbol, timeframe, pattern_id,
                    Json(candles_data), Json(indicators_data), Json(pattern_coordinates)
                ))
                snapshot_id = cur.fetchone()[0]
                conn.commit()
                return snapshot_id
    
    def get_recent_patterns(self, days: int = 7, limit: int = 100) -> List[Dict]:
        """Отримує останні паттерни"""
        query = """
            SELECT * FROM recent_patterns
            LIMIT %s
        """
        
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (limit,))
                return [dict(row) for row in cur.fetchall()]
    
    def get_top_recurring_patterns(self, min_count: int = 5, limit: int = 50) -> List[Dict]:
        """Отримує найчастіші паттерни"""
        query = """
            SELECT * FROM top_recurring_patterns
            WHERE occurrence_count >= %s
            LIMIT %s
        """
        
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (min_count, limit))
                return [dict(row) for row in cur.fetchall()]
    
    def get_top_mtf_signals(self, min_confluence: float = 60.0, limit: int = 20) -> List[Dict]:
        """Отримує топ MTF сигнали"""
        query = """
            SELECT * FROM top_mtf_signals
            WHERE confluence_pct >= %s
            LIMIT %s
        """
        
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (min_confluence, limit))
                return [dict(row) for row in cur.fetchall()]
    
    def get_pattern_stats(self, symbol: str = None) -> List[Dict]:
        """Отримує статистику паттернів"""
        if symbol:
            query = """
                SELECT * FROM pattern_stats
                WHERE symbol = %s
                ORDER BY occurrence_count DESC, avg_confidence DESC
            """
            params = (symbol,)
        else:
            query = """
                SELECT * FROM pattern_stats
                ORDER BY occurrence_count DESC, avg_confidence DESC
                LIMIT 100
            """
            params = ()
        
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                return [dict(row) for row in cur.fetchall()]
    
    def get_chart_snapshot(self, pattern_id: int) -> Optional[Dict]:
        """Отримує збережені дані графіку"""
        query = """
            SELECT * FROM pattern_chart_snapshots
            WHERE pattern_id = %s
            ORDER BY created_at DESC
            LIMIT 1
        """
        
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (pattern_id,))
                row = cur.fetchone()
                return dict(row) if row else None
    
    def mark_telegram_notified(self, signal_id: int):
        """Позначає що Telegram notification відправлено"""
        query = """
            UPDATE mtf_signals
            SET telegram_notified = TRUE
            WHERE signal_id = %s
        """
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (signal_id,))
                conn.commit()
    
    def close(self):
        """Закриває всі з'єднання"""
        if self.pool:
            self.pool.closeall()
            logger.info("PostgreSQL connections closed")


def init_database(schema_file: str = 'schema_mtf_patterns.sql'):
    """
    Ініціалізує базу даних зі схемою
    
    Args:
        schema_file: Шлях до SQL схеми
    """
    # Read schema
    schema_path = Path(schema_file)
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_file}")
    
    schema_sql = schema_path.read_text()
    
    # Connect and execute
    try:
        # First connect to postgres db to create our database
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            database='postgres',
            user='trader',
            password='trading123'
        )
        conn.autocommit = True
        
        with conn.cursor() as cur:
            # Create database if not exists
            cur.execute("""
                SELECT 1 FROM pg_database WHERE datname = 'trading_patterns'
            """)
            
            if not cur.fetchone():
                cur.execute("CREATE DATABASE trading_patterns")
                logger.info("✅ Created database: trading_patterns")
            else:
                logger.info("Database trading_patterns already exists")
        
        conn.close()
        
        # Now connect to our database and create schema
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            database='trading_patterns',
            user='trader',
            password='trading123'
        )
        
        with conn.cursor() as cur:
            cur.execute(schema_sql)
            conn.commit()
            logger.info("✅ Database schema created successfully")
        
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {e}")
        raise


def migrate_from_json(json_file: str = 'pattern_history.json'):
    """
    Мігрує дані з JSON в PostgreSQL
    
    Args:
        json_file: Шлях до JSON файлу
    """
    json_path = Path(json_file)
    if not json_path.exists():
        logger.warning(f"JSON file not found: {json_file}")
        return 0
    
    # Load JSON
    with open(json_path) as f:
        data = json.load(f)
    
    db = MTFDatabase()
    migrated = 0
    
    try:
        for key, pattern_data in data.items():
            # Parse key: symbol_patterntype_direction_timeframe
            parts = key.split('_')
            if len(parts) < 4:
                continue  # Old format without timeframe
            
            symbol = parts[0]
            direction = parts[-2]
            timeframe = parts[-1]
            pattern_type = '_'.join(parts[1:-2])
            
            # Add pattern
            db.add_pattern(
                symbol=symbol,
                pattern_type=pattern_type,
                direction=direction,
                timeframe=timeframe,
                confidence=pattern_data.get('confidence', 0.0),
                price=pattern_data.get('price', 0.0),
                timestamp=datetime.fromisoformat(pattern_data['last_seen']),
                strength='MEDIUM',  # Default
                mtf_score=None
            )
            
            migrated += 1
        
        logger.info(f"✅ Migrated {migrated} patterns from JSON to PostgreSQL")
        return migrated
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    # Test database
    logging.basicConfig(level=logging.INFO)
    
    print("Initializing database...")
    init_database()
    
    print("\nTesting connection...")
    db = MTFDatabase()
    
    print("\nAdding test pattern...")
    pattern_id = db.add_pattern(
        symbol='BTCUSDT',
        pattern_type='Bullish Engulfing',
        direction='LONG',
        timeframe='1h',
        confidence=75.5,
        price=45000.0,
        timestamp=datetime.now(),
        strength='STRONG',
        mtf_score=80.0
    )
    print(f"Created pattern ID: {pattern_id}")
    
    print("\nRecent patterns:")
    patterns = db.get_recent_patterns(limit=5)
    for p in patterns:
        print(f"  {p['symbol']} - {p['pattern_type']} ({p['direction']}) - {p['confidence']:.1f}%")
    
    db.close()
    print("\n✅ Database test completed")
