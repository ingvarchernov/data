import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
db_url = os.getenv('DATABASE_URL')
if not db_url:
    print('❌ DATABASE_URL не знайдено')
    exit(1)

engine = create_engine(db_url)

try:
    with engine.connect() as conn:
        # Перевіряємо які таблиці існують
        result = conn.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """))
        existing_tables = [row[0] for row in result.fetchall()]
        
        # Рахуємо записи в існуючих таблицях
        tables_to_check = ['positions', 'trades', 'historical_data', 'model_performance']
        for table in tables_to_check:
            if table in existing_tables:
                result = conn.execute(text(f'SELECT COUNT(*) FROM {table}'))
                count = result.scalar()
                print(f'📊 {table}: {count} записів')
            else:
                print(f'⚠️ {table}: таблиця не існує')
        
        # Останні позиції
        if 'positions' in existing_tables:
            result = conn.execute(text('''
                SELECT symbol, side, entry_price, status, entry_time 
                FROM positions 
                ORDER BY entry_time DESC 
                LIMIT 10
            '''))
            positions = result.fetchall()
            if positions:
                print('\n📈 Останні позиції:')
                for pos in positions:
                    print(f'  {pos.symbol} {pos.side} @ ${pos.entry_price} - {pos.status}')
                
except Exception as e:
    print(f'❌ Помилка: {e}')
