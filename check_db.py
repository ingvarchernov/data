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
        tables = ['trading_signals', 'positions', 'trades', 'historical_data']
        for table in tables:
            result = conn.execute(text(f'SELECT COUNT(*) FROM {table}'))
            count = result.scalar()
            print(f'📊 {table}: {count} записів')
            
        result = conn.execute(text('SELECT symbol, side, entry_price, status FROM positions LIMIT 10'))
        positions = result.fetchall()
        if positions:
            print('\n📈 Відкриті позиції:')
            for pos in positions:
                print(f'  {pos.symbol} {pos.side} @ ${pos.entry_price} - {pos.status}')
                
except Exception as e:
    print(f'❌ Помилка: {e}')
