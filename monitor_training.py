#!/usr/bin/env python3
"""
📊 Моніторинг прогресу тренування
"""
import os
import glob
import pandas as pd
from datetime import datetime

def monitor_training():
    """Показує прогрес останнього тренування"""
    
    # Знаходимо останній CSV файл
    csv_files = glob.glob('models/classification_BTC/training_*.csv')
    
    if not csv_files:
        print("❌ Файли тренування не знайдено")
        return
    
    latest_csv = max(csv_files, key=os.path.getctime)
    
    print("="*80)
    print(f"📊 МОНІТОРИНГ ТРЕНУВАННЯ")
    print("="*80)
    print(f"\n📁 Файл: {latest_csv}")
    
    try:
        df = pd.read_csv(latest_csv)
        
        if len(df) == 0:
            print("⏳ Тренування ще не почалося...")
            return
        
        # Остання епоха
        last_epoch = len(df)
        last_row = df.iloc[-1]
        
        print(f"\n📈 Прогрес:")
        print(f"   Поточна епоха: {last_epoch}")
        print(f"   Train accuracy: {last_row['accuracy']:.4f} ({last_row['accuracy']*100:.2f}%)")
        print(f"   Val accuracy:   {last_row['val_accuracy']:.4f} ({last_row['val_accuracy']*100:.2f}%)")
        print(f"   Train loss:     {last_row['loss']:.4f}")
        print(f"   Val loss:       {last_row['val_loss']:.4f}")
        print(f"   Learning rate:  {last_row['learning_rate']:.2e}")
        
        # Найкраща val accuracy
        best_epoch = df['val_accuracy'].idxmax() + 1
        best_acc = df['val_accuracy'].max()
        
        print(f"\n🏆 Найкраща val accuracy:")
        print(f"   Епоха: {best_epoch}")
        print(f"   Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
        
        # Тренд (останні 5 epochs)
        if len(df) >= 5:
            recent_trend = df['val_accuracy'].tail(5)
            trend_change = recent_trend.iloc[-1] - recent_trend.iloc[0]
            
            print(f"\n📊 Тренд (останні 5 epochs):")
            if trend_change > 0.01:
                print(f"   📈 Зростання: +{trend_change*100:.2f}%")
            elif trend_change < -0.01:
                print(f"   📉 Падіння: {trend_change*100:.2f}%")
            else:
                print(f"   ➡️ Стабільний: {trend_change*100:.2f}%")
        
        # Графік останніх 20 epochs
        if len(df) >= 10:
            print(f"\n📉 Val Accuracy (останні 10-20 epochs):")
            start_idx = max(0, len(df) - 20)
            for idx in range(start_idx, len(df)):
                epoch_num = idx + 1
                acc = df.iloc[idx]['val_accuracy']
                bar_length = int(acc * 50)  # Scale to 50 chars
                bar = '█' * bar_length + '░' * (50 - bar_length)
                print(f"   E{epoch_num:3d}: {bar} {acc*100:.2f}%")
        
        # Час оновлення
        mod_time = datetime.fromtimestamp(os.path.getmtime(latest_csv))
        time_ago = datetime.now() - mod_time
        
        print(f"\n⏰ Останнє оновлення: {time_ago.seconds // 60} хвилин тому")
        
    except Exception as e:
        print(f"❌ Помилка: {e}")
    
    print("="*80)


if __name__ == "__main__":
    import sys
    import time
    
    # Якщо передано аргумент --watch, оновлювати кожні 60 секунд
    if len(sys.argv) > 1 and sys.argv[1] == '--watch':
        print("👁️ Режим моніторингу (оновлення кожні 60 сек). Ctrl+C для виходу.")
        try:
            while True:
                os.system('clear' if os.name == 'posix' else 'cls')
                monitor_training()
                print("\n⏳ Наступне оновлення через 60 секунд...")
                time.sleep(60)
        except KeyboardInterrupt:
            print("\n👋 Моніторинг зупинено")
    else:
        monitor_training()
        print("\n💡 Підказка: використайте --watch для автоматичного оновлення")
