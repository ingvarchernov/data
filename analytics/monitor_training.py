#!/usr/bin/env python3
"""
Моніторинг процесу тренування в реальному часі
"""
import time
import re
import os
from datetime import datetime

def monitor_training(log_file='train_output.log', interval=10):
    """Моніторинг логу тренування"""
    
    print("="*80)
    print("📊 МОНІТОРИНГ ТРЕНУВАННЯ КЛАСИФІКАЦІЙНОЇ МОДЕЛІ")
    print("="*80)
    print(f"📄 Лог файл: {log_file}")
    print(f"⏱️  Інтервал оновлення: {interval} сек")
    print("="*80)
    
    last_size = 0
    epoch_data = {}
    
    try:
        while True:
            if not os.path.exists(log_file):
                print("⏳ Очікування створення log файлу...")
                time.sleep(interval)
                continue
            
            # Читання нових рядків
            with open(log_file, 'r') as f:
                f.seek(last_size)
                new_lines = f.readlines()
                last_size = f.tell()
            
            # Парсинг нових рядків
            for line in new_lines:
                # Epoch завершення
                match = re.search(r'Epoch (\d+)/(\d+).*━+\s+\d+s.*val_accuracy:\s+([\d.]+).*val_loss:\s+([\d.]+)', line)
                if match:
                    epoch = int(match.group(1))
                    total_epochs = int(match.group(2))
                    val_acc = float(match.group(3))
                    val_loss = float(match.group(4))
                    
                    epoch_data[epoch] = {
                        'val_acc': val_acc,
                        'val_loss': val_loss,
                        'time': datetime.now()
                    }
                    
                    # Виведення прогресу
                    progress = epoch / total_epochs * 100
                    bar_length = 40
                    filled = int(bar_length * epoch / total_epochs)
                    bar = '█' * filled + '░' * (bar_length - filled)
                    
                    print(f"\n{'='*80}")
                    print(f"📈 Epoch {epoch}/{total_epochs} ({progress:.1f}%)")
                    print(f"[{bar}]")
                    print(f"   Val Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
                    print(f"   Val Loss: {val_loss:.4f}")
                    
                    # Порівняння з попереднім
                    if epoch > 1 and (epoch-1) in epoch_data:
                        prev = epoch_data[epoch-1]
                        acc_delta = val_acc - prev['val_acc']
                        loss_delta = val_loss - prev['val_loss']
                        
                        acc_arrow = "📈" if acc_delta > 0 else "📉" if acc_delta < 0 else "➡️"
                        loss_arrow = "📉" if loss_delta < 0 else "📈" if loss_delta > 0 else "➡️"
                        
                        print(f"   Δ Accuracy: {acc_arrow} {acc_delta:+.4f}")
                        print(f"   Δ Loss: {loss_arrow} {loss_delta:+.4f}")
                    
                    # Найкраща епоха
                    best_epoch = max(epoch_data.items(), key=lambda x: x[1]['val_acc'])
                    if best_epoch[0] == epoch:
                        print(f"   🏆 NEW BEST!")
                    else:
                        print(f"   🥇 Best: Epoch {best_epoch[0]} (acc={best_epoch[1]['val_acc']:.4f})")
                
                # Checkpoint збережено
                if "val_accuracy improved" in line:
                    print(f"   💾 Checkpoint збережено")
                elif "val_accuracy did not improve" in line:
                    print(f"   ⏭️  No improvement")
                
                # Early stopping
                if "Restoring model weights from the end of the best epoch" in line:
                    print(f"\n{'='*80}")
                    print("⏹️  EARLY STOPPING - Тренування зупинено")
                    print(f"{'='*80}")
                
                # Помилки
                if "ERROR" in line or "Exception" in line:
                    print(f"\n❌ ПОМИЛКА: {line.strip()}")
            
            # Перевірка чи процес ще працює
            try:
                with open(log_file, 'r') as f:
                    content = f.read()
                    if "✅ Тренування завершено" in content or "❌ Помилка" in content:
                        print(f"\n{'='*80}")
                        print("🏁 ТРЕНУВАННЯ ЗАВЕРШЕНО")
                        print(f"{'='*80}")
                        
                        # Підсумок
                        if epoch_data:
                            best = max(epoch_data.items(), key=lambda x: x[1]['val_acc'])
                            print(f"\n📊 ПІДСУМОК:")
                            print(f"   Всього epochs: {len(epoch_data)}")
                            print(f"   Найкраща: Epoch {best[0]}")
                            print(f"   Accuracy: {best[1]['val_acc']:.4f} ({best[1]['val_acc']*100:.2f}%)")
                            print(f"   Loss: {best[1]['val_loss']:.4f}")
                        
                        break
            except:
                pass
            
            time.sleep(interval)
    
    except KeyboardInterrupt:
        print(f"\n\n{'='*80}")
        print("⏹️  Моніторинг зупинено користувачем")
        print(f"{'='*80}")
        
        if epoch_data:
            best = max(epoch_data.items(), key=lambda x: x[1]['val_acc'])
            print(f"\n📊 ПОТОЧНИЙ СТАН:")
            print(f"   Пройдено epochs: {len(epoch_data)}")
            print(f"   Найкраща: Epoch {best[0]}")
            print(f"   Accuracy: {best[1]['val_acc']:.4f} ({best[1]['val_acc']*100:.2f}%)")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', default='train_output.log', help='Лог файл')
    parser.add_argument('--interval', type=int, default=10, help='Інтервал оновлення (сек)')
    args = parser.parse_args()
    
    monitor_training(args.log, args.interval)
