#!/usr/bin/env python3
"""
Швидка статистика проекту після очистки
"""

import os
from pathlib import Path

print("\n" + "="*60)
print("📊 PROJECT STRUCTURE AFTER CLEANUP")
print("="*60 + "\n")

# Основні директорії
dirs = {
    'training': 'Основні ML файли',
    'models': 'Збережені моделі',
    'fast_indicators': 'Rust індикатори',
    'archive': 'Старі файли (GRU, TensorFlow)',
    'intelligent_sys': 'Data fetching',
    'database': 'DB schemas',
}

for dir_name, description in dirs.items():
    if Path(dir_name).exists():
        py_files = list(Path(dir_name).rglob('*.py'))
        size_mb = sum(f.stat().st_size for f in Path(dir_name).rglob('*') if f.is_file()) / 1024 / 1024
        print(f"📁 {dir_name:20s} - {len(py_files):3d} .py files  ({size_mb:.1f}MB) - {description}")

print("\n" + "="*60)
print("🎯 ACTIVE FILES")  
print("="*60 + "\n")

active_files = [
    'training/simple_trend_classifier.py',
    'training/batch_train_rf.py', 
    'training/rust_features.py',
    'unified_binance_loader.py',
    'check_gpu.py',
]

for f in active_files:
    if Path(f).exists():
        lines = len(Path(f).read_text().splitlines())
        size_kb = Path(f).stat().st_size / 1024
        print(f"  ✅ {f:40s} {lines:4d} lines  ({size_kb:.1f}KB)")

print("\n" + "="*60)
print("🗂️  ARCHIVED")
print("="*60 + "\n")

archive_dirs = list(Path('archive').glob('old_*'))
for d in sorted(archive_dirs):
    files = len(list(d.rglob('*.py')))
    print(f"  📦 {d.name:40s} {files:3d} files")

print("\n" + "="*60)
print("💾 TRAINED MODELS")
print("="*60 + "\n")

if Path('models').exists():
    model_dirs = [d for d in Path('models').iterdir() if d.is_dir() and d.name.startswith('simple_trend_')]
    for m in sorted(model_dirs):
        files = list(m.glob('*.pkl'))
        print(f"  🤖 {m.name}")
        for f in files:
            size_kb = f.stat().st_size / 1024
            print(f"     - {f.name:30s} ({size_kb:.0f}KB)")

print("\n" + "="*60 + "\n")
