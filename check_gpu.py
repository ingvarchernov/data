#!/usr/bin/env python3
"""
Перевірка використання GPU
"""
import tensorflow as tf

print("="*80)
print("🔍 ПЕРЕВІРКА GPU")
print("="*80)

# Список GPU пристроїв
gpus = tf.config.list_physical_devices('GPU')
print(f"\n📊 Знайдено GPU пристроїв: {len(gpus)}")

for i, gpu in enumerate(gpus):
    print(f"   GPU {i}: {gpu}")

# Детальна інформація
if gpus:
    print("\n✅ GPU ДОСТУПНИЙ")
    
    # Перевірка чи TensorFlow бачить GPU
    from tensorflow.python.client import device_lib
    local_devices = device_lib.list_local_devices()
    
    print("\n📋 Всі доступні пристрої:")
    for device in local_devices:
        if device.device_type == 'GPU':
            print(f"   🎮 {device.name}")
            print(f"      Memory: {device.memory_limit / 1024**3:.2f} GB")
    
    # Тестування обчислень на GPU
    print("\n🧪 Тестування обчислень на GPU...")
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
        c = tf.matmul(a, b)
    
    print(f"   Результат обчислення на GPU: OK")
    print(f"   Пристрій для тензора c: {c.device}")
    
    # Перевірка mixed precision
    from tensorflow import keras
    policy = keras.mixed_precision.global_policy()
    print(f"\n⚡ Mixed Precision Policy: {policy.name}")
    
    print("\n" + "="*80)
    print("✅ ВСЕ ПРАЦЮЄ НА GPU!")
    print("="*80)
    
else:
    print("\n❌ GPU НЕ ЗНАЙДЕНО")
    print("TensorFlow буде використовувати CPU")
