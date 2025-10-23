#!/usr/bin/env python3
"""
🔄 ПРОДОВЖЕННЯ ТРЕНУВАННЯ - Resume training from checkpoint
"""
import os
import sys
import asyncio
import logging
import numpy as np
import tensorflow as tf
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_classification import ClassificationTrainer, CLASSIFICATION_CONFIG
from gpu_config import configure_gpu

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Вимикаємо XLA для уникнення конфліктів з динамічними batch_size
configure_gpu(use_xla=False)


async def resume_training():
    """Продовження тренування з checkpoint"""
    
    # Знаходимо останню модель
    model_dir = 'models/classification_BTC'
    
    # Знайти останній checkpoint
    checkpoints = [f for f in os.listdir(model_dir) if f.startswith('model_') and f.endswith('.keras')]
    if not checkpoints:
        logger.error("❌ Checkpoints не знайдено!")
        return None
    
    latest_checkpoint = max(checkpoints, key=lambda f: os.path.getctime(os.path.join(model_dir, f)))
    checkpoint_path = os.path.join(model_dir, latest_checkpoint)
    
    # Знайти відповідний CSV файл
    timestamp = latest_checkpoint.replace('model_', '').replace('.keras', '')
    csv_path = os.path.join(model_dir, f'training_{timestamp}.csv')
    
    logger.info("="*80)
    logger.info("🔄 ПРОДОВЖЕННЯ ТРЕНУВАННЯ")
    logger.info("="*80)
    logger.info(f"\n📁 Checkpoint: {checkpoint_path}")
    logger.info(f"📊 History: {csv_path}")
    
    # Прочитати кількість epochs з CSV
    import pandas as pd
    history_df = pd.read_csv(csv_path)
    completed_epochs = len(history_df)
    last_val_acc = history_df['val_accuracy'].iloc[-1]
    best_val_acc = history_df['val_accuracy'].max()
    best_epoch = history_df['val_accuracy'].idxmax() + 1
    
    logger.info(f"\n📈 Прогрес:")
    logger.info(f"   Завершено epochs: {completed_epochs}")
    logger.info(f"   Остання val_accuracy: {last_val_acc:.4f} ({last_val_acc*100:.2f}%)")
    logger.info(f"   Найкраща val_accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%) на epoch {best_epoch}")
    
    # Створюємо trainer
    trainer = ClassificationTrainer('BTCUSDT')
    
    # Завантаження даних
    logger.info(f"\n📥 Завантаження даних...")
    data = await trainer.load_data()
    if data is None:
        return None
    
    # Розрахунок фічей
    logger.info(f"🔧 Розрахунок фічей...")
    df = trainer.calculate_features(data)
    
    # Створення міток
    logger.info(f"🏷️ Створення міток...")
    df, labels = trainer.create_labels(df)
    
    # Створення послідовностей
    logger.info(f"🔄 Створення послідовностей...")
    X, y = trainer.create_sequences(df, labels)
    
    # Розділення даних (той самий спосіб)
    val_size = int(len(X) * trainer.config['validation_split'])
    test_size = int(len(X) * trainer.config['test_split'])
    train_size = len(X) - val_size - test_size
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    logger.info(f"\n📊 Розподіл даних:")
    logger.info(f"   Train: {len(X_train)}")
    logger.info(f"   Val:   {len(X_val)}")
    logger.info(f"   Test:  {len(X_test)}")
    
    # Class weights
    if trainer.config.get('use_class_weights', True):
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights = np.clip(class_weights, 0.5, 3.0)
        class_weight_dict = dict(enumerate(class_weights))
    else:
        class_weight_dict = None
    
    # Завантаження моделі
    logger.info(f"\n🔄 Завантаження моделі з checkpoint...")
    model = tf.keras.models.load_model(checkpoint_path)
    trainer.model = model
    
    logger.info(f"✅ Модель завантажено!")
    
    # Створюємо нові callbacks
    new_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    new_checkpoint_path = f'{model_dir}/model_resumed_{new_timestamp}.keras'
    new_csv_path = f'{model_dir}/training_resumed_{new_timestamp}.csv'
    
    callbacks_list = [
        tf.keras.callbacks.ModelCheckpoint(
            new_checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=trainer.config['early_stopping_patience'],
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=trainer.config['reduce_lr_patience'],
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(new_csv_path),
        tf.keras.callbacks.TensorBoard(
            log_dir=f'logs/tensorboard_resumed_{new_timestamp}',
            histogram_freq=0
        ),
    ]
    
    # Розрахунок залишку epochs
    remaining_epochs = trainer.config['epochs'] - completed_epochs
    
    logger.info(f"\n🏋️ Продовження тренування...")
    logger.info(f"   Залишилось epochs: {remaining_epochs}")
    logger.info(f"   Початковий epoch: {completed_epochs + 1}")
    logger.info(f"   Кінцевий epoch: {trainer.config['epochs']}")
    logger.info(f"")
    
    # Тренування
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        initial_epoch=completed_epochs,  # ⭐ КЛЮЧОВИЙ ПАРАМЕТР
        epochs=trainer.config['epochs'],
        batch_size=trainer.config['batch_size'],
        class_weight=class_weight_dict,
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Оцінка на тестовому наборі
    logger.info(f"\n📈 Оцінка на тестовому наборі...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Детальна оцінка
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_test, y_pred_classes)
    
    logger.info(f"\n📊 Confusion Matrix:")
    logger.info(f"           Predicted")
    logger.info(f"           DOWN  NEUTRAL  UP")
    logger.info(f"Actual DOWN    {cm[0][0]:4d}    {cm[0][1]:4d}  {cm[0][2]:4d}")
    logger.info(f"    NEUTRAL    {cm[1][0]:4d}    {cm[1][1]:4d}  {cm[1][2]:4d}")
    logger.info(f"         UP    {cm[2][0]:4d}    {cm[2][1]:4d}  {cm[2][2]:4d}")
    
    logger.info(f"\n📋 Classification Report:")
    logger.info(f"\n{classification_report(y_test, y_pred_classes, target_names=['DOWN', 'NEUTRAL', 'UP'])}")
    
    # Результати
    best_val_acc = max(history.history['val_accuracy'])
    
    logger.info(f"\n{'='*80}")
    logger.info(f"✅ ТРЕНУВАННЯ ЗАВЕРШЕНО")
    logger.info(f"{'='*80}")
    logger.info(f"\n📊 ФІНАЛЬНІ РЕЗУЛЬТАТИ:")
    logger.info(f"   Best val_accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    logger.info(f"   Test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    logger.info(f"   Test loss: {test_loss:.6f}")
    logger.info(f"\n💾 ЗБЕРЕЖЕНО:")
    logger.info(f"   Model: {new_checkpoint_path}")
    logger.info(f"   History: {new_csv_path}")
    
    return {
        'symbol': 'BTCUSDT',
        'val_accuracy': best_val_acc,
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'confusion_matrix': cm.tolist(),
        'model_path': new_checkpoint_path,
        'resumed_from': checkpoint_path,
        'completed_epochs': completed_epochs,
    }


if __name__ == "__main__":
    logger.info("\n" + "="*80)
    logger.info("🔄 RESUME TRAINING - Продовження тренування з checkpoint")
    logger.info("="*80 + "\n")
    
    result = asyncio.run(resume_training())
    
    if result:
        logger.info("\n🎉 УСПІХ!")
        sys.exit(0)
    else:
        logger.error("\n❌ НЕВДАЧА")
        sys.exit(1)
