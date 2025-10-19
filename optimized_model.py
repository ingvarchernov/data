import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from gpu_config import configure_gpu

@tf.keras.utils.register_keras_serializable()
def mape(y_true, y_pred):
    """MAPE метрика з кращою стабільністю"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.reduce_mean(tf.abs((y_true - y_pred) / (tf.abs(y_true) + 1e-6))) * 100

@tf.keras.utils.register_keras_serializable()
def directional_accuracy(y_true, y_pred):
    """Точність напрямку руху ціни (зростання/падіння)"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Рахуємо різницю (дельту) між послідовними значеннями
    # Якщо батч має форму (batch_size, 1), береме знак самих значень
    # Це працює, якщо y_true вже є зміною ціни (price_diff)
    true_direction = tf.sign(y_true + 1e-8)  # додаємо epsilon щоб уникнути 0
    pred_direction = tf.sign(y_pred + 1e-8)
    
    # Порівнюємо напрямки
    correct = tf.cast(tf.equal(true_direction, pred_direction), tf.float32)
    return tf.reduce_mean(correct)
from keras import layers, callbacks, optimizers, mixed_precision
from typing import Tuple, Dict, List, Optional, Union
import matplotlib.pyplot as plt
from datetime import datetime
import gc


logger = logging.getLogger(__name__)

# Налаштування mixed precision
mixed_precision.set_global_policy('mixed_float16')


class DatabaseHistoryCallback(callbacks.Callback):
    """Callback для збереження історії тренування в базу даних"""
    
    def __init__(self, db_engine, symbol_id: int, interval_id: int, fold: int = 1):
        super().__init__()
        self.db_engine = db_engine
        self.symbol_id = symbol_id
        self.interval_id = interval_id
        self.fold = fold
    
    def on_epoch_end(self, epoch, logs=None):
        """Збереження метрик епохи в БД"""
        if logs is None:
            return
        
        try:
            from sqlalchemy import text
            with self.db_engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO training_history 
                    (symbol_id, interval_id, fold, epoch, loss, mae, val_loss, val_mae, mape, val_mape, directional_accuracy, real_mae, real_mape)
                    VALUES (:symbol_id, :interval_id, :fold, :epoch, :loss, :mae, :val_loss, :val_mae, :mape, :val_mape, :directional_accuracy, :real_mae, :real_mape)
                    ON CONFLICT (symbol_id, interval_id, fold, epoch)
                    DO UPDATE SET
                        loss = EXCLUDED.loss,
                        mae = EXCLUDED.mae,
                        val_loss = EXCLUDED.val_loss,
                        val_mae = EXCLUDED.val_mae,
                        mape = EXCLUDED.mape,
                        val_mape = EXCLUDED.val_mape,
                        directional_accuracy = EXCLUDED.directional_accuracy,
                        real_mae = EXCLUDED.real_mae,
                        real_mape = EXCLUDED.real_mape
                """), {
                    'symbol_id': self.symbol_id,
                    'interval_id': self.interval_id,
                    'fold': self.fold,
                    'epoch': epoch + 1,  # epoch починається з 0, але зберігаємо з 1
                    'loss': float(logs.get('loss', 0)),
                    'mae': float(logs.get('mae', 0)),
                    'val_loss': float(logs.get('val_loss', 0)),
                    'val_mae': float(logs.get('val_mae', 0)),
                    'mape': float(logs.get('mape', 0)),
                    'val_mape': float(logs.get('val_mape', 0)),
                    'directional_accuracy': float(logs.get('directional_accuracy', 0)),
                    'real_mae': float(logs.get('real_mae', logs.get('mae', 0))),
                    'real_mape': float(logs.get('real_mape', logs.get('mape', 0)))
                })
                conn.commit()
        except Exception as e:
            logger.error(f"❌ Помилка збереження в БД: {e}")


class DenormalizedMetricsCallback(callbacks.Callback):
    """Callback для виводу спрощених метрик"""
    
    def __init__(self, scaler=None, feature_index: int = None, X_val: np.ndarray = None, y_val: np.ndarray = None):
        super().__init__()
        self.scaler = scaler
        self.feature_index = feature_index
        self.X_val = X_val
        self.y_val = y_val
    
    def on_epoch_end(self, epoch, logs=None):
        """Вивід метрик після епохи"""
        if logs is None:
            return
        
        try:
            # Отримуємо передбачення на валідаційних даних
            if self.X_val is not None and self.y_val is not None and self.scaler is not None:
                y_pred = self.model.predict(self.X_val, verbose=0).flatten()
                y_true = self.y_val  # відсоткові зміни
                
                # Розраховуємо метрики на рівні відсоткових змін
                mae_norm = np.mean(np.abs(y_true - y_pred))
                mape_norm = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-6))) * 100
                
                # Directional accuracy: знак відсоткових змін
                dir_acc = np.mean(np.sign(y_true) == np.sign(y_pred)) * 100
                
                logger.info(f"💰 Метрики відсоткових змін (Epoch {epoch + 1}): "
                           f"MAE={mae_norm:.4f}, MAPE={mape_norm:.2f}%, Напрямок={dir_acc:.1f}%")
                
                # Реальні метрики: конвертуємо назад до абсолютних цін для інтерпретації
                # Отримуємо поточні ціни з валідаційних даних (останній timestep, денормалізовані)
                # Оскільки дані нормалізовані, беремо денормалізовані close з X_val
                current_prices = self.X_val[:, -1, self.feature_index].flatten()
                
                # Реальні наступні ціни = поточні + (передбачені відсоткові зміни * поточні / 100)
                y_true_pct = y_true  # вже відсоткові зміни
                y_pred_pct = y_pred  # вже відсоткові зміни
                
                true_next_prices = current_prices * (1 + y_true_pct / 100)
                pred_next_prices = current_prices * (1 + y_pred_pct / 100)
                
                mae_real = np.mean(np.abs(true_next_prices - pred_next_prices))
                mape_real = np.mean(np.abs((true_next_prices - pred_next_prices) / (np.abs(true_next_prices) + 1e-6))) * 100
                
                logger.info(f"💵 Реальні метрики цін (Epoch {epoch + 1}): "
                           f"MAE={mae_real:.2f}, MAPE={mape_real:.2f}%, Напрямок={dir_acc:.1f}%")
                
                # Додаємо метрики до logs для збереження в БД
                logs['mape'] = mape_real
                logs['val_mape'] = mape_real  # для validation використовуємо ті ж метрики
                logs['directional_accuracy'] = dir_acc
                logs['real_mae'] = mae_real
                logs['real_mape'] = mape_real
            
        except Exception as e:
            logger.error(f"❌ Помилка розрахунку метрик: {e}")


class TransformerBlock(layers.Layer):
    """Transformer блок з Multi-Head Attention"""
    
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config

class PositionalEncoding(layers.Layer):
    """Позиційне кодування для Transformer"""
    
    def __init__(self, maxlen, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        
    def build(self, input_shape):
        self.pos_encoding = self.add_weight(
            name="pos_encoding",
            shape=(self.maxlen, self.embed_dim),
            initializer="uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        return inputs + tf.gather(self.pos_encoding, positions)

    def get_config(self):
        config = super().get_config()
        config.update({
            "maxlen": self.maxlen,
            "embed_dim": self.embed_dim,
        })
        return config

class OptimizedPricePredictionModel:
    """Оптимізована модель для прогнозування цін"""
    
    def __init__(self, 
                 input_shape: Tuple[int, int],
                 model_type: str = "transformer_lstm",
                 use_mixed_precision: bool = True,
                 use_xla: bool = True,
                 scaler=None,
                 feature_index: int = None,
                 model_config: Dict = None):
        
        self.input_shape = input_shape
        # Мапінг типів моделей для зручності користувача
        model_type_mapping = {
            "transformer": "transformer_lstm",
            "advanced_lstm": "advanced_lstm", 
            "cnn_lstm": "cnn_lstm"
        }
        self.model_type = model_type_mapping.get(model_type, model_type)
        self.use_mixed_precision = use_mixed_precision
        self.use_xla = use_xla
        self.scaler = scaler
        self.feature_index = feature_index
        
        # Конфігурація моделі (якщо не передана - використовуємо дефолтну)
        if model_config is None:
            from optimized_config import MODEL_CONFIG
            self.model_config = MODEL_CONFIG
        else:
            self.model_config = model_config
        
        # Налаштування GPU
        self._configure_gpu()
        
        # Кастомні об'єкти для серіалізації
        self.custom_objects = {
            'TransformerBlock': TransformerBlock,
            'PositionalEncoding': PositionalEncoding,
            'mape': mape,
            'directional_accuracy': directional_accuracy,
        }
        
        # Ініціалізація моделі
        self.model = None
        
        # Створюємо модель
        self.create_model()
    
    def _configure_gpu(self):
        """Оптимальне використання GPU без надмірного логування"""
        configure_gpu(
            use_mixed_precision=self.use_mixed_precision,
            use_xla=self.use_xla,
            memory_growth=True,
        )
    
    def build_transformer_lstm_model(self) -> keras.Model:
        """Hybrid Transformer-LSTM модель"""
        inputs = layers.Input(shape=self.input_shape)

        # Позиційне кодування
        x = PositionalEncoding(maxlen=self.input_shape[0], embed_dim=self.input_shape[1])(inputs)

        # Transformer блоки
        x = TransformerBlock(embed_dim=self.input_shape[1], num_heads=8, ff_dim=256, rate=0.1)(x)
        x = TransformerBlock(embed_dim=self.input_shape[1], num_heads=8, ff_dim=256, rate=0.1)(x)

        # LSTM блоки
        x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.2, kernel_regularizer=tf.keras.regularizers.l2(0.001)))(x)
        x = layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)(x)

        # Dense блоки з residual connections
        dense_input = x
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)

        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)

        # Residual connection
        if dense_input.shape[-1] == 64:
            x = layers.Add()([x, dense_input])

        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.1)(x)

        # Вихідний шар з float32 для стабільності
        outputs = layers.Dense(1, activation='linear', dtype='float32', name='output')(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name='transformer_lstm_model')

        return model
    
    def build_advanced_lstm_model(self) -> keras.Model:
        """Покращена LSTM модель з attention - використовує параметри з MODEL_CONFIG"""
        inputs = layers.Input(shape=self.input_shape)
        
        # Отримуємо параметри з конфігу
        lstm_units_1 = self.model_config.get('lstm_units_1', 320)
        lstm_units_2 = self.model_config.get('lstm_units_2', 160)
        lstm_units_3 = self.model_config.get('lstm_units_3', 80)
        attention_heads = self.model_config.get('attention_heads', 10)
        attention_key_dim = self.model_config.get('attention_key_dim', 80)
        dense_units = self.model_config.get('dense_units', [640, 320, 160, 80])
        
        # Нормалізація входу
        x = layers.LayerNormalization()(inputs)
        
        # Багатошарові LSTM з residual connections та L2 регуляризацією
        lstm1 = layers.LSTM(lstm_units_1, return_sequences=True, dropout=0.4, recurrent_dropout=0.3,
                           kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        lstm1_norm = layers.LayerNormalization()(lstm1)
        
        lstm2 = layers.LSTM(lstm_units_2, return_sequences=True, dropout=0.4, recurrent_dropout=0.3,
                           kernel_regularizer=tf.keras.regularizers.l2(0.01))(lstm1_norm)
        lstm2_norm = layers.LayerNormalization()(lstm2)
        
        lstm3 = layers.LSTM(lstm_units_3, return_sequences=True, dropout=0.4, recurrent_dropout=0.3,
                           kernel_regularizer=tf.keras.regularizers.l2(0.01))(lstm2_norm)
        lstm3_norm = layers.LayerNormalization()(lstm3)
        
        # Attention механізм з параметрами з конфігу
        attention = layers.MultiHeadAttention(
            num_heads=attention_heads, 
            key_dim=attention_key_dim
        )(lstm3_norm, lstm3_norm)
        attention = layers.Dropout(0.1)(attention)
        
        # Global pooling
        pooled = layers.GlobalAveragePooling1D()(attention)
        
        # Dense блоки з параметрами з конфігу та L2 регуляризацією
        x = pooled
        for i, units in enumerate(dense_units):
            x = layers.Dense(units, activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
            x = layers.BatchNormalization()(x)
            dropout_rate = 0.5 if i == 0 else 0.4 if i == 1 else 0.3
            x = layers.Dropout(dropout_rate)(x)
        
        outputs = layers.Dense(1, activation='linear', dtype='float32', name='output')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='advanced_lstm_model')
        
        return model
    
    def build_cnn_lstm_model(self) -> keras.Model:
        """CNN-LSTM модель для часових рядів"""
        inputs = layers.Input(shape=self.input_shape)
        
        # 1D CNN блоки
        x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)
        
        x = layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)
        
        x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # LSTM блоки
        x = layers.LSTM(32, return_sequences=True, dropout=0.2)(x)
        x = layers.LSTM(16, return_sequences=False, dropout=0.2)(x)
        
        # Dense блоки
        x = layers.Dense(50, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        outputs = layers.Dense(1, activation='linear', dtype='float32', name='output')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='cnn_lstm_model')
        
        return model
    
    def build_dense_model(self) -> keras.Model:
        """Покращена Dense модель для прогнозування з індикаторами"""
        inputs = layers.Input(shape=self.input_shape, name='input')

        # Отримуємо dense_units з конфігу або використовуємо defaults
        dense_units = self.model_config.get('dense_units', [256, 128, 64])

        # Нормалізація входу
        x = layers.LayerNormalization()(inputs)

        # Декілька Dense шарів з dropout та batch normalization
        for i, units in enumerate(dense_units):
            x = layers.Dense(units, activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)

        # Вихідний шар
        outputs = layers.Dense(1, activation='sigmoid', dtype='float32', name='output')(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name='enhanced_dense_model')

        return model
    
    def create_model(self) -> keras.Model:
        """Створення моделі згідно з типом"""
        if self.model_type == "transformer_lstm":
            model = self.build_transformer_lstm_model()
        elif self.model_type == "advanced_lstm":
            model = self.build_advanced_lstm_model()
        elif self.model_type == "cnn_lstm":
            model = self.build_cnn_lstm_model()
        elif self.model_type == "dense":
            model = self.build_dense_model()
        else:
            raise ValueError(f"Невідомий тип моделі: {self.model_type}")
        
        logger.info(f"✅ Створена модель типу: {self.model_type}")
        logger.info(f"📊 Параметрів в моделі: {model.count_params():,}")
        
        # Зберігаємо модель
        self.model = model
        return model
    
    def compile_model(self, model: keras.Model, learning_rate: float = 0.001) -> keras.Model:
        """Компіляція моделі з оптимізованими параметрами"""
        
        # Оптимізатор з gradient clipping
        optimizer = optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            clipnorm=1.0
        )
        
        # Для mixed precision
        if self.use_mixed_precision:
            optimizer = mixed_precision.LossScaleOptimizer(optimizer)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',  # Binary classification
            metrics=[
                'accuracy',  # accuracy for binary classification
                directional_accuracy  # точність напрямку руху
            ],
            jit_compile=self.use_xla
        )
        
        # Зберігаємо додаткові метрики для використання у custom callbacks
        self.additional_metrics = {
            'mape': mape,
            'directional_accuracy': directional_accuracy
        }
        
        return model
    
    def _create_lr_schedule(self, epochs: int = 200, warmup_epochs: int = 10, base_lr: float = 0.001):
        """Створює розклад навчання з warmup та cosine annealing"""
        def lr_schedule(epoch):
            if epoch < warmup_epochs:
                # Linear warmup
                return base_lr * (epoch + 1) / warmup_epochs
            else:
                # Cosine annealing з restarts
                progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
                cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
                
                # Додаємо periodic restarts кожні 50 епох після warmup
                if epoch > warmup_epochs + 50:
                    cycle = (epoch - warmup_epochs) // 50
                    cycle_progress = ((epoch - warmup_epochs) % 50) / 50
                    cosine_decay = cosine_decay * (0.5 + 0.5 * np.cos(2 * np.pi * cycle_progress))
                
                return base_lr * cosine_decay
        
        return lr_schedule
    
    def get_callbacks(self, 
                     model_save_path: str,
                     patience: int = 50,
                     reduce_lr_patience: int = 20) -> List[callbacks.Callback]:
        """Створення оптимізованих callback'ів"""
        
        callback_list = [
            # Покращений Early stopping з моніторингом основних метрик
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1,
                mode='min'
            ),
            
            # ReduceLROnPlateau
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=reduce_lr_patience,
                min_lr=1e-7,
                verbose=1,
                mode='min'
            ),
            
            # Model checkpoint
            callbacks.ModelCheckpoint(
                filepath=model_save_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
                mode='min'
            ),
            
            # TensorBoard
            callbacks.TensorBoard(
                log_dir=f'./logs/{datetime.now().strftime("%Y%m%d-%H%M%S")}',
                histogram_freq=10,
                write_graph=True,
                update_freq='epoch'
            ),
            
            # Покращений Learning rate scheduler з warmup та cosine annealing
            callbacks.LearningRateScheduler(
                self._create_lr_schedule(epochs=200, warmup_epochs=10),
                verbose=0
            ),
            
            # Memory cleanup
            callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: gc.collect()
            )
        ]
        
        return callback_list
    
    def create_data_generators(self, 
                             X_train: np.ndarray, 
                             y_train: np.ndarray,
                             X_val: np.ndarray, 
                             y_val: np.ndarray,
                             batch_size: int = 64) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Створення оптимізованих data generators"""
        
        # Тренувальний датасет
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=1000)
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        
        # Валідаційний датасет
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_dataset = val_dataset.batch(batch_size)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
        
        return train_dataset, val_dataset
    
    def train_model(self,
                   X_train: np.ndarray,
                   y_train: np.ndarray,
                   X_val: np.ndarray,
                   y_val: np.ndarray,
                   model_save_path: str,
                   epochs: int = 200,
                   batch_size: int = 64,
                   learning_rate: float = 0.001,
                   db_callback: callbacks.Callback = None,
                   additional_callbacks: List[callbacks.Callback] = None) -> Tuple[keras.Model, keras.callbacks.History]:
        """Тренування моделі"""
        
        # Створення та компіляція моделі
        model = self.create_model()
        model = self.compile_model(model, learning_rate)
        
        # Створення data generators
        train_dataset, val_dataset = self.create_data_generators(
            X_train, y_train, X_val, y_val, batch_size
        )
        
        # Callback'и
        callback_list = self.get_callbacks(model_save_path)
        
        # Додаємо додаткові callback'и ПЕРЕД DB callback (щоб метрики були розраховані)
        if additional_callbacks is not None:
            callback_list.extend(additional_callbacks)
        
        # Додаємо DB callback якщо переданий (після additional, щоб метрики були в logs)
        if db_callback is not None:
            callback_list.append(db_callback)
        
        # Тренування
        logger.info(f"🚀 Початок тренування моделі {self.model_type}")
        
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callback_list,
            verbose=2  # Одна лінія на епоху
        )
        
        logger.info("✅ Тренування завершено")
        
        return model, history
    
    def save_model_with_metadata(self, 
                                model: keras.Model, 
                                save_path: str, 
                                metadata: Dict = None):
        """Збереження моделі з метаданими"""
        try:
            # Зберігаємо модель у HDF5 форматі для сумісності
            model.save(save_path, save_format='h5')
            
            # Зберігаємо метадані
            if metadata:
                metadata_path = save_path.replace('.h5', '_metadata.json')
                import json
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            logger.info(f"✅ Модель збережена: {save_path}")
            
        except Exception as e:
            logger.error(f"❌ Помилка збереження моделі: {e}")
            raise
    
    def load_model(self, model_path: str):
        """Завантаження моделі з файлу"""
        try:
            # Завантажуємо модель
            self.model = keras.models.load_model(
                model_path, 
                custom_objects=self.custom_objects
            )
            
            logger.info(f"✅ Модель завантажена: {model_path}")
            
        except Exception as e:
            logger.error(f"❌ Помилка завантаження моделі: {e}")
            raise
    
    def plot_training_history(self, history: keras.callbacks.History, save_path: str = None):
        """Візуалізація процесу тренування"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(history.history['loss'], label='Train Loss')
        axes[0, 0].plot(history.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # MAE
        axes[0, 1].plot(history.history['mae'], label='Train MAE')
        axes[0, 1].plot(history.history['val_mae'], label='Val MAE')
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # MAPE
        if 'mape' in history.history:
            axes[1, 0].plot(history.history['mape'], label='Train MAPE')
            axes[1, 0].plot(history.history['val_mape'], label='Val MAPE')
            axes[1, 0].set_title('Mean Absolute Percentage Error')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('MAPE (%)')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Learning Rate
        if 'lr' in history.history:
            axes[1, 1].plot(history.history['lr'])
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✅ Графік збережено: {save_path}")
        
        plt.close()
    
    def predict(self, X: np.ndarray, return_confidence: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Прогнозування з використанням навченої моделі
        
        Args:
            X: Вхідні дані для прогнозування
            return_confidence: Чи повертати також впевненість прогнозу
            
        Returns:
            Прогноз або (прогноз, впевненість) якщо return_confidence=True
        """
        if self.model is None:
            raise ValueError("Модель не ініціалізована. Спочатку завантажте або навчіть модель.")
        
        try:
            # Переконуємося, що дані мають правильну форму
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
            
            # Нормалізуємо дані, якщо є scaler
            if self.scaler is not None:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X
            
            # Робимо прогноз
            predictions = self.model.predict(X_scaled, verbose=0)
            
            # Якщо потрібно повернути впевненість
            if return_confidence:
                # Проста оцінка впевненості на основі відстані від 0
                confidence = 1.0 / (1.0 + np.abs(predictions.flatten()))
                confidence = np.clip(confidence, 0.1, 0.9)  # Обмежуємо діапазон
                return predictions.flatten(), confidence
            
            return predictions.flatten()
            
        except Exception as e:
            logger.error(f"❌ Помилка прогнозування: {e}")
            raise

# Фабричні функції
def create_transformer_lstm_model(input_shape: Tuple[int, ...]) -> OptimizedPricePredictionModel:
    """Створення Transformer-LSTM моделі"""
    return OptimizedPricePredictionModel(input_shape, "transformer_lstm")

def create_advanced_lstm_model(input_shape: Tuple[int, ...]) -> OptimizedPricePredictionModel:
    """Створення покращеної LSTM моделі"""
    return OptimizedPricePredictionModel(input_shape, "advanced_lstm")

def create_cnn_lstm_model(input_shape: Tuple[int, ...]) -> OptimizedPricePredictionModel:
    """Створення CNN-LSTM моделі"""
    return OptimizedPricePredictionModel(input_shape, "cnn_lstm")

def create_dense_model(input_shape: Tuple[int, ...]) -> OptimizedPricePredictionModel:
    """Створення Dense моделі"""
    return OptimizedPricePredictionModel(input_shape, "dense")