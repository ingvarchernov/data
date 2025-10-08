import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras

# --- Кастомні метрики ---

@tf.keras.utils.register_keras_serializable()
def mape(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.reduce_mean(tf.abs((y_true - y_pred) / (tf.abs(y_true) + 1e-6))) * 100

@tf.keras.utils.register_keras_serializable()
def directional_accuracy(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    true_direction = tf.sign(y_true)
    pred_direction = tf.sign(y_pred)
    return tf.reduce_mean(tf.cast(tf.equal(true_direction, pred_direction), tf.float32))
from keras import layers, callbacks, optimizers, mixed_precision
from typing import Tuple, Dict, List, Optional
import matplotlib.pyplot as plt
from datetime import datetime
import gc


logger = logging.getLogger(__name__)

# Налаштування mixed precision
mixed_precision.set_global_policy('mixed_float16')

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
                 use_xla: bool = True):
        
        self.input_shape = input_shape
        self.model_type = model_type
        self.use_mixed_precision = use_mixed_precision
        self.use_xla = use_xla
        
        # Налаштування GPU
        self._configure_gpu()
        
        # Кастомні об'єкти для серіалізації
        self.custom_objects = {
            'TransformerBlock': TransformerBlock,
            'PositionalEncoding': PositionalEncoding,
            'mape': mape,
            'directional_accuracy': directional_accuracy,
        }
    
    def _configure_gpu(self):
        """Оптимальне використання GPU без надмірного логування"""
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                tf.config.set_visible_devices(gpus, 'GPU')
                tf.config.experimental.set_memory_growth(gpus[0], True)
                if self.use_xla:
                    tf.config.optimizer.set_jit(True)
                    logger.info("✅ XLA JIT увімкнено")
                logger.info(f"✅ GPU доступний: {len(gpus)} пристроїв")
            else:
                logger.warning("⚠️ GPU не знайдено, використовується CPU")
        except Exception as e:
            logger.error(f"❌ Помилка налаштування GPU: {e}")
    
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
        """Покращена LSTM модель з attention"""
        inputs = layers.Input(shape=self.input_shape)
        
        # Нормалізація входу
        x = layers.LayerNormalization()(inputs)
        
        # Багатошарові LSTM з residual connections
        lstm1 = layers.LSTM(32, return_sequences=True, dropout=0.2, recurrent_dropout=0.1)(x)
        lstm1_norm = layers.LayerNormalization()(lstm1)
        
        lstm2 = layers.LSTM(16, return_sequences=True, dropout=0.2, recurrent_dropout=0.1)(lstm1_norm)
        lstm2_norm = layers.LayerNormalization()(lstm2)
        
        # Attention механізм
        attention = layers.MultiHeadAttention(num_heads=4, key_dim=64)(lstm2_norm, lstm2_norm)
        attention = layers.Dropout(0.1)(attention)
        
        # Global pooling
        pooled = layers.GlobalAveragePooling1D()(attention)
        
        # Dense блоки
        x = layers.Dense(128, activation='relu')(pooled)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
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
    
    def create_model(self) -> keras.Model:
        """Створення моделі згідно з типом"""
        if self.model_type == "transformer_lstm":
            model = self.build_transformer_lstm_model()
        elif self.model_type == "advanced_lstm":
            model = self.build_advanced_lstm_model()
        elif self.model_type == "cnn_lstm":
            model = self.build_cnn_lstm_model()
        else:
            raise ValueError(f"Невідомий тип моделі: {self.model_type}")
        
        logger.info(f"✅ Створена модель типу: {self.model_type}")
        logger.info(f"📊 Параметрів в моделі: {model.count_params():,}")
        
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
            loss='huber',  # Huber loss більш стійкий до викидів
            metrics=['mae', mape, directional_accuracy],
            jit_compile=self.use_xla
        )
        
        return model
    
    def get_callbacks(self, 
                     model_save_path: str,
                     patience: int = 50,
                     reduce_lr_patience: int = 20) -> List[callbacks.Callback]:
        """Створення оптимізованих callback'ів"""
        
        callback_list = [
            # Early stopping
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
            
            # Learning rate scheduler
            callbacks.LearningRateScheduler(
                lambda epoch: 0.001 * (0.95 ** epoch) if epoch > 100 else 0.001,
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
                   learning_rate: float = 0.001) -> Tuple[keras.Model, keras.callbacks.History]:
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
        
        # Тренування
        logger.info(f"🚀 Початок тренування моделі {self.model_type}")
        
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callback_list,
            verbose=1
        )
        
        logger.info("✅ Тренування завершено")
        
        return model, history
    
    def save_model_with_metadata(self, 
                                model: keras.Model, 
                                save_path: str, 
                                metadata: Dict = None):
        """Збереження моделі з метаданими"""
        try:
            # Зберігаємо модель
            model.save(save_path, save_format='keras')
            
            # Зберігаємо метадані
            if metadata:
                metadata_path = save_path.replace('.keras', '_metadata.json')
                import json
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            logger.info(f"✅ Модель збережена: {save_path}")
            
        except Exception as e:
            logger.error(f"❌ Помилка збереження моделі: {e}")
            raise
    
    def load_model_with_custom_objects(self, model_path: str) -> keras.Model:
        """Завантаження моделі з кастомними об'єктами"""
        try:
            model = keras.models.load_model(model_path, custom_objects=self.custom_objects)
            logger.info(f"✅ Модель завантажена: {model_path}")
            return model
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

# Фабричні функції
def create_transformer_lstm_model(input_shape: Tuple[int, int]) -> OptimizedPricePredictionModel:
    """Створення Transformer-LSTM моделі"""
    return OptimizedPricePredictionModel(input_shape, "transformer_lstm")

def create_advanced_lstm_model(input_shape: Tuple[int, int]) -> OptimizedPricePredictionModel:
    """Створення покращеної LSTM моделі"""
    return OptimizedPricePredictionModel(input_shape, "advanced_lstm")

def create_cnn_lstm_model(input_shape: Tuple[int, int]) -> OptimizedPricePredictionModel:
    """Створення CNN-LSTM моделі"""
    return OptimizedPricePredictionModel(input_shape, "cnn_lstm")