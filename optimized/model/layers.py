"""
Custom Keras Layers

Спеціалізовані шари для моделі:
- TransformerBlock: Multi-Head Attention блок
- PositionalEncoding: Позиційне кодування для Transformer
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers


class TransformerBlock(layers.Layer):
    """
    Transformer блок з Multi-Head Attention
    
    Містить:
    - Multi-Head Attention
    - Feed-Forward Network
    - Layer Normalization
    - Residual connections
    """
    
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        """
        Ініціалізація
        
        Args:
            embed_dim: Розмірність embedding
            num_heads: Кількість attention heads
            ff_dim: Розмірність feed-forward network
            rate: Dropout rate
        """
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
        """Forward pass"""
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        """Config для серіалізації"""
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config


class PositionalEncoding(layers.Layer):
    """
    Позиційне кодування для Transformer
    
    Додає інформацію про позицію в послідовності
    """
    
    def __init__(self, maxlen, embed_dim, **kwargs):
        """
        Ініціалізація
        
        Args:
            maxlen: Максимальна довжина послідовності
            embed_dim: Розмірність embedding
        """
        super().__init__(**kwargs)
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        
    def build(self, input_shape):
        """Створення trainable weights"""
        self.pos_encoding = self.add_weight(
            name="pos_encoding",
            shape=(self.maxlen, self.embed_dim),
            initializer="uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        """Forward pass"""
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        return inputs + tf.gather(self.pos_encoding, positions)

    def get_config(self):
        """Config для серіалізації"""
        config = super().get_config()
        config.update({
            "maxlen": self.maxlen,
            "embed_dim": self.embed_dim,
        })
        return config


__all__ = [
    'TransformerBlock',
    'PositionalEncoding',
]
