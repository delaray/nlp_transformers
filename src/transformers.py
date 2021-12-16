# *********************************************************************
# Tranformer Example Using Keras Embedding & Transformer Layers
#
# *********************************************************************
#
# Methodology Outline
# 
# ▻ Implement a Transformer block as a layer
# ▻ Implement embedding layer
# ▻ Download and prepare dataset
# ▻ Create classifier model using transformer layer
# ▻ Train and Evaluate
#
# *********************************************************************
#
# Source: https://keras.io/examples/nlp/text_classification_with_transformer/
#
# *********************************************************************

# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ---------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------

# Implement a Transformer block as a layer

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads,
                                             key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"),
             layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


# ---------------------------------------------------------------------
# Implement embedding layer
# ---------------------------------------------------------------------

# Two seperate embedding layers, one for tokens, one for token
# index (positions).

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size,
                                          output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


# ---------------------------------------------------------------------
# Download and prepare dataset
# ---------------------------------------------------------------------

# Only consider the top 20k words
VOCAB_SIZE = 20000

# Only consider the first 200 words of each movie review
MAXLEN = 200


def get_and_prepare_data(vocab_size=VOCAB_SIZE):
    (x_train, y_train), (x_val, y_val) = \
        keras.datasets.imdb.load_data(num_words=vocab_size)
    print(len(x_train), "Training sequences")
    print(len(x_val), "Validation sequences")
    x_train = \
        keras.preprocessing.sequence.pad_sequences(x_train, maxlen=MAXLEN)
    x_val = \
        keras.preprocessing.sequence.pad_sequences(x_val, maxlen=MAXLEN)
    return x_train, y_train, x_val, y_val


# ----------------------------------------------------------------------
# Create classifier model using transformer layer
# ----------------------------------------------------------------------

# Transformer layer outputs one vector for each time step of our input
# sequence. Here, we take the mean across all time steps and use a
# feed forward network on top of it to classify text.

# Embedding size for each token
EMBED_DIM = 32


# Number of attention heads
NUM_HEADS = 2


# Hidden layer size in feed forward network inside transformer
FF_DIM = 32  


def create_keras_model():
    inputs = layers.Input(shape=(MAXLEN,))
    embedding_layer = TokenAndPositionEmbedding(MAXLEN, VOCAB_SIZE, EMBED_DIM)

    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(EMBED_DIM, NUM_HEADS, FF_DIM)

    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)

    outputs = layers.Dense(2, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model


# ----------------------------------------------------------------------
# Train and Evaluate
# ----------------------------------------------------------------------

def train_keras_model(model, x_train, y_train, x_val, y_val):
    model.compile("adam", "sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    history = model.fit(x_train, y_train, batch_size=32, epochs=2,
                        validation_data=(x_val, y_val))

    return history

# ----------------------------------------------------------------------
# Endo of File
# ----------------------------------------------------------------------
