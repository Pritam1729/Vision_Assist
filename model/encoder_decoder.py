from tensorflow.keras.layers import (
    Input, LSTM, Dense, MultiHeadAttention, Concatenate,
    Dropout, LayerNormalization, Embedding
)
from tensorflow.keras.models import Model

def build_model(vocab_size, max_length, units, embedding_dim, dropout_rate, key_dim, num_heads):
    video_input = Input(shape=(240, 4096))
    encoder_lstm = LSTM(units, return_sequences=True, return_state=True, dropout=dropout_rate, recurrent_dropout=dropout_rate)
    encoder_lstm_output, state_h, state_c = encoder_lstm(video_input)

    encoder_attention = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
    encoder_attention_output = encoder_attention(query=encoder_lstm_output, value=encoder_lstm_output, key=encoder_lstm_output)

    encoder_combined_output = Concatenate(axis=-1)([encoder_lstm_output, encoder_attention_output])
    encoder_output = LayerNormalization()(encoder_combined_output)
    encoder_projection = Dense(units, activation='linear')(encoder_output)

    caption_input = Input(shape=(max_length,))
    embedding_layer = Embedding(vocab_size, embedding_dim, trainable=True)
    caption_embeddings = embedding_layer(caption_input)

    decoder_lstm = LSTM(units, return_sequences=True, return_state=True, dropout=dropout_rate, recurrent_dropout=dropout_rate)
    decoder_output, _, _ = decoder_lstm(caption_embeddings, initial_state=[state_h, state_c])

    multihead_attention = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
    attention_output = multihead_attention(query=decoder_output, value=encoder_projection, key=encoder_projection)

    decoder_combined_context = Concatenate(axis=-1)([attention_output, decoder_output])
    dense_output = Dense(vocab_size, activation='softmax')(decoder_combined_context)

    model = Model(inputs=[video_input, caption_input], outputs=dense_output)
    return model

