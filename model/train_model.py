import tensorflow as tf

def train_model(train_data, val_data, video_features, tokenizer, model, batch_size, epochs, learning_rate):
    train_dataset = tf.data.Dataset.from_tensor_slices(((train_data[0], train_data[1]), train_data[1])).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices(((val_data[0], val_data[1]), val_data[1])).batch(batch_size)
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        learning_rate, decay_steps=1000, decay_rate=0.96, staircase=True
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss=masked_loss)
    
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)
    return history

