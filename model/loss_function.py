import tensorflow as tf

def masked_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    vocab_size = tf.shape(y_pred)[-1]
    y_true_one_hot = tf.one_hot(y_true, depth=vocab_size)
    mask = tf.cast(tf.not_equal(y_true, 0), dtype=tf.float32)
    loss = tf.keras.losses.categorical_crossentropy(y_true_one_hot, y_pred)
    loss *= mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)

