import  tensorflow as tf

def init_weight(shape):
    w = tf.truncated_normal(shape=shape, mean=0, stddev=0.1)
    return tf.Variable(w, trainable=True)


def init_bias(shape):
    b = tf.zeros(shape)
    return tf.Variable(b, trainable=True)