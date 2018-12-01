
import tensorflow as tf
import nn_aux as aux

bias_shape = (200, 1)  # TODO: SO: verify if this is the real value
weight_shape = (200, 200)   # TODO: SO: verify if this is the real value

def __get_weight_n_bias():
    return aux.init_weight(weight_shape), aux.init_weight(weight_shape), aux.init_bias(bias_shape)


def lstm_cell(a_x, a_c, a_h, a_isDropout, a_isTraining):
    if(a_isDropout): #TODO: SO: not sure if that's the value to be dropped out
        a_x = tf.layers.dropout(a_x, aux.DROPOUT_RATE, training=a_isTraining)

    w_xi, w_hi, b_i = __get_weight_n_bias()
    i = tf.matmul(a_x, w_xi) + tf.matmul(a_h, w_hi) + b_i
    i = tf.nn.sigmoid(i)

    w_xf, w_hf, b_f = __get_weight_n_bias()
    f = tf.matmul(a_x, w_xf) + tf.matmul(a_h, w_hf) + b_f
    f = tf.nn.sigmoid(f)

    w_xo, w_ho, b_o = __get_weight_n_bias()
    o = tf.matmul(a_x, w_xo) + tf.matmul(a_h, w_ho) + b_o
    o = tf.nn.sigmoid(o)

    w_xg, w_hg, b_g = __get_weight_n_bias()
    g = tf.matmul(a_x, w_xg) + tf.matmul(a_h, w_hg) + b_g
    g = tf.nn.tanh(g)

    c = tf.math.multiply(f, a_c) + tf.math.multiply(i, g)
    h = tf.math.multiply(o, tf.nn.tanh(c))

    return h, c

def gru_cell(a_x, a_h, a_isDropout, a_isTraining): # TODO: SO: reduce code volume by avoiding copy paste of code blocks

    if(a_isDropout): #TODO: SO: not sure if that's the value to be dropped out
        a_x = tf.layers.dropout(a_x, aux.DROPOUT_RATE, training=a_isTraining)

    w_xz, w_hz, b_z = __get_weight_n_bias()
    z = tf.matmul(a_x, w_xz) + tf.matmul(a_h, w_hz) + b_z
    z = tf.nn.sigmoid(z)

    w_xr, w_hr, b_r = __get_weight_n_bias()
    r = tf.matmul(a_x, w_xr) + tf.matmul(a_h, w_hr) + b_r
    r = tf.nn.sigmoid(r)

    w_xg, w_hg, b_g = __get_weight_n_bias()
    g = tf.matmul(w_xg, a_x) + tf.matmul(w_hg, tf.math.multiply(r, a_h)) + b_g
    g = tf.nn.tanh(g)

    h = tf.math.multiply(z, a_h)
    ones_vect = tf.ones(bias_shape)
    z = tf.math.negative(z)
    h = h + tf.math.multiply((ones_vect+z), g)
    return h

def train_lstm(a_isDropout, a_isTraining):

    ########### PlaceHolders ############
    X = tf.placeholder(tf.float32, shape=[None, HEIGHT, WIDTH, 1], name='X')
    # Y = tf.placeholder(tf.float32, [None, 10], name='Y')
    # Train_Flag = tf.placeholder(tf.bool, name='Train_Flag')
    # ####################################
    #
    # ########### Model ##################
    # logits = LeNet(X, drop_flag, bn_flag, Train_Flag)
    # ####################################



# def lstm_cell():
#   return tf.contrib.rnn.BasicLSTMCell(lstm_size)
# stacked_lstm = tf.contrib.rnn.MultiRNNCell(
#     [lstm_cell() for _ in range(number_of_layers)])
#
# def make_lstm_rnn( a_isDropout ):
#     return tf.contrib.rnn.MultiRNNCell(
#         [make_cell() for _ in range(config.num_layers)], state_is_tuple=True)
