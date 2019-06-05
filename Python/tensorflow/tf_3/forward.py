import tensorflow as tf
import generate_dataset as gd


def forward(l1_shape, l2_shape, regularizer):
    w1=gd.get_weight(regularizer,l1_shape)
    b1=gd.get_bias([l1_shape[1]])
    w2=gd.get_weight(regularizer,l2_shape)
    b2=gd.get_bias([l2_shape[1]])
    x = tf.placeholder(dtype=tf.float32, shape=(None, 2))
    #y1=tf.nn.relu(tf.matmul(x,w1)+b1)
    #y1 = tf.nn.sigmoid(tf.matmul(x, w1) + b1)
    y1 = tf.nn.tanh(tf.matmul(x, w1) + b1)
    y=tf.matmul(y1,w2)+b2
    return x,y


