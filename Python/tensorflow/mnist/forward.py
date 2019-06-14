import tensorflow as tf

INPUT_NODE=784
LAYER_NODE=500
OUTPUT_NODE=10
REGULARIZOR=0.0001


def get_weighs(shape):
    w=tf.Variable(tf.truncated_normal(shape=shape,stddev=0.1),dtype=tf.float32)
    tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(REGULARIZOR)(w))
    return w


def get_bias(shape):
    b=tf.Variable(tf.zeros(shape=shape))
    return b

def forward(x):
    w1=get_weighs([INPUT_NODE,LAYER_NODE])
    b1=get_bias([LAYER_NODE])
    y1=tf.nn.relu(tf.matmul(x,w1)+b1)

    w2=get_weighs([LAYER_NODE,OUTPUT_NODE])
    b2=get_bias([OUTPUT_NODE])
    y=tf.matmul(y1,w2)+b2

    return y