

import tensorflow as tf

def get_weight(shape,REGULARIZER=None):
    w=tf.Variable(tf.random_normal(shape),dtype=tf.float32)
    if REGULARIZER!=None:
        tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(REGULARIZER)(w))
    return w

def get_bias(shape):
    b=tf.Variable(tf.zeros(shape),dtype=tf.float32)
    return b

def forward(x,INPUT_NODE,LAYER_NODE,OUTPUT_NODE,REGULARIZER=None):
    w1=get_weight([INPUT_NODE,LAYER_NODE],REGULARIZER)
    b1=get_bias([LAYER_NODE])
    y1=tf.nn.relu(tf.matmul(x,w1)+b1)

    w2=get_weight([LAYER_NODE,OUTPUT_NODE],REGULARIZER)
    b2=get_bias([OUTPUT_NODE])

    y=tf.matmul(y1,w2)+b2
    return y