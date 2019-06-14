import tensorflow as tf

IMAGE_SIZE=28
NUM_CHANNAL=1
CONV1_SIZE=5
CONV1_KERNEL_NUM=32
CONV2_SIZE=5
CONV2_KERNEL_NUM=64
FC_SIZE=500
OUTPUT_NODE=10
REGULARIZER=0.001


def get_weights(shape,REGULARIZER=None):
    w=tf.Variable(tf.truncated_normal(shape=shape,stddev=0.1),dtype=tf.float32)
    if REGULARIZER!=None:
        tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(REGULARIZER)(w))
    return w


def get_bias(shape):
    return tf.Variable(tf.zeros(shape=shape))

def get_controv(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

def get_max_pool(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def forward(x,train=True):
    conv1_w=get_weights([CONV1_SIZE,CONV1_SIZE,NUM_CHANNAL,CONV1_KERNEL_NUM],REGULARIZER)
    conv1_b=get_bias([CONV1_KERNEL_NUM])
    conv1=get_controv(x,conv1_w)
    activator1=tf.nn.relu(tf.nn.bias_add(conv1,conv1_b))
    y1=get_max_pool(activator1)

    conv2_w=get_weights([CONV2_SIZE,CONV2_SIZE,CONV1_KERNEL_NUM,CONV2_KERNEL_NUM],REGULARIZER)
    conv2_b=get_bias([CONV2_KERNEL_NUM])
    conv2=get_controv(y1,conv2_w)
    activator2=tf.nn.relu(tf.nn.bias_add(conv2,conv2_b))
    pool2=get_max_pool(activator2)

    pool_shape=pool2.get_shape().as_list()
    pool_nodes=pool_shape[1]*pool_shape[2]*pool_shape[3]
    reshaped=tf.reshape(pool2,[pool_shape[0],pool_nodes])


    fc_w1=get_weights([pool_nodes,FC_SIZE],REGULARIZER)
    fc_b1=get_bias([FC_SIZE])
    y1=tf.nn.relu(tf.matmul(reshaped,fc_w1)+fc_b1)
    if train:
        tf.nn.dropout(y1,0.5)

    fc_w2=get_weights([FC_SIZE,OUTPUT_NODE],REGULARIZER)
    fc_b2=get_bias([OUTPUT_NODE])
    y=tf.matmul(y1,fc_w2)+fc_b2

    return y