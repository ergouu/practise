import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def generator(num):
    #seed=2
    #rdm=np.random.RandomState(seed)
    #X = rdm.randn(num, 2)
    X=np.random.randn(num,2)
    Y_ = [int((x0 * x0 + x1 * x1 < 2)) for (x0, x1) in X]
    Y_C = ['red' if i else 'blue' for i in Y_]
    X = np.vstack(X).reshape(-1, 2)
    Y_ = np.vstack(Y_).reshape(-1, 1)
    return X,Y_,Y_C


def plot_dataset(X,Y_C):
    plt.scatter(X[:,0],X[:,1],c=np.squeeze(Y_C))

def get_weight(regularizer,shape):
    w=tf.Variable(tf.random_normal(shape),dtype=tf.float32)
    tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):
    b=tf.Variable(tf.random_normal(shape),dtype=tf.float32)
    return b