import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#数据集
BACH_SIZE =30
DATASET_SIZE = 300
seed = 2

rdm=np.random.RandomState(seed)

X=rdm.randn(DATASET_SIZE,2)

Y_=[int(x0*x0+x1*x1<2) for (x0,x1) in X]

Y_C=[['red' if y else 'blue'] for y in Y_]

X=np.vstack(X).reshape(-1,2)

Y_=np.vstack(Y_).reshape(-1,1)

plt.scatter(X[:,0],X[:,1],c=np.squeeze(Y_C))
plt.show()


#前向传播

def get_weight(shape,regularizer):
    w=tf.Variable(tf.random_normal(shape),dtype=tf.float32)
    tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):
    b=tf.Variable(tf.constant(0.01,shape=shape))
    #b = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    return b

x=tf.placeholder(dtype=tf.float32,shape=(None,2))
y_=tf.placeholder(dtype=tf.float32,shape=(None,1))

w1=get_weight([2,11],0.01)
b1=get_bias([11])
w2=get_weight([11,1],0.01)
b2=get_bias([1])

y1=tf.nn.relu(tf.matmul(x,w1)+b1)
y=tf.matmul(y1,w2)+b2

#后向传播

loss=tf.reduce_mean(tf.square(y-y_))+tf.add_n(tf.get_collection('losses'))

train_step=tf.train.AdamOptimizer(0.0001).minimize(loss)

with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    STEPS=80000
    for i in range(STEPS):
        start=(i*BACH_SIZE) % DATASET_SIZE
        end=start+BACH_SIZE
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y_[start:end]})
        if i%2000==0:
            print("Step:",i,"loss is:",sess.run(loss,feed_dict={x:X,y_:Y_}))

    xx,yy=np.mgrid[-3:3:0.01,-3:3:0.01]
    grid=np.c_[xx.ravel(),yy.ravel()]
    probs=sess.run(y,feed_dict={x:grid})
    probs=probs.reshape(xx.shape)

plt.scatter(X[:,0],X[:,1],c=np.squeeze(Y_C))
plt.contour(xx,yy,probs,levels=[0.5])
plt.show()