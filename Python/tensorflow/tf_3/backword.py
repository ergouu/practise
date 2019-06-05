import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import forward
import generate_dataset as gd

REGULARIZER=0.01
STEPS=40000
BACH_SIZE=3000
DATASET_SIZE=30000
L1_shape=[2,11]
L2_shape=[11,1]
LEARNING_RATE_BASE=0.09
LEARNING_RATE_STEP=DATASET_SIZE/BACH_SIZE
LEARNING_RATE_DECAY_RATE=0.1

X,Y_,Y_C=gd.generator(DATASET_SIZE)


x,y=forward.forward(L1_shape,L2_shape,REGULARIZER)

y_=tf.placeholder(dtype=tf.float32,shape=(None,1))

loss_mse = tf.reduce_mean(tf.square(y-y_))
loss = loss_mse+tf.add_n(tf.get_collection('losses'))

global_step=tf.Variable(0,trainable=False)
learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,LEARNING_RATE_STEP,LEARNING_RATE_DECAY_RATE,staircase=True)

train_step=tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(STEPS):
        start=(i*BACH_SIZE)%DATASET_SIZE
        end=start+BACH_SIZE
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y_[start:end]})
        if i%5000==0:
            print("Step(s):",i,"loss is :",sess.run(loss,feed_dict={x:X,y_:Y_}))
    xx,yy=np.mgrid[-3:3:0.001,-3:3:0.001]
    grid=np.c_[xx.ravel(),yy.ravel()]
    probs=sess.run(y,feed_dict={x:grid})
    probs=probs.reshape(xx.shape)

plt.scatter(X[:,0],X[:,1],c=Y_C)
plt.contour(xx,yy,probs,levels=[0.5])
plt.show()