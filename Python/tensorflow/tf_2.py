import tensorflow as tf
import numpy as np

rng = np.random.RandomState(8)

X = rng.rand(2048, 2)

Y = [[int(x0 + x1 < 1)] for (x0, x1) in X]

print("X:\n", X)
print("Y:\n", Y)

x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

loss = tf.reduce_mean(tf.square(y - y_))
# # train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

# # train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

train_step = tf.train.MomentumOptimizer(0.001, 0.9).minimize(loss)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    print("W1:\n", sess.run(w1))
    print("W2:\n", sess.run(w2))
    print()

    STEPS = 30000
    for i in range(STEPS):
        start = (i*256) % 2048
        end = start + 256
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 500 == 0:
            total_loss = sess.run(loss, feed_dict={x: X, y_: Y})
            print("After %d step(s), loss on all data is %g" % (i, total_loss))
    print("W1:\n", sess.run(w1))
    print("W2:\n", sess.run(w2))
    print()
