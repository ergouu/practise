import tensorflow as tf

#x = tf.constant([[0.7, 0.5]])
x = tf.placeholder(tf.float32, shape=(None, 2))
w1 = tf.Variable(tf.random.normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random.normal([3, 1], stddev=1, seed=1))

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print("y in tf1.py is:", sess.run(y, feed_dict={x: [[0.3, 0.9], [0.4, 0.9], [0.7, 0.8]]}))
