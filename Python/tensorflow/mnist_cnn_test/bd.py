import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import fd
import numpy as np

BATCH_SIZE=3000
LERNING_RATE_BASE=0.1
LERNING_RATE_DECAY_RATE=0.8
MOVING_AVERAGE=0.99
STEPS=50000
CHECKPOINT_PATH="/Users/ergouu/Documents/ergouu-github/practise/Python/tensorflow/mnist_cnn_test/model/"


def backward(mnist):
    x=tf.placeholder(dtype=tf.float32,shape=(BATCH_SIZE,fd.IMAGE_SIZE,fd.IMAGE_SIZE,fd.NUM_CHANNAL))
    y_=tf.placeholder(dtype=tf.float32,shape=(None,fd.OUTPUT_NODE))

    y=fd.forward(x)

    global_step=tf.Variable(0,trainable=False)

    learning_rate=tf.train.exponential_decay(learning_rate=LERNING_RATE_BASE,
                                             staircase=True,
                                             decay_rate=LERNING_RATE_DECAY_RATE,
                                             decay_steps=mnist.train.num_examples/BATCH_SIZE,
                                             global_step=global_step
                                             )

    ce=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_,1),logits=y)
    cem=tf.reduce_mean(ce)
    loss=cem+tf.add_n(tf.get_collection('losses'))

    train=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

    ema=tf.train.ExponentialMovingAverage(MOVING_AVERAGE,global_step)
    ema_op=ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train,ema_op]):
        train_op=tf.no_op(name='train')

    saver=tf.train.Saver()

    with tf.Session() as sess:
        init_op=tf.global_variables_initializer()
        sess.run(init_op)

        ckpt=tf.train.get_checkpoint_state(CHECKPOINT_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess=sess,save_path=ckpt.model_checkpoint_path)

        for i in range(STEPS):
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            xs_reshape=np.reshape(xs,(
                BATCH_SIZE,
                fd.IMAGE_SIZE,
                fd.IMAGE_SIZE,
                fd.NUM_CHANNAL
            ))
            _,loss_val,step=sess.run([train_op,loss,global_step],feed_dict={x:xs_reshape,y_:ys})

            if i%10==0:
                print("After %d training step(s), loss is %g" %(step,loss_val))
                saver.save(sess=sess,save_path=CHECKPOINT_PATH+"_CNN_MNIST",global_step=global_step)


def main():
    mnist=input_data.read_data_sets("./data/",one_hot=True)
    backward(mnist)

if __name__=='__main__':
    main()

