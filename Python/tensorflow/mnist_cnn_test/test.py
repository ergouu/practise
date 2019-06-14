import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import fd
import bd
import time
import numpy as np

TEST_INTERVAL=15

def test(mnist):
    with tf.Graph().as_default() as g:
        x=tf.placeholder(dtype=tf.float32,shape=(
            10000,fd.IMAGE_SIZE,fd.IMAGE_SIZE,
            fd.NUM_CHANNAL
        ))
        y_=tf.placeholder(dtype=tf.float32,shape=(None,fd.OUTPUT_NODE))
        y=fd.forward(x,train=False)

        correct=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))

        ema=tf.train.ExponentialMovingAverage(bd.MOVING_AVERAGE)
        ema_restore=ema.variables_to_restore(tf.trainable_variables())
        saver=tf.train.Saver(ema_restore)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(bd.CHECKPOINT_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess=sess, save_path=ckpt.model_checkpoint_path)
                    print("Model %s loaeded!" % ckpt.model_checkpoint_path)
                else:
                    print("No model was found!")
                    return
                img=mnist.test.images
                img_reshape=np.reshape(img,(
                    10000,
                    fd.IMAGE_SIZE,fd.IMAGE_SIZE,
                    fd.NUM_CHANNAL
                ))
                acc_val=sess.run(accuracy,feed_dict={x:img_reshape,y_:mnist.test.labels})
                print("now the accuracy is:",acc_val)
            time.sleep(TEST_INTERVAL)

def main():
    mnist=input_data.read_data_sets("./data",one_hot=True)
    test(mnist)

if __name__ == '__main__':
    main()