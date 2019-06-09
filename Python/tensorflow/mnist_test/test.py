import time
import tensorflow as tf
import forward


TEST_INTERVAL_SECS=5

def test(mnist,INPUT_NODE,LAYER_NODE,OUTPUT_NODE,MOVING_AVERAGE_DECAY):
    with tf.Graph().as_default() as g:
        x=tf.placeholder(tf.float32,shape=(None,INPUT_NODE))
        y_=tf.placeholder(tf.float32,shape=(None,OUTPUT_NODE))
        y = forward.forward(x,INPUT_NODE,LAYER_NODE,OUTPUT_NODE)

        ema=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        ema_restore=ema.variables_to_restore()
        saver=tf.train.Saver(ema_restore)

        correct_perdiction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy=tf.reduce_mean(tf.cast(correct_perdiction,tf.float32))

        while True:
            with tf.Session() as sess:
                ckpt=tf.train.get_checkpoint_state("/Users/ergouu/Documents/ergouu-github/practise/Python/tensorflow/mnist_test/model/")
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    #print("ckpt path is ",ckpt.model_checkpoint_path)
                    global_step=ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score=sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels})
                    print("After %s training step(s),test accuracy = %g" %(global_step,accuracy_score))
                else:
                    print("No checkpoint file")
                    return
            time.sleep(TEST_INTERVAL_SECS)