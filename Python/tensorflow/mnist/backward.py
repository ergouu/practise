import tensorflow as tf
import forward
import generate_record_from_mnist as grfm

LEARNING_RATE_BASE=0.1
BATCH_SIZE=3000
LEARNING_RATE_DECAY_RATE=0.5
MOVING_AVERAGE_DECAY=0.99
MODEL_SAVE_PATH="/Users/ergouu/Documents/ergouu-github/practise/Python/tensorflow/mnist/model"
TRAIN_STEPS=50000

def backward():
    x=tf.placeholder(dtype=tf.float32,shape=(None,forward.INPUT_NODE))
    y_=tf.placeholder(dtype=tf.float32,shape=(None,forward.OUTPUT_NODE))

    y=forward.forward(x)

    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem=tf.reduce_mean(ce)
    loss=cem+tf.add_n(tf.get_collection('losses'))

    global_step=tf.Variable(0,trainable=False)

    learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE,global_step=global_step,
                                             decay_steps=grfm.TFRECORD_CAPCITY/BATCH_SIZE,
                                             decay_rate=LEARNING_RATE_DECAY_RATE,staircase=True
                                             )

    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

    img_raw,label=grfm.get_batch(BATCH_SIZE)

    saver=tf.train.Saver()

    ema=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    ema_op=ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step,ema_op]):
        train_op=tf.no_op(name='train')

    with tf.Session() as sess:
        init_op=tf.global_variables_initializer()
        sess.run(init_op)

        ckpt=tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess=sess,save_path=ckpt.model_checkpoint_path)

        coord=tf.train.Coordinator()
        threads=tf.train.start_queue_runners(sess=sess,coord=coord)

        for i in range(TRAIN_STEPS):
            xs,ys=sess.run([img_raw,label])
            _,steps,loss_val=sess.run([train_op,global_step,loss],feed_dict={x:xs,y_:ys})

            if(i%3000)==0:
                print("Step(s):",steps,"loss is:",loss_val)
                saver.save(sess=sess,save_path=MODEL_SAVE_PATH,global_step=global_step)
        coord.request_stop()
        coord.join(threads)


def main():
    backward()

if __name__=='__main__':
    main()