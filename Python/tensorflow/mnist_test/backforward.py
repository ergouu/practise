import tensorflow as tf
import forward



def backforward(mnist,INPUT_NODE,LAYER_NODE,OUTPUT_NODE,DATASET_SIZE,MOVING_AVERAGE_DECAY,LEARNING_RATE_BASE=0.1,LEARNING_RATE_DECAY_RATE=0.99,TRAIN_STPES=50000,BATCH_SIZE=200,REGULARIZER=0.0001):
    x = tf.placeholder(dtype=tf.float32,shape=(None,INPUT_NODE))
    y_= tf.placeholder(dtype=tf.float32,shape=(None,OUTPUT_NODE))

    y=forward.forward(x,INPUT_NODE,LAYER_NODE,OUTPUT_NODE,REGULARIZER)

    global_steps=tf.Variable(0,trainable=False)

    ce=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cem=tf.reduce_mean(ce)
    loss= cem+tf.add_n(tf.get_collection('losses'))

    learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE,global_steps,DATASET_SIZE/BATCH_SIZE,LEARNING_RATE_DECAY_RATE,staircase=True)

    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_steps)

    ema=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_steps)
    ema_op=ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step,ema_op]):
        train_op=tf.no_op(name='train')

    saver=tf.train.Saver()

    with tf.Session() as sess:
        init_op=tf.global_variables_initializer()
        sess.run(init_op)
        # train after checkpoint
        ckpt=tf.train.get_checkpoint_state("/Users/ergouu/Documents/ergouu-github/practise/Python/tensorflow/mnist_test/model/")
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpoint_path)
        else:
            print("No checkpoint was found!")
        # train after checkpoint end

        for i in range(TRAIN_STPES):
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            _,loss_value,step=sess.run([train_op,loss,global_steps],feed_dict={x:xs,y_:ys})
            if i%1000==0:
                print("Step(s): %d, loss on training batch is %g." %(step,loss_value))
                #saver.save(sess,os.path.join("./model/","mnist_model"),global_step=global_steps)
                saver.save(sess,"/Users/ergouu/Documents/ergouu-github/practise/Python/tensorflow/mnist_test/model/mnist_model",global_step=global_steps)