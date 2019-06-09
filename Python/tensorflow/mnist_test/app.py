import tensorflow as tf
import forward
from PIL import Image
import numpy as np

def resore_model(testPicArr,INPUT_NODE,LAYER_NODE,OUTPUT_NODE,MOVING_AVERAGE_DECAY,REGULARIZER=0.0001):
    with tf.Graph().as_default() as g:
        x=tf.placeholder(dtype=tf.float32,shape=(None,INPUT_NODE))
        y=forward.forward(x,INPUT_NODE,LAYER_NODE,OUTPUT_NODE,REGULARIZER)

        preValue = tf.argmax(y,1)

        ema=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        ema_restore=ema.variables_to_restore(tf.trainable_variables())
        saver=tf.train.Saver(ema_restore)

        with tf.Session() as sess:
            ckpt=tf.train.get_checkpoint_state("/Users/ergouu/Documents/ergouu-github/practise/Python/tensorflow/mnist_test/model/")

            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess,ckpt.model_checkpoint_path)
                preValue=sess.run(preValue, feed_dict={x: testPicArr})
                return preValue
            else:
                print("No model havs been found!")
                return -1

def preprocess(picName):
    img=Image.open(picName)
    resize_img=img.resize((28,28),Image.ANTIALIAS)
    im_arr=np.array(resize_img.convert('L'))
    threshold=50
    for i in range(28):
        for j in range(28):
            im_arr[i][j]=255-im_arr[i][j]
            if im_arr[i][j]<threshold:
                im_arr[i][j]=0
            else:
                im_arr[i][j]=255
    nm_arr=im_arr.reshape([1,784])
    nm_arr=nm_arr.astype(np.float32)
    img_ready=np.multiply(nm_arr,1.0/255.0)
    return img_ready

def app(INPUT_NODE,LAYER_NODE,OUTPUT_NODE,MOVING_AVERAGE_DECAY):
    while True:
        pic=input(("input the path of the test pic:(\"q\" to quit)"))
        if pic=='q':
            return
        else:
            testpic=preprocess(pic)
            value=resore_model(testpic,INPUT_NODE,LAYER_NODE,OUTPUT_NODE,MOVING_AVERAGE_DECAY)
            print ("The prediction number is:",value)