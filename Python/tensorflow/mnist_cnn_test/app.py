import fd
import bd
import tensorflow as tf
from PIL import Image
import numpy as np

def pre_proc(img_name):
    img=Image.open(img_name)
    img_resized=img.resize((28,28),Image.ANTIALIAS)
    img_arr=np.array(img_resized.convert('L'))
    threshold=50
    for i in range(28):
        for j in range(28):
            img_arr[i][j]=255-img_arr[i][j]
            if img_arr[i][j]<threshold:
                img_arr[i][j]=0
            else:
                img_arr[i][j]=255
    nm_arr=img_arr.reshape([1,28,28,1])
    nm_arr=nm_arr.astype(np.float32)
    img_ready=np.multiply(nm_arr,1./255)
    return img_ready


def hand_writing_recog(pic_name):
    with tf.Graph().as_default() as g:
        x=tf.placeholder(dtype=tf.float32,shape=(
            1,
            fd.IMAGE_SIZE,fd.IMAGE_SIZE,
            fd.NUM_CHANNAL
        ))
        y=fd.forward(x,train=False)

        pre_val=tf.argmax(y,1)

        ema=tf.train.ExponentialMovingAverage(bd.MOVING_AVERAGE)
        ema_restore=ema.variables_to_restore(tf.trainable_variables())

        saver=tf.train.Saver(ema_restore)

        with tf.Session() as sess:
            ckpt=tf.train.get_checkpoint_state(bd.CHECKPOINT_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess=sess,save_path=ckpt.model_checkpoint_path)
            else:
                print("No model was found!")
                return

            xs=pre_proc(pic_name)
            val=sess.run(pre_val,feed_dict={x:xs})
            print("The prediction is :%d" %val)


def main():
    path = input("Plesa input the path of the picture:")
    if path[-1]!='/':
        path+='/'
    while True:
        picname=input("Plesa input the name of the picture:")
        picname=path+picname
        hand_writing_recog(picname)
        c=input("Do you want to continue?(Y/n)")
        if c=='Y' or c=='':
            continue
        else:
            break

if __name__=='__main__':
    main()
