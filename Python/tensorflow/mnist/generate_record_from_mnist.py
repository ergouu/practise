import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np

MNIST_PATH="/Users/ergouu/Documents/ergouu-github/practise/Python/tensorflow/mnist/mnist_data/"
TFRECORD_PATH="/Users/ergouu/Documents/ergouu-github/practise/Python/tensorflow/mnist/tfrecord/"
TFRECORD_CAPCITY=60000


def generate_tfRecord():
    if not os.path.exists(TFRECORD_PATH):
        os.mkdir(TFRECORD_PATH)
    mnist=input_data.read_data_sets(MNIST_PATH,one_hot=True)
    writer = tf.python_io.TFRecordWriter(TFRECORD_PATH + "_mnist_test")
    for i in range(60000):
        """
        #read train data
        img_raw,labels=mnist.train.next_batch(1)
        img_raw=img_raw.tobytes()
        labels=labels.tobytes()
        """

        #read test data
        img_raw=mnist.test.images.tobytes()
        labels=mnist.test.labels.tobytes()

        example=tf.train.Example(features=tf.train.Features(feature=
                    {
                        'img_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                        'labels':tf.train.Feature(bytes_list=tf.train.BytesList(value=[labels]))
                    }
                )
            )
        writer.write(example.SerializeToString())
        print(i,"pic(s) have been saved!")
    writer.close()


def get_record(isTrain=True):
    if isTrain:
        filename_queue=tf.train.string_input_producer([TFRECORD_PATH+"_mnist"])
    else:
        filename_queue=tf.train.string_input_producer([TFRECORD_PATH+"_mnist_test"])

    reader=tf.TFRecordReader()
    _,serialized_example=reader.read(filename_queue)

    features=tf.parse_single_example(serialized_example,features={
        'label':tf.FixedLenFeature([],dtype=tf.string),
        'img_raw':tf.FixedLenFeature([], dtype=tf.string)
    })
    img=tf.decode_raw(features['img_raw'],tf.float32)
    label=tf.decode_raw(features['label'],tf.float32)
    img.set_shape([784])
    label.set_shape([10])
    return img,label

def get_batch(BATCH_SIZE):
    img,label=get_record()
    img_batch,label_batch=tf.train.shuffle_batch([img,label],
                                                 batch_size=BATCH_SIZE,
                                                 capacity=TFRECORD_CAPCITY,
                                                 min_after_dequeue=1000,
                                                 num_threads=2
                                                 )
    return img_batch,label_batch


def main():

    #generate_tfRecord()
    get_record()

if __name__=='__main__':
    main()
