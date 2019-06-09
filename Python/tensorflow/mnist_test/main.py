from tensorflow.examples.tutorials.mnist import input_data
import backforward
import test
import app

mnist = input_data.read_data_sets("./data/",one_hot=True)
INPUT_NODE=784
LAYER_NODE=500
OUTPUT_NODE=10
DATASET_SIZE=mnist.train.num_examples
MOVING_AVERAGE_DECAY=0.99




def main():
    # train 50000 steps on every calling
    #backforward.backforward(mnist,INPUT_NODE,LAYER_NODE,OUTPUT_NODE,DATASET_SIZE,MOVING_AVERAGE_DECAY)
    #test.test(mnist,INPUT_NODE,LAYER_NODE,OUTPUT_NODE,MOVING_AVERAGE_DECAY)
    app.app(INPUT_NODE,LAYER_NODE,OUTPUT_NODE,MOVING_AVERAGE_DECAY)


if __name__=='__main__':
    main()