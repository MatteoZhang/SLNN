import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np

import sys
import argparse

# global flags for convenience
FLAGS=None

# Parameters
NUM_PIXELS = 784
NUM_CLASSES = 10
BATCH_SIZE = 50
TRAIN_STEPS = 36000 
NUM_HIDDEN_1 = 350
NUM_HIDDEN_2 = 100


def train_and_test(_):

	# Check if log_dir exists, if so delete contents
    if tf.gfile.Exists(FLAGS.log_dir):
            tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)

	# Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

#################################################################################
############################	YOUR CODE HERE   ################################


	# define placeholders for batch of training images and labels
    x=tf.placeholder(tf.float32, shape=(None,784), name='input_images')
    y=tf.placeholder(tf.float32, shape=(None,10), name='output')
    

	# define variables for weights and biases of the three fully connected layers
    w1=tf.Variable(tf.truncated_normal(shape=(784,NUM_HIDDEN_1), stddev=0.1, dtype=tf.float32), name='weight1')
    w2=tf.Variable(tf.truncated_normal(shape=(NUM_HIDDEN_1,NUM_HIDDEN_2), stddev=0.1, dtype=tf.float32), name='weight2')
    w3=tf.Variable(tf.truncated_normal(shape=(NUM_HIDDEN_2,10), stddev=0.1, dtype=tf.float32), name='weight3')
    #w è 784xnum_neuroni, diverso per ogni step, ho 4 layer (input, hidden1, hidden2, output) quindi 3 w
    b1=tf.Variable(tf.truncated_normal(shape=(1,NUM_HIDDEN_1), stddev=0.1), name='bias1')
    b2=tf.Variable(tf.truncated_normal(shape=(1,NUM_HIDDEN_2), stddev=0.1), name='bias2')
    b3=tf.Variable(tf.truncated_normal(shape=(1,10), stddev=0.1), name='bias3')


	# computation graph
    for i in range (1,BATCH_SIZE):
        h1=x*w1+b1
        h2=h1*w2+b2
        h3=h2*w3+b3

	# define loss function
    loss=tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=h3)

	# make the loss a "summary" to visualise it in tensorboard
    tf.summary.scalar('loss',loss)

	# define the optimizer and what is optimizing
    optimizer=tf.train.GradientDescentOptimizer(3.0)
    train_step=optimizer.minimize(loss)

	# measure accuracy on the batch and make it a summary for tensorboard
    accuracy=1/20*sum(np.linalg.norm(y-h3))
    tf.summary.scalar('accuracy',accuracy)

	# create session
    sess=tf.InteractiveSession()

	# merge summaries for tensorboard
    merged=tf.summary.merge_all()
    train_writer=tf.summary.FileWriter(FLAGS.log_dir+'/train',sess.graph)
    test_writer=tf.summary.FileWriter(FLAGS.log_dir+'/t',sess.graph)        

	# initialize variables
    tf.global_variables_initializer().run()
	
	# training iterations: fetch training batch and run
    # devo fare abbastanza giri da completare almeno una decina di epoch
    for i in range (1,TRAIN_STEPS):
        batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
        summary_train,_=sess.run([merged,train_step],feed_dict={x:batch_xs,y:batch_ys})
        train_writer.add_summary(summary_train,i)        
        
        sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys})

	# after training fetch test set and measure accuracy
    for i in range (1,200):
        batch_xs, batch_ys = mnist.test.next_batch(BATCH_SIZE)
        accuracy_test=1/20*sum(np.linalg.norm(y-h3))
        tf.summary.scalar('accuracy',accuracy_test)
        
        summary_test,_=sess.run([merged,train_step],feed_dict={x:batch_xs,y:batch_ys})
        test_writer.add_summary(summary_test,i)        
        
        sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys})


###################################################################################		
	






if __name__ == '__main__':

	# use nice argparse module to aprte cli arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='.\data_dir', help='Directory for training data')
	parser.add_argument('--log_dir', type=str, default='.\log_dir', help='Directory for Tensorboard event files')
	FLAGS, unparsed = parser.parse_known_args()
	# app.run is a simple wrapper that parses flags and sends them to main function
	tf.app.run(main=train_and_test, argv=[sys.argv[0]] + unparsed)