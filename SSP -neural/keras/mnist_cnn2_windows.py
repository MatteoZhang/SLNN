import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras import models
from tensorflow.contrib.keras import initializers

import sys
import argparse

# global flags for convenience
FLAGS = None

# Parameters
NUM_PIXELS = 28
NUM_CLASSES = 10
BATCH_SIZE = 100
TRAIN_STEPS = 5000

NUM_FILTERS_1 = 100
NUM_FILTERS_2 = 50


def classifier_model():
    model = models.Sequential()
    model.add(
        layers.Conv2D(NUM_FILTERS_1, [3, 3], strides=(2, 2), padding='same', activation='relu', input_shape=(28, 28, 1),
                      kernel_initializer=initializers.glorot_normal(), bias_initializer=initializers.Zeros()))
    model.add(layers.Conv2D(NUM_FILTERS_2, [3, 3], strides=(2, 2), padding='same', activation='relu',
                            kernel_initializer=initializers.glorot_normal(), bias_initializer=initializers.Zeros()))
    model.add(layers.Flatten())
    model.add(layers.Dense(NUM_CLASSES, kernel_initializer=initializers.glorot_normal(),
                           bias_initializer=initializers.Zeros()))

    return model


def train_and_test(_):
    # Check if log_dir exists, if so delete contents
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)

    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True, reshape=False)

    #################################################################################
    ############################	YOUR CODE HERE   ################################

    # define placeholders for batch of training images and labels
    x = tf.placeholder(tf.float32, shape=(None, NUM_PIXELS, NUM_PIXELS, 1), name='x')
    y = tf.placeholder(tf.float32, shape=(None, NUM_CLASSES), name='y')

    # create model
    my_net = classifier_model()
    my_net.summary()

    # use model on input image batch to compute logits
    h = my_net(x)

    # define loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=h))

    # make the loss a "summary" to visualise it in tensorboard
    tf.summary.scalar('loss', loss)

    # define the optimizer and what is optimizing
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train_step = optimizer.minimize(loss)

    # measure accuracy on the batch and make it a summary for tensorboard
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(h, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # create session
    sess = tf.InteractiveSession()

    # merge summaries for tensorboard
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)

    # initialize variables
    tf.global_variables_initializer().run()

    # training iterations: fetch training batch and run
    for i in range(1, TRAIN_STEPS):
        batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
        summary_train, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y: batch_ys})
        train_writer.add_summary(summary_train, i)

    # after training fetch test set and measure accuracy and print it to screen
    batch_xs, batch_ys = mnist.test.next_batch(10000)
    test_accuracy = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys})
    print('Test accuracy: %.4f' % test_accuracy)


###################################################################################		


if __name__ == '__main__':
    # use nice argparse module to aprte cli arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='.\data_dir', help='Directory for training data')
    parser.add_argument('--log_dir', type=str, default='.\log_dir', help='Directory for Tensorboard event files')
    FLAGS, unparsed = parser.parse_known_args()
    # app.run is a simple wrapper that parses flags and sends them to main function
    tf.app.run(main=train_and_test, argv=[sys.argv[0]] + unparsed)
