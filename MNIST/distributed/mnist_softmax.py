from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

from random import randint


FLAGS = "../MNIST_data/"


def main(_):
  # Import data
  mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

  sess = tf.InteractiveSession("grpc://127.0.0.1:8888")

  with tf.device("/job:ps/task:0"):
      # Create the model
      x = tf.placeholder(tf.float32, [None, 784])
      W = tf.Variable(tf.zeros([784, 10]))
      b = tf.Variable(tf.zeros([10]))
      y = tf.matmul(x, W) + b

      # Define loss and optimizer
      y_ = tf.placeholder(tf.float32, [None, 10])

      init = tf.global_variables_initializer()
      sess.run(init)


  with tf.device("/job:worker/task:0"):

      cross_entropy = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
      train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

      # Train
      for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  
  with tf.device("/job:worker/task:0"):
      # Test trained model
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                          y_: mnist.test.labels}))
      num = randint(0, mnist.test.images.shape[0])
      img = mnist.test.images[num]
      print (sess.run(tf.argmax(y, 1), feed_dict={x: [img]}))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='../MNIST_data/',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
