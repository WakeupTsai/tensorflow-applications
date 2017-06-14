import librosa
import numpy as np
np.set_printoptions(threshold=np.nan)

import tensorflow as tf
import sys

sys.path.append("../audio-data-extraction")
import parse_audio as parse

# sound classification
sound_classification = {
    0:"air_conditioner",
    1:"car_horn",
    2:"children_playiny",
    3:"dog_bark",
    4:"drilling",
    5:"enginge_idling",
    6:"gun_shot",
    7:"jackhammer",
    8:"siren",
    9:"street_music"
}

# Model Variables
training_epochs = 6000
n_dim = 193  # The model has 193 features extracted with librosa
n_classes = 10
n_hidden_units_one = 300
n_hidden_units_two = 200
n_hidden_units_three = 100
learning_rate = 0.1
sd = 1 / np.sqrt(n_dim)

# Tf's Variables
X = tf.placeholder(tf.float32, [None, n_dim])
Y = tf.placeholder(tf.float32, [None, n_classes])

W_1 = tf.Variable(tf.random_normal([n_dim, n_hidden_units_one], mean=0, stddev=sd), name="w1")
b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean=0, stddev=sd), name="b1")
h_1 = tf.nn.sigmoid(tf.matmul(X, W_1) + b_1)

W_2 = tf.Variable(tf.random_normal([n_hidden_units_one, n_hidden_units_two], mean=0, stddev=sd), name="w2")
b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean=0, stddev=sd), name="b2")
h_2 = tf.nn.tanh(tf.matmul(h_1, W_2) + b_2)

W_3 = tf.Variable(tf.random_normal([n_hidden_units_two, n_hidden_units_three], mean=0, stddev=sd), name="w3")
b_3 = tf.Variable(tf.random_normal([n_hidden_units_three], mean=0, stddev=sd), name="b3")
h_3 = tf.nn.sigmoid(tf.matmul(h_2, W_3) + b_3)

W = tf.Variable(tf.random_normal([n_hidden_units_three, n_classes], mean=0, stddev=sd), name="w")
b = tf.Variable(tf.random_normal([n_classes], mean = 0, stddev=sd), name="b")

# The proposed model
y_ = tf.nn.softmax(tf.matmul(h_3, W) + b)

saver = tf.train.Saver()
model_path = '../model-ckpt/pretrain.ckpt'

with tf.Session() as sess:
    load_path = saver.restore(sess, model_path)
    X_input, y_input = parse.parse_audio_files(sys.argv[1])

    answer = sess.run(tf.argmax(y_, 1), feed_dict={X: X_input})
    print "Sound Classification: " + sound_classification[answer[0]]
