import librosa
import numpy as np
np.set_printoptions(threshold=np.nan)

import tensorflow as tf
import sys
import random
import threading

sys.path.append("../audio-data-extraction")
import parse_audio as parse

# sound classification
sound_classification = {
  0:["air_conditioner","aircon.wav"],
  1:["car_horn","carhorn.wav"],
  2:["children_playiny","play.wav"],
  3:["dog_bark","dogbark.wav"],
  4:["drilling","drill.wav"],
  5:["enginge_idling","engine.wav"],
  6:["gun_shot","gunshots.wav"],
  7:["jackhammer","jackhammer.wav"],
  8:["siren","siren.wav"],
  9:["street_music","music.wav"]
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

######
with tf.device("/job:ps/task:0"):
    audio_input = tf.placeholder(tf.float32)
    audioQueue = tf.FIFOQueue(200, [tf.float32])
    enqueue_audio = audioQueue.enqueue([audio_input])

    def parse_audio():
      while True:
        a,b = random.choice(list(sound_classification.items()))
        audio, _ = parse.parse_audio_files("../audio-min-samples/" + b[1])
        sess.run(enqueue_audio, feed_dict={audio_input: audio})


######
with tf.device("/job:worker/task:0"):
    X = tf.placeholder(tf.float32, [None, n_dim])
    W_1 = tf.Variable(tf.random_normal([n_dim, n_hidden_units_one], mean=0, stddev=sd), name="w1")
    b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean=0, stddev=sd), name="b1")
    h_1 = tf.nn.sigmoid(tf.matmul(X, W_1) + b_1)

    dequeue_audio = audioQueue.dequeue()
    h1_input = tf.placeholder(tf.float32)
    h1Queue = tf.FIFOQueue(200, [tf.float32])
    enqueue_h1 = h1Queue.enqueue([h1_input])

    def h1():
      while True:
        dequeue = sess.run(dequeue_audio)
        h1_result = sess.run(h_1, feed_dict={X:dequeue})
        sess.run(enqueue_h1, feed_dict={h1_input: h1_result})


######
with tf.device("/job:worker/task:1"):
    h_1_input = tf.placeholder(tf.float32)
    W_2 = tf.Variable(tf.random_normal([n_hidden_units_one, n_hidden_units_two], mean=0, stddev=sd), name="w2")
    b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean=0, stddev=sd), name="b2")
    h_2 = tf.nn.tanh(tf.matmul(h_1_input, W_2) + b_2)

    dequeue_h1 = h1Queue.dequeue()
    h2_input = tf.placeholder(tf.float32)
    h2Queue = tf.FIFOQueue(200, [tf.float32])
    enqueue_h2 = h2Queue.enqueue([h2_input])

    def h2():
      while True:
        dequeue = sess.run(dequeue_h1)
        h2_result = sess.run(h_2, feed_dict={h_1_input:dequeue})
        sess.run(enqueue_h2, feed_dict={h2_input: h2_result})


######
with tf.device("/job:worker/task:2"):
    h_2_input =  tf.placeholder(tf.float32)
    W_3 = tf.Variable(tf.random_normal([n_hidden_units_two, n_hidden_units_three], mean=0, stddev=sd), name="w3")
    b_3 = tf.Variable(tf.random_normal([n_hidden_units_three], mean=0, stddev=sd), name="b3")
    h_3 = tf.nn.sigmoid(tf.matmul(h_2_input, W_3) + b_3)

    dequeue_h2 = h2Queue.dequeue()
    h3_input = tf.placeholder(tf.float32)
    h3Queue = tf.FIFOQueue(200, [tf.float32])
    enqueue_h3 = h3Queue.enqueue([h3_input])

    def h3():
      while True:
        dequeue = sess.run(dequeue_h2)
        h3_result = sess.run(h_3, feed_dict={h_2_input:dequeue})
        sess.run(enqueue_h3, feed_dict={h3_input: h3_result})


######
with tf.device("/job:worker/task:3"):
    h_3_input =  tf.placeholder(tf.float32)
    W = tf.Variable(tf.random_normal([n_hidden_units_three, n_classes], mean=0, stddev=sd), name="w")
    b = tf.Variable(tf.random_normal([n_classes], mean = 0, stddev=sd), name="b")
    y_ = tf.nn.softmax(tf.matmul(h_3_input, W) + b)
    argmax =  tf.argmax(y_,1)

    dequeue_h3 = h3Queue.dequeue()

    def result():
      while True:
        dequeue = sess.run(dequeue_h3)
        answer = sess.run(argmax,feed_dict={h_3_input: dequeue})
        print "Sound Classification: " + sound_classification[answer[0]][0]


if __name__ == "__main__":
  saver = tf.train.Saver()
  model_path = '../model-ckpt/pretrain.ckpt'

  with tf.Session("grpc://127.0.0.1:8888") as sess:
    load_path = saver.restore(sess, model_path)

    a = threading.Thread(target=parse_audio)
    b = threading.Thread(target=h1)
    c = threading.Thread(target=h2)
    d = threading.Thread(target=h3)
    e = threading.Thread(target=result)

    a.start()
    b.start()
    c.start()
    d.start()
    e.start()

    a.join()
    b.join()
    c.join()
    d.join()
    e.join()
