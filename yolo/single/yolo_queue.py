import sys

sys.path.append('./')

from yolo_tiny_net_queue import YoloTinyNet 
import tensorflow as tf 
import cv2
import numpy as np
import threading
import random
import time

images_name = ["aeroplane.jpg","bicycle.jpg","cat.jpg","dog.jpg","person.jpg","car.jpg","bus.jpg"]
classes_name =  ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]


finished_frame = 0
def process_conv(conv):
  p_classes = conv[0, :, :, 0:20]
  C = conv[0, :, :, 20:22]
  coordinate = conv[0, :, :, 22:]

  p_classes = np.reshape(p_classes, (7, 7, 1, 20))
  C = np.reshape(C, (7, 7, 2, 1))

  P = C * p_classes

  index = np.argmax(P)

  index = np.unravel_index(index, P.shape)

  class_num = index[3]

  coordinate = np.reshape(coordinate, (7, 7, 2, 4))

  max_coordinate = coordinate[index[0], index[1], index[2], :]

  xcenter = max_coordinate[0]
  ycenter = max_coordinate[1]
  w = max_coordinate[2]
  h = max_coordinate[3]

  xcenter = (index[1] + xcenter) * (448/7.0)
  ycenter = (index[0] + ycenter) * (448/7.0)

  w = w * 448
  h = h * 448

  xmin = xcenter - w/2.0
  ymin = ycenter - h/2.0

  xmax = xmin + w
  ymax = ymin + h

  return xmin, ymin, xmax, ymax, class_num

def read_image():
  while True:
    np_img = cv2.imread("images/" + random.choice (images_name))
    resized_img = cv2.resize(np_img, (448, 448))

    np_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    np_img = np_img.astype(np.float32)
    np_img = np_img / 255.0 * 2 - 1
    np_img = np.reshape(np_img, (1, 448, 448, 3))

    sess.run(enqueue_image, feed_dict={np_img_input: np_img,
                                    resized_img_input: resized_img})
    #time.sleep(2)


def build_model():
  while True: 
    np_img, resized_img = sess.run(dequeue_image) 

    np_predict = sess.run(conv, feed_dict={image: np_img})

    sess.run(enqueue_conv, feed_dict={conv_input: np_predict,
                                      resized_img_input2: resized_img})


def fully_connected_layer():
  global finished_frame
  while True:
    conv_output, resized_img = sess.run(dequeue_conv) 
    predict_result = sess.run (predict, feed_dict={temp_conv: conv_output,})


    xmin, ymin, xmax, ymax, class_num = process_conv(predict_result)
    class_name = classes_name[class_num]
    print ("Object Detection: " + classes_name[class_num])

    cv2.rectangle(resized_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255))
    cv2.putText(resized_img, class_name, (int(xmin), int(ymin)), 2, 1.5, (0, 0, 255))
    cv2.imwrite('out.jpg', resized_img)

    finished_frame = finished_frame +1
    #print (finished_frame)


if __name__ == "__main__":

  common_params = {'image_size': 448, 'num_classes': 20, 
                  'batch_size':1}
  net_params = {'cell_size': 7, 'boxes_per_cell':2, 'weight_decay': 0.0005}
  net = YoloTinyNet(common_params, net_params, test=True)

  image = tf.placeholder(tf.float32, (1, 448, 448, 3))
  conv = net.model(image)

  temp_conv = tf.placeholder(tf.float32)
  predict = net.connected_layer(temp_conv)

  # imagesQueue
  np_img_input = tf.placeholder(tf.float32, (1, 448, 448, 3))
  resized_img_input = tf.placeholder(tf.float32)
  imagesQueue = tf.FIFOQueue(200, [tf.float32, tf.float32])
  enqueue_image = imagesQueue.enqueue([np_img_input, resized_img_input])
  dequeue_image = imagesQueue.dequeue()

  # convQueue
  conv_input = tf.placeholder(tf.float32)
  resized_img_input2 = tf.placeholder(tf.float32)
  convQueue = tf.FIFOQueue(200, [tf.float32, tf.float32])
  enqueue_conv = convQueue.enqueue([conv_input,resized_img_input2])
  dequeue_conv = convQueue.dequeue()


  saver = tf.train.Saver(net.trainable_collection)

  with tf.Session() as sess:

    saver.restore(sess, 'models/pretrain/yolo_tiny.ckpt')

    a = threading.Thread(target=read_image)
    b = threading.Thread(target=build_model)
    c = threading.Thread(target=fully_connected_layer)
    
    a.start()
    b.start()
    c.start()

    a.join()
    b.join()
    c.join()
