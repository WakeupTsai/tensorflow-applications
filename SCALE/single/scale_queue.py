import tensorflow as tf

import paho.mqtt.client as mqtt
import numpy as np
import json
import curses
import threading

BROKER = "[broker-address]"
TOPIC = "[subscribe-topic]"
QUEUE_SIZE = 10

mq5 = np.array([])
mq7 = np.array([])
mq131 = np.array([])
mq135 = np.array([])

mq5_average = 0
mq7_average = 0
mq131_average = 0
mq135_average = 0

sess = tf.Session()

'''placeholder'''
item = tf.placeholder(tf.float32, name='item')
size = tf.placeholder(tf.float32, name='size')

'''MQ5 OPERATORS'''
mq5_sum_variable = tf.Variable(0.0, name='mq5_sum_variable')

mq5_op_add = tf.add(item, mq5_sum_variable, name="mq5_op_add")
assign_mq5_add = tf.assign(mq5_sum_variable, mq5_op_add)
mq5_op_divide = tf.divide(mq5_sum_variable, size, name="mq5_op_divide")
mq5_sum_zero = tf.assign(mq5_sum_variable, 0.0)

mq5_value = tf.placeholder(tf.float32)
mq5Queue = tf.FIFOQueue(200, [tf.float32])
enqueue_mq5 = mq5Queue.enqueue([mq5_value])
dequeue_mq5 = mq5Queue.dequeue()

'''MQ7 OPERATORS'''
mq7_sum_variable = tf.Variable(0.0, name='mq7_sum_variable')

mq7_op_add = tf.add(item, mq7_sum_variable, name="mq7_op_add")
assign_mq7_add = tf.assign(mq7_sum_variable, mq7_op_add)
mq7_op_divide = tf.divide(mq7_sum_variable, size, name="mq7_op_divide")
mq7_sum_zero = tf.assign(mq7_sum_variable, 0.0)

mq7_value = tf.placeholder(tf.float32)
mq7Queue = tf.FIFOQueue(200, [tf.float32])
enqueue_mq7 = mq7Queue.enqueue([mq7_value])
dequeue_mq7 = mq7Queue.dequeue()

'''MQ131 OPERATORS'''
mq131_sum_variable = tf.Variable(0.0, name='mq131_sum_variable')

mq131_op_add = tf.add(item, mq131_sum_variable, name="mq131_op_add")
assign_mq131_add = tf.assign(mq131_sum_variable, mq131_op_add)
mq131_op_divide = tf.divide(mq131_sum_variable, size, name="mq131_op_divide")
mq131_sum_zero = tf.assign(mq131_sum_variable, 0.0)

mq131_value = tf.placeholder(tf.float32)
mq131Queue = tf.FIFOQueue(200, [tf.float32])
enqueue_mq131 = mq131Queue.enqueue([mq131_value])
dequeue_mq131 = mq131Queue.dequeue()

'''MQ135 OPERATORS'''
mq135_sum_variable = tf.Variable(0.0, name='mq135_sum_variable')

mq135_op_add = tf.add(item, mq135_sum_variable, name="mq135_op_add")
assign_mq135_add = tf.assign(mq135_sum_variable, mq135_op_add)
mq135_op_divide = tf.divide(mq135_sum_variable, size, name="mq135_op_divide")
mq135_sum_zero = tf.assign(mq135_sum_variable, 0.0)

mq135_value = tf.placeholder(tf.float32)
mq135Queue = tf.FIFOQueue(200, [tf.float32])
enqueue_mq135 = mq135Queue.enqueue([mq135_value])
dequeue_mq135 = mq135Queue.dequeue()

sess.run(tf.global_variables_initializer())

def mq5_result():
    global mq5
    global mq5_average
    while True:
        sess.run(mq5_sum_zero)
        dequeue = sess.run(dequeue_mq5)

        if mq5.size == QUEUE_SIZE:
            mq5 = mq5[1:]
            mq5 = np.append(mq5, dequeue)
        else:
            mq5 = np.append(mq5, dequeue)

        for index, mq5_item in enumerate(mq5):
            result = sess.run(assign_mq5_add, feed_dict={item:mq5_item})

        mq5_average = sess.run(mq5_op_divide, feed_dict={size:mq5.size})

        #result_print()


def mq7_result():
    global mq7
    global mq7_average
    while True:
        sess.run(mq7_sum_zero)
        dequeue = sess.run(dequeue_mq7)

        if mq7.size == QUEUE_SIZE:
            mq7 = mq7[1:]
            mq7 = np.append(mq7, dequeue)
        else:
            mq7 = np.append(mq7, dequeue)

        for index, mq7_item in enumerate(mq7):
            result = sess.run(assign_mq7_add, feed_dict={item:mq7_item})

        mq7_average = sess.run(mq7_op_divide, feed_dict={size:mq7.size})
        #result_print()


def mq131_result():
    global mq131
    global mq131_average
    while True:
        sess.run(mq131_sum_zero)
        dequeue = sess.run(dequeue_mq131)

        if mq131.size == QUEUE_SIZE:
            mq131 = mq131[1:]
            mq131 = np.append(mq131, dequeue)
        else:
            mq131 = np.append(mq131, dequeue)

        for index, mq131_item in enumerate(mq131):
            result = sess.run(assign_mq131_add, feed_dict={item:mq131_item})

        mq131_average = sess.run(mq131_op_divide, feed_dict={size:mq131.size})
        #result_print()

def mq135_result():
    global mq135
    global mq135_average
    while True:
        sess.run(mq135_sum_zero)
        dequeue = sess.run(dequeue_mq135)

        if mq135.size == QUEUE_SIZE:
            mq135 = mq135[1:]
            mq135 = np.append(mq135, dequeue)
        else:
            mq135 = np.append(mq135, dequeue)

        for index, mq135_item in enumerate(mq135):
            result = sess.run(assign_mq135_add, feed_dict={item:mq135_item})
        
        mq135_average = sess.run(mq135_op_divide, feed_dict={size:mq135.size})
        result_print()


def result_print():
    global mq5_average
    global mq7_average
    global mq131_average
    global mq135_average


    print 'MQ5:{:06.2f}, MQ7:{:06.2f}, MQ131:{:06.2f}, MQ135:{:06.2f}' \
            .format(mq5_average, mq7_average,mq131_average,mq135_average)



def json_parse(msg):
    data = json.loads(msg)
    event = data['d']['event']
    value = data['d']['value']
    return event, value

def on_connect(client, userdata, flags, rc):
    client.subscribe(TOPIC)


def on_message(client, userdata, msg):
    event, value = json_parse(msg.payload)

    if event == "pollution_air_mq5":
        sess.run(enqueue_mq5, feed_dict={mq5_value: value})
    elif event == "pollution_air_mq7":
        sess.run(enqueue_mq7, feed_dict={mq7_value: value})
    elif event == "pollution_air_mq131":
        sess.run(enqueue_mq131, feed_dict={mq131_value: value})
    elif event == "pollution_air_mq135":
        sess.run(enqueue_mq135, feed_dict={mq135_value: value})



if __name__ == "__main__":
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(BROKER, 1883, 60)

    client_thread =threading.Thread(target=client.loop_forever)
    mq5_thread = threading.Thread(target=mq5_result)
    mq7_thread = threading.Thread(target=mq7_result)
    mq131_thread = threading.Thread(target=mq131_result)
    mq135_thread = threading.Thread(target=mq135_result)

    client_thread.start()
    mq5_thread.start()
    mq7_thread.start()
    mq131_thread.start()
    mq135_thread.start()

    client_thread.join()
    mq5_thread.join()
    mq7_thread.join()
    mq131_thread.join()
    mq135_thread.join()


