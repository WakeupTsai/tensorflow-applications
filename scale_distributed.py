import tensorflow as tf

import paho.mqtt.client as mqtt
import numpy as np
import json

BROKER = "[broker-address]"
TOPIC = "[subscribe-topic]"
QUEUE_SIZE = 10

mq5 = np.array([])
mq7 = np.array([])
mq131 = np.array([])
mq135 = np.array([])

sess = tf.Session("grpc://127.0.0.1:8888")

with tf.device("/job:ps/task:0"):
    mq5_variable = tf.Variable(0,dtype=tf.float64, name='mq5_variable')
    mq7_variable = tf.Variable(0, name='mq7_variable')
    mq131_variable = tf.Variable(0, name='mq131_variable')
    mq135_variable = tf.Variable(0, name='mq135_variable')

    sess.run(tf.global_variables_initializer())

    '''placeholder'''
    item = tf.placeholder(tf.float32, name='item')
    size = tf.placeholder(tf.float32, name='size')

'''MQ5 OPERATORS'''
with tf.device("/job:worker/task:0"):
    mq5_sum_variable = tf.Variable(0.0, name='mq5_sum_variable')
    mq5_average_variable =  tf.Variable(0.0, name='mq5_average_variable')

    mq5_op_add = tf.add(item, mq5_sum_variable, name="mq5_op_add")
    assign_mq5_add = tf.assign(mq5_sum_variable, mq5_op_add)

    mq5_op_divide = tf.divide(mq5_sum_variable, size, name="mq5_op_divide")
    assign_mq5_divide = tf.assign(mq5_average_variable, mq5_op_divide)

'''MQ7 OPERATORS'''
with tf.device("/job:worker/task:0"):
    mq7_sum_variable = tf.Variable(0.0, name='mq7_sum_variable')
    mq7_average_variable =  tf.Variable(0.0, name='mq7_average_variable')

    mq7_op_add = tf.add(item, mq7_sum_variable, name="mq7_op_add")
    assign_mq7_add = tf.assign(mq7_sum_variable, mq7_op_add)

    mq7_op_divide = tf.divide(mq7_sum_variable, size, name="mq7_op_divide")
    assign_mq7_divide = tf.assign(mq7_average_variable, mq7_op_divide)

'''MQ131 OPERATORS'''
with tf.device("/job:worker/task:1"):
    mq131_sum_variable = tf.Variable(0.0, name='mq131_sum_variable')
    mq131_average_variable =  tf.Variable(0.0, name='mq131_average_variable')

    mq131_op_add = tf.add(item, mq131_sum_variable, name="mq131_op_add")
    assign_mq131_add = tf.assign(mq131_sum_variable, mq131_op_add)

    mq131_op_divide = tf.divide(mq131_sum_variable, size, name="mq131_op_divide")
    assign_mq131_divide = tf.assign(mq131_average_variable, mq131_op_divide)

'''MQ135 OPERATORS'''
with tf.device("/job:worker/task:1"):
    mq135_sum_variable = tf.Variable(0.0, name='mq135_sum_variable')
    mq135_average_variable =  tf.Variable(0.0, name='mq135_average_variable')

    mq135_op_add = tf.add(item, mq135_sum_variable, name="mq135_op_add")
    assign_mq135_add = tf.assign(mq135_sum_variable, mq135_op_add)

    mq135_op_divide = tf.divide(mq135_sum_variable, size, name="mq135_op_divide")
    assign_mq135_divide = tf.assign(mq135_average_variable, mq135_op_divide)


def mq5_average(value):
    global mq5
    sess.run(tf.global_variables_initializer())

    if mq5.size == QUEUE_SIZE:
        mq5 = mq5[1:]
        mq5 = np.append(mq5, value)
    else:
        mq5 = np.append(mq5, value)

    for index, mq5_item in enumerate(mq5):
        result = sess.run(assign_mq5_add, feed_dict={item:value})

    print "Average MQ5: " + str(sess.run(assign_mq5_divide, feed_dict={size:mq5.size}))


def mq7_average(value):
    global mq7
    sess.run(tf.global_variables_initializer())

    if mq7.size == QUEUE_SIZE:
        mq7 = mq7[1:]
        mq7 = np.append(mq7, value)
    else:
        mq7 = np.append(mq7, value)

    for index, mq7_item in enumerate(mq7):
        result = sess.run(assign_mq7_add, feed_dict={item:value})

    print "Average MQ7: " + str(sess.run(assign_mq7_divide, feed_dict={size:mq7.size}))

def mq131_average(value):
    global mq131
    sess.run(tf.global_variables_initializer())

    if mq131.size == QUEUE_SIZE:
        mq131 = mq131[1:]
        mq131 = np.append(mq131, value)
    else:
        mq131 = np.append(mq131, value)

    for index, mq131_item in enumerate(mq131):
        result = sess.run(assign_mq131_add, feed_dict={item:value})

    print "Average MQ131: " + str(sess.run(assign_mq131_divide, feed_dict={size:mq131.size}))


def mq135_average(value):
    global mq135
    sess.run(tf.global_variables_initializer())

    if mq135.size == QUEUE_SIZE:
        mq135 = mq135[1:]
        mq135 = np.append(mq135, value)
    else:
        mq135 = np.append(mq135, value)

    for index, mq135_item in enumerate(mq135):
        result = sess.run(assign_mq135_add, feed_dict={item:value})

    print "Average MQ135: " + str(sess.run(assign_mq135_divide, feed_dict={size:mq135.size}))



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
        mq5_average(value)
    elif event == "pollution_air_mq7":
        mq7_average(value)
    elif event == "pollution_air_mq131":
        mq131_average(value)
    elif event == "pollution_air_mq135":
        mq135_average(value)



if __name__ == "__main__":
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(BROKER, 1883, 60)

    client.loop_forever()
