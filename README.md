# tensorflow-applications

Run TensorFlow in distributed system

'''
#execute ps0
python worker_test.py \
--ps_hosts=127.0.0.1:8887 \
--worker_hosts=127.0.0.1:8888,127.0.0.1:8889 \
--job_name=ps \
--task_index=0

#execute worker0
python worker_test.py \
--ps_hosts=127.0.0.1:8887 \
--worker_hosts=127.0.0.1:8888,127.0.0.1:8889 \
--job_name=worker \
--task_index=0

#execute worker1
python worker_test.py \
--ps_hosts=127.0.0.1:8887 \
--worker_hosts=127.0.0.1:8888,127.0.0.1:8889 \
--job_name=worker \
--task_index=1

#SCALE
python scale.py

#audio-classification
python audio_classify.py ../audio-min-samples/dogbark.wav

#yolo
python yolo.py dog.jpg
'''
