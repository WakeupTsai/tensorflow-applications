import tensorflow as tf

# Configuration of cluster 
ps_hosts = [ "127.0.0.1:8887" ]
worker_hosts = [ "127.0.0.1:8888", "127.0.0.1:8889" ]
cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

tf.app.flags.DEFINE_string("job_name", "worker", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

FLAGS = tf.app.flags.FLAGS

def main(_):
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)
    server.join()

if __name__ == "__main__":
    tf.app.run()
