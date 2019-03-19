from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np
#from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from nets import nets_factory

#from batchsizemanager import BatchSizeManager
import cifar10
from tensorflow.python.client import timeline

from ssp_manager import *

FLAGS = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)

INITIAL_LEARNING_RATE = 0.04       # Initial learning rate.
INITIAL_LEARNING_RATE = 0.00001       # Initial learning rate.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.

updated_batch_size_num = 28
_HEIGHT = 32
_WIDTH = 32
_DEPTH = 3
_NUM_CLASSES = 10
_NUM_DATA_FILES = 5
_WEIGHT_DECAY = 2e-4

def train():
    global updated_batch_size_num
    global passed_info
    global shall_update
    ps_hosts = FLAGS.ps_hosts.split(',')
    worker_hosts = FLAGS.worker_hosts.split(',')
    print ('PS hosts are: %s' % ps_hosts)
    print ('Worker hosts are: %s' % worker_hosts)

    server = tf.train.Server({'ps': ps_hosts, 'worker': worker_hosts},
                             job_name = FLAGS.job_name,
                             task_index=FLAGS.task_id)

    sspManager = SspManager(len(worker_hosts), 5)

    if FLAGS.job_name == 'ps':
    	if FLAGS.task_id == 0:
	    rpcServer = sspManager.create_rpc_server(ps_hosts[0].split(':')[0])
            rpcServer.serve()
        server.join()

    time.sleep(5)
    is_chief = (FLAGS.task_id == 0)
    rpcClient = sspManager.create_rpc_client(ps_hosts[0].split(':')[0])
    if is_chief:
        if tf.gfile.Exists(FLAGS.train_dir):
            tf.gfile.DeleteRecursively(FLAGS.train_dir)
        tf.gfile.MakeDirs(FLAGS.train_dir)

    device_setter = tf.train.replica_device_setter(ps_tasks=len(ps_hosts))
    with tf.device('/job:worker/task:%d' % FLAGS.task_id):
      partitioner=tf.fixed_size_partitioner(len(ps_hosts), axis=0)
      with tf.variable_scope('root', partitioner=partitioner):
        with tf.device(device_setter):
            global_step = tf.Variable(0, trainable=False)

	    decay_steps = 50000*350.0/FLAGS.batch_size
	    batch_size = tf.placeholder(dtype=tf.int32, shape=(), name='batch_size')
            images, labels = cifar10.distorted_inputs(batch_size)
#            print (str(tf.shape(images))+ str(tf.shape(labels)))
	    re = tf.shape(images)[0]
            inputs = tf.reshape(images, [-1, _HEIGHT, _WIDTH, _DEPTH])
#            labels = tf.reshape(labels, [-1, _NUM_CLASSES])
            labels = tf.one_hot(labels, 10, 1, 0)
  	    #network_fn = nets_factory.get_network_fn('inception_v3',num_classes=10) 
  	    network_fn = nets_factory.get_network_fn('vgg_16',num_classes=10) 
  	    (logits,_) = network_fn(inputs)
            print(logits.get_shape())
            cross_entropy = tf.losses.softmax_cross_entropy(
                logits=logits, 
                onehot_labels=labels)

#            logits = cifar10.inference(images, batch_size)

#            loss = cifar10.loss(logits, labels, batch_size)
            loss = cross_entropy + _WEIGHT_DECAY * tf.add_n(
                [tf.nn.l2_loss(v) for v in tf.trainable_variables()])

	    train_op = cifar10.train(loss, global_step)

            sv = tf.train.Supervisor(is_chief=is_chief,
                                     logdir=FLAGS.train_dir,
				     init_op=tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()),
                                     summary_op=None,
                                     global_step=global_step,
#                                     saver=saver,
                                     saver=None,
				     recovery_wait_secs=1,
                                     save_model_secs=60)

            tf.logging.info('%s Supervisor' % datetime.now())
   	    sess_config = tf.ConfigProto(allow_soft_placement=True,
   	                                 log_device_placement=FLAGS.log_device_placement)
	    sess_config.gpu_options.allow_growth = True

   	    # Get a session.
   	    sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)
#	    sess.run(tf.global_variables_initializer())

            # Start the queue runners.
            queue_runners = tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)
            sv.start_queue_runners(sess, queue_runners)

#            sv.start_queue_runners(sess, chief_queue_runners)
#            sess.run(init_tokens_op)

            """Train CIFAR-10 for a number of steps."""
#            available_cpu = psutil.cpu_percent(interval=None)

#            thread = threading2.Thread(target = local_update_batch_size, name = "update_batch_size_thread", args = (rpcClient, FLAGS.task_id,))
#            thread.start()

	    time0 = time.time()
	    batch_size_num = FLAGS.batch_size
	    loss_list = []
	    threshold = 0.95
            for step in range(FLAGS.max_steps):

                start_time = time.time()

      		run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      		run_metadata = tf.RunMetadata()

#                batch_size_num = updated_batch_size_num
		if step <= 5:
		    batch_size_num = FLAGS.batch_size
		if step >= 0:
		    batch_size_num = int(step/5)%512 + 1
		    batch_size_num = 128

                num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / batch_size_num
                decay_steps_num = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

#                mgrads, images_, train_val, real, loss_value, gs = sess.run([grads, images, train_op, re, loss, global_step], feed_dict={batch_size: batch_size_num},  options=run_options, run_metadata=run_metadata)
                _, loss_value, gs = sess.run([train_op, loss, global_step], feed_dict={batch_size: batch_size_num},  options=run_options, run_metadata=run_metadata)
#                _, loss_value, gs = sess.run([train_op, loss, global_step], feed_dict={batch_size: batch_size_num})
		b = time.time()
#    		tl = timeline.Timeline(run_metadata.step_stats)
##	        ctf = tl.generate_chrome_trace_format()
#		last_batch_time = tl.get_local_step_duration('sync_token_q_Dequeue')
#		thread = threading2.Thread(target=get_computation_time, name="get_computation_time",args=(run_metadata.step_stats,step,))
#		thread.start()

#                available_cpu = 100-psutil.cpu_percent(interval=None)
#                available_memory = psutil.virtual_memory()[1]/1000000
                c0 = time.time()

#	        batch_size_num = rpcClient.update_batch_size(FLAGS.task_id, last_batch_time, available_cpu, available_memory, step, batch_size_num)
#		if gs < 10:
#	            with open('timeline.json', 'w') as f:
#	                f.write(ctf)
#		    tf.logging.info('write json')


#	        batch_size_num = rpcClient.update_batch_size(FLAGS.task_id, 0,0,0, step, batch_size_num)

                if step % 1 == 0:
                    duration = time.time() - start_time
                    num_examples_per_step = batch_size_num
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

		    c = time.time()
##                    tf.logging.info("time statistics - batch_process_time: " + str( last_batch_time)  + " - train_time: " + str(b-start_time) + " - get_batch_time: " + str(c0-b) + " - get_bs_time:  " + str(c-c0) + " - accum_time: " + str(c-time0))

                    format_str = ("time: " + str(time.time()) + '; %s: step %d (global_step %d), loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
                    tf.logging.info(format_str % (datetime.now(), step, gs, loss_value, examples_per_sec, sec_per_batch))
##		    tf.logging.info("time: "+str(time.time()) + "; batch_size,"+str(batch_size_num)+"; last_batch_time," + str(last_batch_time) + '\n')
		    loss_list.append(loss_value)
		    if step > 10 and loss_list[-1] < loss_list[0]*0.95 and loss_list[-2] < loss_list[0]*0.95:
		    	tf.logging.info("Haha cc-terminate: step "+str(step))
			exit()
		    rpcClient.check_staleness(FLAGS.task_id, step)

def main(argv=None):
    cifar10.maybe_download_and_extract()
    train()

if __name__ == '__main__':
    tf.app.run()
