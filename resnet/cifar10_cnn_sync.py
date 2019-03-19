from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np
#from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import resnet_model

#from batchsizemanager import BatchSizeManager
import cifar10
from tensorflow.python.client import timeline

FLAGS = tf.app.flags.FLAGS


tf.logging.set_verbosity(tf.logging.INFO)

INITIAL_LEARNING_RATE = 0.32       # Initial learning rate.
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

#    batchSizeManager = BatchSizeManager(FLAGS.batch_size, len(worker_hosts))

    if FLAGS.job_name == 'ps':
#	rpcServer = batchSizeManager.create_rpc_server(ps_hosts[0].split(':')[0])
#        rpcServer.serve()
        server.join()

#    rpcClient = batchSizeManager.create_rpc_client(ps_hosts[0].split(':')[0])
    is_chief = (FLAGS.task_id == 0)
    if is_chief:
        if tf.gfile.Exists(FLAGS.train_dir):
            tf.gfile.DeleteRecursively(FLAGS.train_dir)
        tf.gfile.MakeDirs(FLAGS.train_dir)

    device_setter = tf.train.replica_device_setter(ps_tasks=1)
    with tf.device('/job:worker/task:%d' % FLAGS.task_id):
        with tf.device(device_setter):
            global_step = tf.Variable(0, trainable=False)

	    decay_steps = 50000*350.0/FLAGS.batch_size
	    batch_size = tf.placeholder(dtype=tf.int32, shape=(), name='batch_size')
            images, labels = cifar10.distorted_inputs(batch_size)
            print('zx0')
            print(images.get_shape().as_list())
#            print (str(tf.shape(images))+ str(tf.shape(labels)))
	    re = tf.shape(images)[0]
#            network = resnet_model.cifar10_resnet_v2_generator(FLAGS.resnet_size, _NUM_CLASSES)
#            inputs = tf.reshape(images, [-1, _HEIGHT, _WIDTH, _DEPTH])
#            labels = tf.reshape(labels, [-1, _NUM_CLASSES])
	    logits = cifar10.inference(images, batch_size)
	    loss = cifar10.loss(logits, labels, batch_size)

            # Decay the learning rate exponentially based on the number of steps.
            lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                            global_step,
                                            decay_steps,
                                            LEARNING_RATE_DECAY_FACTOR,
                                            staircase=True)
            opt = tf.train.GradientDescentOptimizer(lr)

            # Track the moving averages of all trainable variables.
            exp_moving_averager = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
            variables_to_average = (tf.trainable_variables() + tf.moving_average_variables())

            opt = tf.train.SyncReplicasOptimizer(
                opt,
                replicas_to_aggregate=len(worker_hosts),
#                replica_id=FLAGS.task_id,
                total_num_replicas=len(worker_hosts),
                variable_averages=exp_moving_averager,
                variables_to_average=variables_to_average)

            # Compute gradients with respect to the loss.
#            grads0 = opt.compute_gradients(loss) 
#	    grads = list()
#	    for grad, var in grads0:
#		grads.append((tf.scalar_mul(tf.cast(batch_size/FLAGS.batch_size, tf.float32), grad), var))
            grads0 = opt.compute_gradients(loss) 
	    grads = [(tf.scalar_mul(tf.cast(batch_size/FLAGS.batch_size, tf.float32), grad), var) for grad, var in grads0]
	    #grads = tf.map_fn(lambda x : (tf.scalar_mul(tf.cast(batch_size/FLAGS.batch_size, tf.float32), x[0]), x[1]), grads0)
	    #grads = tf.while_loop(lambda x : x, grads0)

#            grads = opt.compute_gradients(loss) 

            apply_gradients_op = opt.apply_gradients(grads, global_step=global_step)

            with tf.control_dependencies([apply_gradients_op]):
                train_op = tf.identity(loss, name='train_op')

            chief_queue_runners = [opt.get_chief_queue_runner()]
            init_tokens_op = opt.get_init_tokens_op()

#            saver = tf.train.Saver()
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

   	    # Get a session.
   	    sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)
#	    sess.run(tf.global_variables_initializer())

            # Start the queue runners.
            queue_runners = tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)
            sv.start_queue_runners(sess, queue_runners)

            sv.start_queue_runners(sess, chief_queue_runners)
            sess.run(init_tokens_op)

            """Train CIFAR-10 for a number of steps."""
#            available_cpu = psutil.cpu_percent(interval=None)

#            thread = threading2.Thread(target = local_update_batch_size, name = "update_batch_size_thread", args = (rpcClient, FLAGS.task_id,))
#            thread.start()

	    time0 = time.time()
	    batch_size_num = FLAGS.batch_size
            for step in range(FLAGS.max_steps):

                start_time = time.time()

      		run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      		run_metadata = tf.RunMetadata()

#                batch_size_num = updated_batch_size_num
		if step <= 5:
		    batch_size_num = FLAGS.batch_size
		if step >= 0:
		    batch_size_num = int(step/6) % 2000 + 1
		    batch_size_num = FLAGS.batch_size

                num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / batch_size_num
                decay_steps_num = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

#                mgrads, images_, train_val, real, loss_value, gs = sess.run([grads, images, train_op, re, loss, global_step], feed_dict={batch_size: batch_size_num},  options=run_options, run_metadata=run_metadata)
                _, loss_value, gs = sess.run([train_op, loss, global_step], feed_dict={batch_size: batch_size_num},  options=run_options, run_metadata=run_metadata)
		b = time.time()
#	        print (mgrads)
    		tl = timeline.Timeline(run_metadata.step_stats)
	        ctf = tl.generate_chrome_trace_format()
		last_batch_time = tl.get_local_step_duration('sync_token_q_Dequeue')


#                available_cpu = 100-psutil.cpu_percent(interval=None)
#                available_memory = psutil.virtual_memory()[1]/1000000
                c0 = time.time()

#	        batch_size_num = rpcClient.update_batch_size(FLAGS.task_id, last_batch_time, available_cpu, available_memory, step, batch_size_num)
#		if gs < 10 and step%6==0 :
		if step==306:
	            with open('timeline.json', 'w') as f:
	                f.write(ctf)
		    tf.logging.info('write json')
		if step==307:
	            with open('timeline1.json', 'w') as f:
	                f.write(ctf)
		    tf.logging.info('write json')


                if step % 1 == 0:
                    duration = time.time() - start_time
                    num_examples_per_step = batch_size_num
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

		    c = time.time()
#                    tf.logging.info("time statistics - batch_process_time: " + str( last_batch_time)  + " - train_time: " + str(b-start_time) + " - get_batch_time: " + str(c0-b) + " - get_bs_time:  " + str(c-c0) + " - accum_time: " + str(c-time0))

                    format_str = ('%s: step %d (global_step %d), loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
                    tf.logging.info(format_str % (datetime.now(), step, gs, loss_value, examples_per_sec, sec_per_batch))
		    tf.logging.info("batch_size,"+str(batch_size_num)+",last_batch_time," + str(last_batch_time) + '\n')

def main(argv=None):
    cifar10.maybe_download_and_extract()
    train()

if __name__ == '__main__':
    tf.app.run()
