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


from batchsizemanager import *

FLAGS = tf.app.flags.FLAGS


tf.logging.set_verbosity(tf.logging.INFO)

INITIAL_LEARNING_RATE = 0.01       # Initial learning rate.
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
    ps_hosts = FLAGS.ps_hosts.split(',')
    worker_hosts = FLAGS.worker_hosts.split(',')
    print ('PS hosts are: %s' % ps_hosts)
    print ('Worker hosts are: %s' % worker_hosts)
    configP=tf.ConfigProto()

    server = tf.train.Server({'ps': ps_hosts, 'worker': worker_hosts},
                             job_name = FLAGS.job_name,
                             task_index=FLAGS.task_id,
			     config=configP)

    batchSizeManager = BatchSizeManager(FLAGS.batch_size, len(worker_hosts))

    if FLAGS.job_name == 'ps':
        server.join()

    is_chief = (FLAGS.task_id == 0)
    if is_chief:
        if tf.gfile.Exists(FLAGS.train_dir):
            tf.gfile.DeleteRecursively(FLAGS.train_dir)
        tf.gfile.MakeDirs(FLAGS.train_dir)

    device_setter = tf.train.replica_device_setter(ps_tasks=len(ps_hosts))
    with tf.device('/job:worker/task:%d' % FLAGS.task_id):
        with tf.device(device_setter):
            global_step = tf.Variable(0, trainable=False)

	    decay_steps = 50000*350.0/FLAGS.batch_size
	    batch_size = tf.placeholder(dtype=tf.int32, shape=(), name='batch_size')
            images, labels = cifar10.distorted_inputs(batch_size)
	    re = tf.shape(images)[0]
	    with tf.variable_scope('root', partitioner=tf.fixed_size_partitioner(len(ps_hosts), axis=0)):
            	network = resnet_model.cifar10_resnet_v2_generator(FLAGS.resnet_size, _NUM_CLASSES)
            inputs = tf.reshape(images, [-1, _HEIGHT, _WIDTH, _DEPTH])
#            labels = tf.reshape(labels, [-1, _NUM_CLASSES])
            print(labels.get_shape())
            labels = tf.one_hot(labels, 10, 1, 0)
            print(labels.get_shape())
            logits = network(inputs, True)
            print(logits.get_shape())
            cross_entropy = tf.losses.softmax_cross_entropy(
                logits=logits, 
                onehot_labels=labels)

            loss = cross_entropy + _WEIGHT_DECAY * tf.add_n(
                [tf.nn.l2_loss(v) for v in tf.trainable_variables()])

	    train_op = cifar10.train(loss, global_step)

	    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
	    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#            lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
#                                            global_step,
#                                            decay_steps,
#                                            LEARNING_RATE_DECAY_FACTOR,
#                                            staircase=True)
#            opt = tf.train.GradientDescentOptimizer(lr)
#
#            exp_moving_averager = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
#            variables_to_average = (tf.trainable_variables() + tf.moving_average_variables())
#
#            opt = tf.train.SyncReplicasOptimizer(
#                opt,
#                replicas_to_aggregate=len(worker_hosts),
#                total_num_replicas=len(worker_hosts),
#                variable_averages=exp_moving_averager,
#                variables_to_average=variables_to_average)
#
#            grads0 = opt.compute_gradients(loss) 
#	    grads = [(tf.scalar_mul(tf.cast(batch_size/FLAGS.batch_size, tf.float32), grad), var) for grad, var in grads0]
#
#            apply_gradients_op = opt.apply_gradients(grads, global_step=global_step)
#
#            with tf.control_dependencies([apply_gradients_op]):
#                train_op = tf.identity(loss, name='train_op')

#            chief_queue_runners = [opt.get_chief_queue_runner()]
#            init_tokens_op = opt.get_init_tokens_op()

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
					intra_op_parallelism_threads=1,
					inter_op_parallelism_threads=1,
   	                                 log_device_placement=FLAGS.log_device_placement)
	    sess_config.gpu_options.allow_growth = True

   	    # Get a session.
   	    sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)
#	    sess.run(tf.global_variables_initializer())

            # Start the queue runners.
            queue_runners = tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)
            sv.start_queue_runners(sess, queue_runners)
#
#            sv.start_queue_runners(sess, chief_queue_runners)
#            sess.run(init_tokens_op)

            """Train CIFAR-10 for a number of steps."""

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
		    batch_size_num = int(step/5)%512 + 1
		    batch_size_num = 128

                num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / batch_size_num
                decay_steps_num = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

#                _, loss_value, gs, accuracy_score = sess.run([train_op, loss, global_step, accuracy], feed_dict={batch_size: batch_size_num},  options=run_options, run_metadata=run_metadata)
                _, loss_value, gs = sess.run([train_op, loss, global_step], feed_dict={batch_size: batch_size_num},  options=run_options, run_metadata=run_metadata)
#                _, loss_value, gs = sess.run([train_op, loss, global_step], feed_dict={batch_size: batch_size_num})
		b = time.time()
#    		tl = timeline.Timeline(run_metadata.step_stats)
#		last_batch_time = tl.get_local_step_duration('sync_token_q_Dequeue')
#		thread = threading2.Thread(target=get_computation_time, name="get_computation_time",args=(run_metadata.step_stats,step,))
#		thread.start()

                c0 = time.time()

#	        batch_size_num = rpcClient.update_batch_size(FLAGS.task_id, last_batch_time, available_cpu, available_memory, step, batch_size_num)
#	        batch_size_num = rpcClient.update_batch_size(FLAGS.task_id, 0,0,0, step, batch_size_num)



                if step % 1 == 0:
                    duration = time.time() - start_time
                    num_examples_per_step = batch_size_num
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

		    c = time.time()
##                    tf.logging.info("time statistics - batch_process_time: " + str( last_batch_time)  + " - train_time: " + str(b-start_time) + " - get_batch_time: " + str(c0-b) + " - get_bs_time:  " + str(c-c0) + " - accum_time: " + str(c-time0))

                    format_str = ("time: " + str(time.time()) + '; %s: step %d (global_step %d), loss = %.2f (%.1f examples/sec; %.3f sec/batch) - batch_size: '+str(batch_size_num))
                    tf.logging.info(format_str % (datetime.now(), step, gs, loss_value, examples_per_sec, sec_per_batch))
#		    tf.logging.info(" - accuracy: " + str(accuracy_score))
##		    tf.logging.info("time: "+str(time.time()) + "; batch_size,"+str(batch_size_num)+"; last_batch_time," + str(last_batch_time) + '\n')

def main(argv=None):
    print("cccc"+str(tf.__version__))
    cifar10.maybe_download_and_extract()
    train()

if __name__ == '__main__':
    tf.app.run()
