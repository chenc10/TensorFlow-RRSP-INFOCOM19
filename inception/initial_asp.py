# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=line-too-long
"""A binary to train Inception in a distributed manner using multiple systems.

Please see accompanying README.md for details and instructions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from datetime import datetime
import os.path
import time
import numpy as np
import threading2
import image_processing
from nets import nets_factory
from tensorflow.python.client import timeline

from imagenet_data import ImagenetData

if os.path.exists("timeline.json"):
  os.remove("timeline.json")

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('job_name', '', 'One of "ps", "worker"')
tf.app.flags.DEFINE_string('ps_hosts', '',
                           """Comma-separated list of hostname:port for the """
                           """parameter server jobs. e.g. """
                           """'machine1:2222,machine2:1111,machine2:2222'""")
tf.app.flags.DEFINE_string('worker_hosts', '',
                           """Comma-separated list of hostname:port for the """
                           """worker jobs. e.g. """
                           """'machine1:2222,machine2:1111,machine2:2222'""")
tf.app.flags.DEFINE_string('protocol', 'grpc',
                           """Communication protocol to use in distributed """
                           """execution (default grpc) """)

tf.app.flags.DEFINE_string('train_dir', '/home/ubuntu/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 5000, 'Number of batches to run.')
tf.app.flags.DEFINE_string('subset', 'train', 'Either "train" or "validation".')
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            'Whether to log device placement.')

# Task ID is used to select the chief and also to access the local_step for
# each replica to check staleness of the gradients in SyncReplicasOptimizer.
tf.app.flags.DEFINE_integer(
    'task_id', 0, 'Task ID of the worker/replica running the training.')

# More details can be found in the SyncReplicasOptimizer class:
# tensorflow/python/training/sync_replicas_optimizer.py
tf.app.flags.DEFINE_integer('num_replicas_to_aggregate', -1,
                            """Number of gradients to collect before """
                            """updating the parameters.""")
tf.app.flags.DEFINE_integer('save_interval_secs', 10 * 60,
                            'Save interval seconds.')
tf.app.flags.DEFINE_integer('save_summaries_secs', 180,
                            'Save summaries interval seconds.')

# **IMPORTANT**
# Please note that this learning rate schedule is heavily dependent on the
# hardware architecture, batch size and any changes to the model architecture
# specification. Selecting a finely tuned learning rate schedule is an
# empirical process that requires some experimentation. Please see README.md
# more guidance and discussion.
#
# Learning rate decay factor selected from https://arxiv.org/abs/1604.00981
tf.app.flags.DEFINE_float('initial_learning_rate', 0.045,
                          'Initial learning rate.')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 2.0,
                          'Epochs after which learning rate decays.')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.94,
                          'Learning rate decay factor.')

# Constants dictating the learning rate schedule.
RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.

_WEIGHT_DECAY = 2e-4
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.

#def get_computation_time(step_stats, gs):
#    tl = timeline.Timeline(step_stats)
#    [computation_time, communication_time, barrier_wait_time, processing_time] = tl.get_local_step_duration()
#    tf.logging.info('  gs: '+str(gs)+'; computation-phase1: '+str(computation_time) + '; communication-phase1: ' + str(communication_time))
#
#    [computation_time, communication_time, barrier_wait_time] = tl.get_local_step_duration('sync_token_q_Dequeue')
#    tf.logging.info('  gs: '+str(gs)+'; computation: '+str(computation_time) + '; communication: ' + str(communication_time) + '; barrier_wait: '+str(barrier_wait_time) + '; total processing time: '+ str(processing_time)+ '\n')
##    if gs == 10:
#	tf.logging.info('ccc-start-'+str(gs))
#    	ctf = tl.generate_chrome_trace_format()
#	tf.logging.info('ccc-finish-generate-'+str(gs))
#        with open('timeline-inception-10.json', 'w') as f:
#            f.write(ctf)
#        tf.logging.info('write json')



def train(target, dataset, cluster_spec):
  """Train Inception on a dataset for a number of steps."""
  # Number of workers and parameter servers are inferred from the workers and ps
  # hosts string.
  num_workers = len(cluster_spec.as_dict()['worker'])
  num_parameter_servers = len(cluster_spec.as_dict()['ps'])
  # If no value is given, num_replicas_to_aggregate defaults to be the number of
  # workers.
  if FLAGS.num_replicas_to_aggregate == -1:
    num_replicas_to_aggregate = num_workers
  else:
    num_replicas_to_aggregate = FLAGS.num_replicas_to_aggregate

  # Both should be greater than 0 in a distributed training.
  assert num_workers > 0 and num_parameter_servers > 0, (' num_workers and '
                                                         'num_parameter_servers'
                                                         ' must be > 0.')

  # Choose worker 0 as the chief. Note that any worker could be the chief
  # but there should be only one chief.
  is_chief = (FLAGS.task_id == 0)

  #batchSizeManager = BatchSizeManager(32, 4)

  # Ops are assigned to worker by default.
  tf.logging.info('cccc-num_parameter_servers:'+str(num_parameter_servers))
  partitioner = tf.fixed_size_partitioner(num_parameter_servers, 0)  

  device_setter = tf.train.replica_device_setter(ps_tasks=num_parameter_servers)
  slim = tf.contrib.slim
  with tf.device('/job:worker/task:%d' % FLAGS.task_id):
   with tf.variable_scope('root', partitioner=partitioner):
    # Variables and its related init/assign ops are assigned to ps.
#    with slim.arg_scope(
#        [slim.variables.variable, slim.variables.global_step],
#        device=slim.variables.VariableDeviceChooser(num_parameter_servers)):
    with tf.device(device_setter):
#	partitioner=partitioner):
      # Create a variable to count the number of train() calls. This equals the
      # number of updates applied to the variables.
#      global_step = slim.variables.global_step()
      global_step = tf.Variable(0, trainable=False)

      # Calculate the learning rate schedule.

      batch_size = tf.placeholder(dtype=tf.int32, shape=(), name='batch_size')
      num_batches_per_epoch = (dataset.num_examples_per_epoch() /
                               FLAGS.batch_size)
      # Decay steps need to be divided by the number of replicas to aggregate.
      decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay /
                        num_replicas_to_aggregate)

      # Decay the learning rate exponentially based on the number of steps.
      lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                      global_step,
                                      decay_steps,
                                      FLAGS.learning_rate_decay_factor,
                                      staircase=True)
      # Add a summary to track the learning rate.
#      tf.summary.scalar('learning_rate', lr)

      # Create an optimizer that performs gradient descent.

      images, labels = image_processing.distorted_inputs(
          dataset,
          batch_size,
          num_preprocess_threads=FLAGS.num_preprocess_threads)
      print(images.get_shape())
      print(labels.get_shape())

      # Number of classes in the Dataset label set plus 1.
      # Label 0 is reserved for an (unused) background class.
#      num_classes = dataset.num_classes() + 1
      num_classes = dataset.num_classes()
      print(num_classes)
#      logits = inception.inference(images, num_classes, for_training=True)
      network_fn = nets_factory.get_network_fn('inception_v3',num_classes=num_classes) 
      (logits,_) = network_fn(images)
      print(logits.get_shape())
      # Add classification loss.
#      inception.loss(logits, labels, batch_size)

      # Gather all of the losses including regularization losses.
      labels = tf.one_hot(labels, 1000, 1, 0)
      cross_entropy = tf.losses.softmax_cross_entropy(
          logits=logits, 
          onehot_labels=labels)
#      losses = tf.get_collection(slim.losses.LOSSES_COLLECTION)
#      losses += tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
      losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
      total_loss = cross_entropy + _WEIGHT_DECAY * tf.add_n(
          [tf.nn.l2_loss(v) for v in tf.trainable_variables()])

#      total_loss = tf.add_n(losses, name='total_loss')

      loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
      loss_averages_op = loss_averages.apply(losses + [total_loss])

      with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.RMSPropOptimizer(lr,
                                      RMSPROP_DECAY,
                                      momentum=RMSPROP_MOMENTUM,
                                      epsilon=RMSPROP_EPSILON)
        grads0 = opt.compute_gradients(total_loss) 
        grads = [(tf.scalar_mul(tf.cast(batch_size/FLAGS.batch_size, tf.float32), grad), var) for grad, var in grads0]
        total_loss = tf.identity(total_loss)

      exp_moving_averager = tf.train.ExponentialMovingAverage(
          MOVING_AVERAGE_DECAY, global_step)
      variables_averages_op = exp_moving_averager.apply(tf.trainable_variables())


      apply_gradients_op = opt.apply_gradients(grads, global_step=global_step)

      with tf.control_dependencies([apply_gradients_op, variables_averages_op]):
        train_op = tf.identity(total_loss, name='train_op')

      # Get chief queue_runners and init_tokens, which is used to synchronize
      # replicas. More details can be found in SyncReplicasOptimizer.
#      chief_queue_runners = [opt.get_chief_queue_runner()]
#      init_tokens_op = opt.get_init_tokens_op()

      # Create a saver.
      saver = tf.train.Saver()

      # Build the summary operation based on the TF collection of Summaries.
#      summary_op = tf.summary.merge_all()

      # Build an initialization operation to run below.
      init_op = tf.global_variables_initializer()

      # We run the summaries in the same thread as the training operations by
      # passing in None for summary_op to avoid a summary_thread being started.
      # Running summaries and training operations in parallel could run out of
      # GPU memory.
      sv = tf.train.Supervisor(is_chief=is_chief,
                               logdir=FLAGS.train_dir,
                               init_op=init_op,
                               summary_op=None,
                               global_step=global_step,
                               recovery_wait_secs=1,
                               saver=None,
                               save_model_secs=FLAGS.save_interval_secs)

      tf.logging.info('%s Supervisor' % datetime.now())

      sess_config = tf.ConfigProto(
          allow_soft_placement=True,
          log_device_placement=FLAGS.log_device_placement)

      # Get a session.
      sess = sv.prepare_or_wait_for_session(target, config=sess_config)

      # Start the queue runners.
      queue_runners = tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)
      sv.start_queue_runners(sess, queue_runners)
      tf.logging.info('Started %d queues for processing input data.',
                      len(queue_runners))

#      if is_chief:
#        sv.start_queue_runners(sess, chief_queue_runners)
#        sess.run(init_tokens_op)

      # Train, checking for Nans. Concurrently run the summary operation at a
      # specified interval. Note that the summary_op and train_op never run
      # simultaneously in order to prevent running out of GPU memory.
#      next_summary_time = time.time() + FLAGS.save_summaries_secs
      step = 0
      time0 = time.time()
      batch_size_num = 1
      while not sv.should_stop():
        try:
          start_time = time.time()

	  batch_size_num = 32
#	   batch_size_num = int((int(step)/3*10)) % 100000 + 1
#          if step < 5:
#            batch_size_num = 32 
#          batch_size_num = (batch_size_num ) % 64 + 1
#          else:
#            batch_size_num = 80

          run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
          run_metadata = tf.RunMetadata()

          my_images, loss_value, step = sess.run([images, train_op, global_step], feed_dict={batch_size: batch_size_num}, options=run_options, run_metadata=run_metadata)
	  b = time.time()
#          assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
          if step > FLAGS.max_steps:
            break
          duration = time.time() - start_time
#	  thread = threading2.Thread(target=get_computation_time, name="get_computation_time",args=(run_metadata.step_stats,step,))
#	  thread.start()
#          tl = timeline.Timeline(run_metadata.step_stats)
#          last_batch_time = tl.get_local_step_duration('sync_token_q_Dequeue')
          c0 = time.time()
#          batch_size_num = batchSizeManager.dictate_new_batch_size(FLAGS.task_id, last_batch_time)
#          batch_size_num = rpcClient.update_batch_size(FLAGS.task_id, last_batch_time, available_cpu, available_memory, step, batch_size_num) 
#          ctf = tl.generate_chrome_trace_format()
#          with open("timeline.json", 'a') as f:
#            f.write(ctf)

          if step % 1 == 0:
            examples_per_sec = FLAGS.batch_size / float(duration)
            c = time.time()
            tf.logging.info("time statistics" + " - train_time: " + str(b-start_time) + " - get_batch_time: " + str(c0-b) + " - get_bs_time:  " + str(c-c0) + " - accum_time: " + str(c-time0) + " - batch_size: " + str(batch_size_num))
            format_str = ('Worker %d: %s: step %d, loss = %.2f'
                          '(%.1f examples/sec; %.3f  sec/batch)')
            tf.logging.info(format_str %
                            (FLAGS.task_id, datetime.now(), step, loss_value,
                             examples_per_sec, duration))

          # Determine if the summary_op should be run on the chief worker.
#          if is_chief and next_summary_time < time.time():
#            tf.logging.info('Running Summary operation on the chief.')
#            summary_str = sess.run(summary_op)
#            sv.summary_computed(sess, summary_str)
#            tf.logging.info('Finished running Summary operation.')

            # Determine the next time for running the summary.
#            next_summary_time += FLAGS.save_summaries_secs
        except:
          if is_chief:
            tf.logging.info('Chief got exception while running!')
          raise

      # Stop the supervisor.  This also waits for service threads to finish.
      sv.stop()

      # Save after the training ends.
#      if is_chief:
#        saver.save(sess,
#                   os.path.join(FLAGS.train_dir, 'model.ckpt'),
#                   global_step=global_step)


def main(unused_args):
  assert FLAGS.job_name in ['ps', 'worker'], 'job_name must be ps or worker'

  # Extract all the hostnames for the ps and worker jobs to construct the
  # cluster spec.
  ps_hosts = FLAGS.ps_hosts.split(',')
  worker_hosts = FLAGS.worker_hosts.split(',')
  tf.logging.info('PS hosts are: %s' % ps_hosts)
  tf.logging.info('Worker hosts are: %s' % worker_hosts)

  cluster_spec = tf.train.ClusterSpec({'ps': ps_hosts,
                                       'worker': worker_hosts})
  server = tf.train.Server(
      {'ps': ps_hosts,
       'worker': worker_hosts},
      job_name=FLAGS.job_name,
      task_index=FLAGS.task_id,
      protocol=FLAGS.protocol)

  if FLAGS.job_name == 'ps':
    # `ps` jobs wait for incoming connections from the workers.
    server.join()
  else:
    # `worker` jobs will actually do the work.
    dataset = ImagenetData(subset=FLAGS.subset)
    assert dataset.data_files()
    # Only the chief checks for or creates train_dir.
    if FLAGS.task_id == 0:
      if not tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.MakeDirs(FLAGS.train_dir)
    train(server.target, dataset, cluster_spec)

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
