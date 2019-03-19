from example import UpdateBatchSize
from example import ttypes
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer
from thrift.server import TNonblockingServer

from example.UpdateBatchSize import Client


import time
import tensorflow as tf

import copy
import os

import threading

import numpy
import math

import sys

class BatchSizeHandler():
    def __init__(self, initial_batch_size, num_of_replicas, thread_num):
        self.num = num_of_replicas
        self.total_batch_size = initial_batch_size * num_of_replicas
	self.client_thread_num = thread_num
	self.iteration_start_times = [[] for i in range(self.num)]
	self.release_times = [[0] for i in range(self.num)]
	self.gap_time = 0.0
	self.latest_chief_start_time = sys.maxint
	self.max_average_iteration_time = 0
	self.block_times = [[] for i in range(self.num)]
	self.next_task_id = 0
	self.next_pull_time = 0
        tf.logging.info('--------cc create batchsizemanager')
	self.block_times = [[] for i in range(self.num)]
	self.sec_per_sample = [0.0023349,0.00158689,0.001095636,0.001095636]
#	self.sec_per_sample = [0.00158689,0.001095636,0.001095636]
#	self.batch_sizes = [512, 700, 1200, 1200]
	self.batch_sizes = [512, 512, 512]

    def update_batch_size(self, task_index, last_batch_time, avail_cpu, avail_memory, step, batch_size_num):	
	try:
	    tf.logging.info("worker-"+str(task_index)+"; step: " + str(step) + "; batch_size_num: "+str(batch_size_num)+"; report time: "+str(time.time())+"; outside_time:"+str(time.time() - self.release_times[task_index][-1]))
	    self.iteration_start_times[task_index].append(time.time())
	    if len(self.iteration_start_times[0]) > 10:
		shall_block = True
		while shall_block:
		    current_time = time.time()
		    if current_time > self.next_pull_time and task_index == self.next_task_id:
		    	shall_block = False
			self.next_pull_time = current_time + self.gap_time
			self.next_task_id = (task_index + 1)%self.num
		    	tf.logging.info("  worker-"+str(task_index)+" at time "+str(current_time) + "- set next_pull_time to "+str(self.next_pull_time) + " of task_id: "+str(self.next_task_id))
			self.block_times[task_index].append(current_time - self.iteration_start_times[task_index][-1])
			tf.logging.info(" blocked by:" + str(current_time - self.iteration_start_times[task_index][-1]))
		    else:
		        time.sleep(0.002)
		    	
		if task_index == 0:
#		    self.max_average_iteration_time = (self.iteration_start_times[0][-1]-self.iteration_start_times[0][5])/(len(self.iteration_start_times[0])-6)
		    self.max_average_iteration_time = (self.iteration_start_times[0][-1]-self.iteration_start_times[0][5]-sum(self.block_times[0][5:]))/(len(self.iteration_start_times[0])-6)
		    tf.logging.info(" --- max_average_iteration_time: " +str(self.max_average_iteration_time))
#		    tf.logging.info(str(self.iteration_start_times))
		    self.gap_time = self.max_average_iteration_time / (1.2*self.num)
		else:
		    tf.logging.info(" len of block_times: " + str(len(self.block_times[task_index])))
		    if len(self.block_times[task_index])%10 == 0:
		        average_block_time = sum(self.block_times[task_index][-5:])/5.0
			tf.logging.info(" average_block_time: " + str(average_block_time))
			tf.logging.info(" max_average_iteration_time: " + str(self.max_average_iteration_time))
#			if average_block_time > 0.1 * self.max_average_iteration_time:
#			    tf.logging.info('! - ! block time too long! ' + str(task_index) + ' - ' + str(average_block_time))
#			    batch_size_num += average_block_time / self.sec_per_sample[task_index] 
#			    batch_size_num = self.batch_sizes[task_index]
	    tf.logging.info("\n")

	except Exception as  e:  
	    print "error!"
            print e  
	current_time = time.time()
	self.release_times[task_index].append(current_time)
        return batch_size_num

class BatchSizeManager:
    def __init__(self, initial_batch_size, num_of_replicas):
	self.thread_num = num_of_replicas
	self.initial_batch_size = initial_batch_size
	self.num_of_replicas = num_of_replicas
    	# create batchsizehandler

    def create_rpc_server(self, ps_host_name):
	batchSizeHandler = BatchSizeHandler(self.initial_batch_size, self.num_of_replicas, self.thread_num)
	handler = batchSizeHandler
	processor = UpdateBatchSize.Processor(handler)
	#transport = TSocket.TServerSocket(ps_host_name, 8000)
	transport = TSocket.TServerSocket(host=ps_host_name, port=8000)
        tfactory = TTransport.TBufferedTransportFactory()
#        tfactory = TTransport.TFramedTransportFactory()
        pfactory = TBinaryProtocol.TBinaryProtocolFactory()
	#rpcServer = TServer.TSimpleServer(processor,transport, tfactory, pfactory)
	rpcServer = TNonblockingServer.TNonblockingServer(processor,transport, threads=self.thread_num)
	tf.logging.info("Listening on port 8000...")	
	return rpcServer

    def create_rpc_client(self, ps_host_name):
	tsocket = TSocket.TSocket(ps_host_name, 8000)
	#transport = TTransport.TBufferedTransport(tsocket)
	transport = TTransport.TFramedTransport(tsocket)
	protocol = TBinaryProtocol.TBinaryProtocol(transport)
	rpcClient = Client(protocol)
	transport.open()
        print ('------ create client: '+ ps_host_name)
	return rpcClient

