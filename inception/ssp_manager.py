from ssp import CheckStaleness
from ssp import ttypes
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer
from thrift.server import TNonblockingServer

from ssp.CheckStaleness import Client

import time
import tensorflow as tf

class SspHandler():
    def __init__(self, num_of_replicas, max_stale):
        self.num = num_of_replicas
	self.max_stale = max_stale
	self.local_step_list = [0]*self.num
	self.min_step = 0

    def check_staleness(self, task_index, local_step):
	result = 0
	self.local_step_list[task_index] = local_step	
	self.min_step = min(self.local_step_list)
	tf.logging.info(" max_stale:" +str(self.max_stale))
	tf.logging.info(" check staleness from worker: "+str(task_index) + " at step " + str(local_step) +" (current min step:" + str(self.min_step)+")")
	while local_step > self.min_step + self.max_stale:
		# shall wait
		tf.logging.info('------ Hey! You are too fast! Wait! Current task: ' + str(task_index) + '; Current step: ' + str(local_step) + "; Current min_step: " + str(min(self.local_step_list)))
		time.sleep(0.1)
	return 1

class SspManager():
    def __init__(self, num_of_replicas, max_stale):
        tf.logging.info('--------cc create sspManager')
        self.num = num_of_replicas
	self.thread_num = num_of_replicas
	self.max_stale = max_stale

    def create_rpc_server(self, ps_host_name):
	sspHandler = SspHandler(self.num, self.max_stale)
	handler = sspHandler
	processor = CheckStaleness.Processor(handler)

	transport = TSocket.TServerSocket(host=ps_host_name, port=8000)
        tfactory = TTransport.TBufferedTransportFactory()

	rpcServer = TNonblockingServer.TNonblockingServer(processor,transport, threads=self.thread_num)
	tf.logging.info("Listening on port 8000...")

	return rpcServer

    def create_rpc_client(self, ps_host_name):
	tsocket = TSocket.TSocket(ps_host_name, 8000)
	transport = TTransport.TFramedTransport(tsocket)
	protocol = TBinaryProtocol.TBinaryProtocol(transport)
	rpcClient = Client(protocol)
	transport.open()
        print ('------ create client: '+ ps_host_name)
	return rpcClient
