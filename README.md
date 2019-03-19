# chen-models
TensorFlow code for distributed training process of ResNet, AlexNet, VGG, Inception under different synchronization schemes (BSP, ASP, SSP, RRSP)

For detail introduction of RRSP, please refer to the INFOCOM'19 paper "Round-Robin Synchronization: Mitigating Communication Bottlenecks in Parameter Servers".

1. What's this repository?

  It contains some unofficial codes for distributed training process of ResNet, AlexNet, VGG and Inception.
  
2. How to run the codes?

  (1) Where to place the folder?
  
    Under "/home/ubuntu".
    
  (2) Environmental Setup:
  
    export ps='ps-0-ip ps-1-ip'
    
    export slaves='slave-0-ip slave-1-ip slave-2-ip'

  (3) Command to launch a distributed training process
  
    "bash run_<model_name>_<scheme_name>.sh"
    
  (4) Where to find the logs?
  
    Remote worker-<i> would send the stdout log to "/home/ubuntu/slave<i>.log".
    
    PS-<i> would send the stdout log to "/home/ubuntu/master<i>.log".
    
   (5) Some auxillary scripts.
   
    "bash clear.sh" to stop the distributed training process, by killing the python processes in each worker.
    
    "bash sync.sh <file_name>" to synchronize the <file_name> to all workers.

*The core code for describing the Round-Robin Synchronization mechanism is in the file 'batchsizemanager.py'.
