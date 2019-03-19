#
# Autogenerated by Thrift Compiler (0.9.1)
#
# DO NOT EDIT UNLESS YOU ARE SURE THAT YOU KNOW WHAT YOU ARE DOING
#
#  options string: py
#

from thrift.Thrift import TType, TMessageType, TException, TApplicationException
from ttypes import *
from thrift.Thrift import TProcessor
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol, TProtocol
try:
  from thrift.protocol import fastbinary
except:
  fastbinary = None


class Iface:
  def update_batch_size(self, task_index, last_batch_time, avail_cpu, avail_memory, step, batch_size):
    """
    Parameters:
     - task_index
     - last_batch_time
     - avail_cpu
     - avail_memory
     - step
     - batch_size
    """
    pass


class Client(Iface):
  def __init__(self, iprot, oprot=None):
    self._iprot = self._oprot = iprot
    if oprot is not None:
      self._oprot = oprot
    self._seqid = 0

  def update_batch_size(self, task_index, last_batch_time, avail_cpu, avail_memory, step, batch_size):
    """
    Parameters:
     - task_index
     - last_batch_time
     - avail_cpu
     - avail_memory
     - step
     - batch_size
    """
    self.send_update_batch_size(task_index, last_batch_time, avail_cpu, avail_memory, step, batch_size)
    return self.recv_update_batch_size()

  def send_update_batch_size(self, task_index, last_batch_time, avail_cpu, avail_memory, step, batch_size):
    self._oprot.writeMessageBegin('update_batch_size', TMessageType.CALL, self._seqid)
    args = update_batch_size_args()
    args.task_index = task_index
    args.last_batch_time = last_batch_time
    args.avail_cpu = avail_cpu
    args.avail_memory = avail_memory
    args.step = step
    args.batch_size = batch_size
    args.write(self._oprot)
    self._oprot.writeMessageEnd()
    self._oprot.trans.flush()

  def recv_update_batch_size(self):
    (fname, mtype, rseqid) = self._iprot.readMessageBegin()
    if mtype == TMessageType.EXCEPTION:
      x = TApplicationException()
      x.read(self._iprot)
      self._iprot.readMessageEnd()
      raise x
    result = update_batch_size_result()
    result.read(self._iprot)
    self._iprot.readMessageEnd()
    if result.success is not None:
      return result.success
    raise TApplicationException(TApplicationException.MISSING_RESULT, "update_batch_size failed: unknown result");


class Processor(Iface, TProcessor):
  def __init__(self, handler):
    self._handler = handler
    self._processMap = {}
    self._processMap["update_batch_size"] = Processor.process_update_batch_size

  def process(self, iprot, oprot):
    (name, type, seqid) = iprot.readMessageBegin()
    if name not in self._processMap:
      iprot.skip(TType.STRUCT)
      iprot.readMessageEnd()
      x = TApplicationException(TApplicationException.UNKNOWN_METHOD, 'Unknown function %s' % (name))
      oprot.writeMessageBegin(name, TMessageType.EXCEPTION, seqid)
      x.write(oprot)
      oprot.writeMessageEnd()
      oprot.trans.flush()
      return
    else:
      self._processMap[name](self, seqid, iprot, oprot)
    return True

  def process_update_batch_size(self, seqid, iprot, oprot):
    args = update_batch_size_args()
    args.read(iprot)
    iprot.readMessageEnd()
    result = update_batch_size_result()
    result.success = self._handler.update_batch_size(args.task_index, args.last_batch_time, args.avail_cpu, args.avail_memory, args.step, args.batch_size)
    oprot.writeMessageBegin("update_batch_size", TMessageType.REPLY, seqid)
    result.write(oprot)
    oprot.writeMessageEnd()
    oprot.trans.flush()


# HELPER FUNCTIONS AND STRUCTURES

class update_batch_size_args:
  """
  Attributes:
   - task_index
   - last_batch_time
   - avail_cpu
   - avail_memory
   - step
   - batch_size
  """

  thrift_spec = (
    None, # 0
    (1, TType.I32, 'task_index', None, None, ), # 1
    (2, TType.I32, 'last_batch_time', None, None, ), # 2
    (3, TType.DOUBLE, 'avail_cpu', None, None, ), # 3
    (4, TType.DOUBLE, 'avail_memory', None, None, ), # 4
    (5, TType.I32, 'step', None, None, ), # 5
    (6, TType.I32, 'batch_size', None, None, ), # 6
  )

  def __init__(self, task_index=None, last_batch_time=None, avail_cpu=None, avail_memory=None, step=None, batch_size=None,):
    self.task_index = task_index
    self.last_batch_time = last_batch_time
    self.avail_cpu = avail_cpu
    self.avail_memory = avail_memory
    self.step = step
    self.batch_size = batch_size

  def read(self, iprot):
    if iprot.__class__ == TBinaryProtocol.TBinaryProtocolAccelerated and isinstance(iprot.trans, TTransport.CReadableTransport) and self.thrift_spec is not None and fastbinary is not None:
      fastbinary.decode_binary(self, iprot.trans, (self.__class__, self.thrift_spec))
      return
    iprot.readStructBegin()
    while True:
      (fname, ftype, fid) = iprot.readFieldBegin()
      if ftype == TType.STOP:
        break
      if fid == 1:
        if ftype == TType.I32:
          self.task_index = iprot.readI32();
        else:
          iprot.skip(ftype)
      elif fid == 2:
        if ftype == TType.I32:
          self.last_batch_time = iprot.readI32();
        else:
          iprot.skip(ftype)
      elif fid == 3:
        if ftype == TType.DOUBLE:
          self.avail_cpu = iprot.readDouble();
        else:
          iprot.skip(ftype)
      elif fid == 4:
        if ftype == TType.DOUBLE:
          self.avail_memory = iprot.readDouble();
        else:
          iprot.skip(ftype)
      elif fid == 5:
        if ftype == TType.I32:
          self.step = iprot.readI32();
        else:
          iprot.skip(ftype)
      elif fid == 6:
        if ftype == TType.I32:
          self.batch_size = iprot.readI32();
        else:
          iprot.skip(ftype)
      else:
        iprot.skip(ftype)
      iprot.readFieldEnd()
    iprot.readStructEnd()

  def write(self, oprot):
    if oprot.__class__ == TBinaryProtocol.TBinaryProtocolAccelerated and self.thrift_spec is not None and fastbinary is not None:
      oprot.trans.write(fastbinary.encode_binary(self, (self.__class__, self.thrift_spec)))
      return
    oprot.writeStructBegin('update_batch_size_args')
    if self.task_index is not None:
      oprot.writeFieldBegin('task_index', TType.I32, 1)
      oprot.writeI32(self.task_index)
      oprot.writeFieldEnd()
    if self.last_batch_time is not None:
      oprot.writeFieldBegin('last_batch_time', TType.I32, 2)
      oprot.writeI32(self.last_batch_time)
      oprot.writeFieldEnd()
    if self.avail_cpu is not None:
      oprot.writeFieldBegin('avail_cpu', TType.DOUBLE, 3)
      oprot.writeDouble(self.avail_cpu)
      oprot.writeFieldEnd()
    if self.avail_memory is not None:
      oprot.writeFieldBegin('avail_memory', TType.DOUBLE, 4)
      oprot.writeDouble(self.avail_memory)
      oprot.writeFieldEnd()
    if self.step is not None:
      oprot.writeFieldBegin('step', TType.I32, 5)
      oprot.writeI32(self.step)
      oprot.writeFieldEnd()
    if self.batch_size is not None:
      oprot.writeFieldBegin('batch_size', TType.I32, 6)
      oprot.writeI32(self.batch_size)
      oprot.writeFieldEnd()
    oprot.writeFieldStop()
    oprot.writeStructEnd()

  def validate(self):
    return


  def __repr__(self):
    L = ['%s=%r' % (key, value)
      for key, value in self.__dict__.iteritems()]
    return '%s(%s)' % (self.__class__.__name__, ', '.join(L))

  def __eq__(self, other):
    return isinstance(other, self.__class__) and self.__dict__ == other.__dict__

  def __ne__(self, other):
    return not (self == other)

class update_batch_size_result:
  """
  Attributes:
   - success
  """

  thrift_spec = (
    (0, TType.I32, 'success', None, None, ), # 0
  )

  def __init__(self, success=None,):
    self.success = success

  def read(self, iprot):
    if iprot.__class__ == TBinaryProtocol.TBinaryProtocolAccelerated and isinstance(iprot.trans, TTransport.CReadableTransport) and self.thrift_spec is not None and fastbinary is not None:
      fastbinary.decode_binary(self, iprot.trans, (self.__class__, self.thrift_spec))
      return
    iprot.readStructBegin()
    while True:
      (fname, ftype, fid) = iprot.readFieldBegin()
      if ftype == TType.STOP:
        break
      if fid == 0:
        if ftype == TType.I32:
          self.success = iprot.readI32();
        else:
          iprot.skip(ftype)
      else:
        iprot.skip(ftype)
      iprot.readFieldEnd()
    iprot.readStructEnd()

  def write(self, oprot):
    if oprot.__class__ == TBinaryProtocol.TBinaryProtocolAccelerated and self.thrift_spec is not None and fastbinary is not None:
      oprot.trans.write(fastbinary.encode_binary(self, (self.__class__, self.thrift_spec)))
      return
    oprot.writeStructBegin('update_batch_size_result')
    if self.success is not None:
      oprot.writeFieldBegin('success', TType.I32, 0)
      oprot.writeI32(self.success)
      oprot.writeFieldEnd()
    oprot.writeFieldStop()
    oprot.writeStructEnd()

  def validate(self):
    return


  def __repr__(self):
    L = ['%s=%r' % (key, value)
      for key, value in self.__dict__.iteritems()]
    return '%s(%s)' % (self.__class__.__name__, ', '.join(L))

  def __eq__(self, other):
    return isinstance(other, self.__class__) and self.__dict__ == other.__dict__

  def __ne__(self, other):
    return not (self == other)