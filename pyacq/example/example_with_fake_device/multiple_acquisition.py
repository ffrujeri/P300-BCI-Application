# -*- coding: utf-8 -*-
"""
Multiple acquisition with several device at different sampling_rate.
Use gevent for threading so need green version of zmq.

"""

from pyacq import StreamHandler, FakeMultiSignals
import msgpack
import gevent
import zmq.green as zmq




def test_recv_loop(port, stop):
    import zmq.green as zmq
    print 'start receiver loop' , port
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE,'')
    socket.connect("tcp://localhost:{}".format(port))
    while not stop.is_set():
        message = socket.recv()
        pos = msgpack.loads(message)
        print 'On port {} read pos is {}'.format( port, pos)
    print 'stop receiver'



def test1():
    streamhandler = StreamHandler()
    #~ timestampserver = timestampserver()
    n= 4 
    devices = [ ]
    sampling_rates = [10., 500., 1000., 10000.]
    packet_sizes =  [4, 32, 64, 128]
    for i in range(4):
        dev = FakeMultiSignals(streamhandler = streamhandler)
        sampling_rate = sampling_rates[i%4]
        packet_size = packet_sizes[i%4]
        dev.configure( 
                                    nb_channel = 3,
                                    sampling_rate =sampling_rate,
                                    buffer_length = 10.  * (sampling_rate//packet_size)/(sampling_rate/packet_size),
                                    packet_size = packet_size,
                                    )
        dev.initialize()
        devices.append(dev)
        #~ timestampserver.follow_stream(dev.stream)
        
    
    gevent.sleep(1)
    
    for i, dev in enumerate(devices):
        dev.start()
    gevent.sleep(1)
    
    stops = [ gevent.event.Event() for dev in devices ]
    greenlets = [ gevent.spawn(test_recv_loop, dev.streams[0]['port'], stop) for dev, stop in zip(devices, stops) ]
    gevent.sleep(5)
    for stop in stops:
        stop.set()
    
    gevent.sleep(1.)
    
    for i, dev in enumerate(devices):
        dev.stop()



if __name__ == '__main__':
    test1()
