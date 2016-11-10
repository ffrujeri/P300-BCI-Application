# -*- coding: utf-8 -*-
"""



"""

from pyacq import StreamHandler, BrainvisionSocket
from pyacq.gui import Oscilloscope, TimeFreq
from pyacq.core.devices.brainvisionsocket import dtype_trigger

import msgpack
#~ import gevent
#~ import zmq.green as zmq
import multiprocessing as mp

from PyQt4 import QtCore,QtGui

import zmq
import msgpack
import time

import numpy as np


def test1():
    streamhandler = StreamHandler()
    
    
    # Configure and start
    dev = BrainvisionSocket(streamhandler = streamhandler)
    dev.configure(buffer_length = 64,
                brain_host = 'localhost',
                brain_port = 51244,
                )
    
    dev.initialize()
    print dev.streams[0]
    print dev.streams[0]['port']

    dev.start()

    app = QtGui.QApplication([])
    
    w1=Oscilloscope(stream = dev.streams[0])
    w1.show()
    #~ w1.auto_gain_and_offset(mode = 2)
    #~ visibles = np.ones(16, dtype = bool)
    #~ visibles[4:] = False
    #~ w1.set_params(xsize = 1.,
                                    #~ mode = 'scan',
                                #~ visibles = visibles)

    #~ print w1.get_params()
    #~ w1.set_params(**w1.get_params())

    #~ w2=Oscilloscope(stream = dev.streams[0])
    #~ w2.show()
    #~ w2.auto_gain_and_offset(mode = 0)
    #~ w2.set_params(xsize = 5, mode = 'scroll')
    
    #~ w3=TimeFreq(stream = dev.streams[0])
    #~ w3.show()
    #~ w1.change_param_tfr(colormap = 'bone')
    #~ visibles = np.zeros(64, dtype = bool)
    #~ visibles[::3] = True
    
    #~ w3.set_params(colormap = 'hot',  xsize=10., nb_column = 2)

    
    app.exec_()


    
    # Stope and release the device
    dev.stop()
    dev.close()







def test2():
    streamhandler = StreamHandler()
    
    
    # Configure and start
    dev = BrainvisionSocket(streamhandler = streamhandler)
    dev.configure(buffer_length = 64,
                brain_host = 'localhost',
                brain_port = 51244,
                )
    
    dev.initialize()
    print dev.streams[0]
    print dev.streams[0]['port']
    print dev.streams[1]
    print dev.streams[1]['port']
    dev.start()

    # Create recv trigger loop
    stream1 = dev.streams[1]

    print 'start receiver loop' , stream1['port']
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE,'')
    socket.connect("tcp://localhost:{}".format(stream1['port']))
    while True:
        buf = socket.recv()
        triggers = np.frombuffer(buf, dtype = dtype_trigger)
        print 'triggers n=', triggers.size
        print triggers
        
    print 'stop receiver'

    time.sleep(10.)

    
    # Stope and release the device
    dev.stop()
    dev.close()


if __name__ == '__main__':
    #~ test1()
    test2()
    
