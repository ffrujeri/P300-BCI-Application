# -*- coding: utf-8 -*-
"""
Oscilloscope example.
2 instances simultaneous with differents parameters
"""

from pyacq import StreamHandler, FakeMultiSignals
from pyacq.gui import Oscilloscope

import msgpack
#~ import gevent
#~ import zmq.green as zmq

from PyQt4 import QtCore,QtGui

import zmq
import msgpack
import time

import numpy as np

def test1():
    streamhandler = StreamHandler()
    
    
    # Configure and start
    dev = FakeMultiSignals(streamhandler = streamhandler)
    dev.configure( nb_channel = 16,
                                sampling_rate =1000.,
                                buffer_length = 64,
                                packet_size = 10,
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
    w1.set_params(**w1.get_params())

    #~ w2=Oscilloscope(stream = dev.streams[0])
    #~ w2.show()
    #~ w2.auto_gain_and_offset(mode = 0)
    #~ w2.set_params(xsize = 5, mode = 'scroll')
    

    
    app.exec_()
    
    # Stope and release the device
    dev.stop()
    dev.close()



if __name__ == '__main__':
    test1()
