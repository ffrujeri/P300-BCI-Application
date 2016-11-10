# -*- coding: utf-8 -*-
"""
Oscilloscope example
"""

from pyacq import StreamHandler, EmotivMultiSignals #, FakeMultiSignals
from pyacq.gui import Oscilloscope, TimeFreq

# import msgpack
#~ import gevent
#~ import zmq.green as zmq

from PyQt4 import QtGui #, QtCore

# import zmq
# import msgpack
# import time

def emotiv_oscillo():
    streamhandler = StreamHandler()
    
    # Configure and start
    dev = EmotivMultiSignals(streamhandler=streamhandler)
    dev.configure(buffer_length=1800, device_path='/dev/hidraw3',)
    dev.initialize()
    dev.start()
    
    # EEG
    app = QtGui.QApplication([])
    w1 = Oscilloscope(stream=dev.streams[0])
    w1.show()
    
    # Impedances
    w2 = Oscilloscope(stream=dev.streams[1])
    w2.show()
    
    # Gyro
    w3 = Oscilloscope(stream=dev.streams[2])
    w3.show()

    # Tf chan
    wTf = TimeFreq(stream=dev.streams[0])
    wTf.show()
    
    app.exec_()
    
    # Stop and release the device
    dev.stop()
    dev.close()


if __name__ == '__main__':
    emotiv_oscillo()
