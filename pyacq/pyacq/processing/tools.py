# -*- coding: utf-8 -*-

from PyQt4 import QtCore,QtGui
import zmq
import msgpack

import numpy as np
import time

class RecvPosThread(QtCore.QThread):
    """
    Thread for reading in loop in a stream to get the current pos.
    """
    newpacket = QtCore.pyqtSignal(int, int)
    def __init__(self, parent=None, socket = None, port=None):
        QtCore.QThread.__init__(self, parent)
        self.running = False
        self.socket = socket
        self.port = port
        self.pos = None
    
    def run(self):
        self.running = True
        while self.running:
            events = self.socket.poll(50)
            if events ==0:
                time.sleep(.05)
                continue
            
            message = self.socket.recv()
            self.pos = msgpack.loads(message)
            self.newpacket.emit(self.port, self.pos)
    
    def stop(self):
        self.running = False

class WaitLimitThread(QtCore.QThread):
    """
    thread for waiting in a stream a pos.
    """
    limit_reached = QtCore.pyqtSignal(int)
    def __init__(self, parent=None, socket = None, pos_limit = None):
        QtCore.QThread.__init__(self, parent)
        self.running = False
        self.socket = socket
        self.pos_limit = pos_limit
    
    def run(self):
        message = self.socket.recv()
        pos = msgpack.loads(message)
        
        self.running = True
        while self.running:
            events = self.socket.poll(50)
            if events ==0:
                time.sleep(.05)
                continue
            message = self.socket.recv()
            pos = msgpack.loads(message)
            
            if pos>=self.pos_limit:
                self.limit_reached.emit(self.pos_limit)
                self.running = False
                break
    
    def stop(self):
        self.running = False


