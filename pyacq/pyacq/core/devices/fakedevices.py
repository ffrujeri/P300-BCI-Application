# -*- coding: utf-8 -*-

import multiprocessing as mp
import numpy as np
import msgpack
import time
from collections import OrderedDict

from .base import DeviceBase


#####
## MultiSignals
def fake_multisignal_mainLoop(stop_flag, stream,  precomputed):
    import zmq
    pos = 0
    abs_pos = pos2 = 0
    
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:{}".format(stream['port']))
    
    socket.send(msgpack.dumps(abs_pos))
    
    packet_size = stream['packet_size']
    sampling_rate = stream['sampling_rate']
    np_arr = stream['shared_array'].to_numpy_array()
    half_size = np_arr.shape[1]/2
    while True:
        t1 = time.time()
        #~ print 'pos', pos, 'abs_pos', abs_pos
        #double copy
        np_arr[:,pos2:pos2+packet_size] = precomputed[:,pos:pos+packet_size] 
        np_arr[:,pos2+half_size:pos2+packet_size+half_size] = precomputed[:,pos:pos+packet_size]
        pos += packet_size
        pos = pos%precomputed.shape[1]
        abs_pos += packet_size
        pos2 = abs_pos%half_size
        socket.send(msgpack.dumps(abs_pos))
        
        if stop_flag.value:
            print 'will stop'
            break
        t2 = time.time()
        #~ time.sleep(packet_size/sampling_rate-(t2-t1))
        
        time.sleep(packet_size/sampling_rate)
        #~ gevent.sleep(packet_size/sampling_rate)








def create_analog_subdevice_param(n):
    d = {
                'type' : 'AnalogInput',
                'nb_channel' : n,
                'params' :{  }, 
                'by_channel_params' : { 
                                        'channel_indexes' : range(n),
                                        'channel_names' : [ 'AI Channel {}'.format(i) for i in range(n)],
                                        'channel_selection' : [True]*n,
                                        }
            }
    return d

def create_event_subdevice_param():
    d = {
                'type' : 'TriggerInput',
                'params' :{  }, 
            }
    return d

class FakeMultiSignals(DeviceBase):
    """
    Usage:
        dev = FakeMultiSignals()
        dev.configure(...)
        dev.initialize()
        dev.start()
        dev.stop()
        
    Configuration Parameters:
        nb_channel
        sampling_rate
        buffer_length
        packet_size
        channel_names
        channel_indexes
    
    
    """
    def __init__(self,  **kargs):
        DeviceBase.__init__(self, **kargs)
    
    def configure(self, 
                                    buffer_length= 10.,
                                    sampling_rate =1000.,
                                    packet_size = 10,
                                    subdevices =[ create_analog_subdevice_param(4) ],
                                    precomputed = None,
                                    # if subdevices is None
                                    nb_channel = None,
                                    last_channel_is_trig = False,
                                    ):
        
        if nb_channel is not None:
            subdevices = [create_analog_subdevice_param(nb_channel), ]
        self.params = {
                                'buffer_length' : buffer_length,
                                'sampling_rate' : sampling_rate,
                                'packet_size' : packet_size,
                                'subdevices' : subdevices,
                                'precomputed' : precomputed,
                                'last_channel_is_trig' : last_channel_is_trig,
                                }
        self.__dict__.update(self.params)
        self.configured = True

    @classmethod
    def get_available_devices(cls):
        devices = OrderedDict()
        
        for n in [4,8,16, 64]:
            name = 'fake {} analog input'.format(n)
            info = {'board_name' : name,
                        'class' : 'FakeMultiSignals',
                        'global_params' : {'sampling_rate' : 1000.,
                                                                 'buffer_length' : 60.,
                                                                 'packet_size' : 10,
                                                                 'last_channel_is_trig': True,
                                                        },
                        'subdevices' : [ create_analog_subdevice_param(n),],
                        }
            devices[name] = info

        return devices


    def initialize(self, streamhandler = None):
        self.sampling_rate = float(self.sampling_rate)
        
        sub0 = self.subdevices[0]
        sel = sub0['by_channel_params']['channel_selection']
        self.nb_channel = np.sum(sel)
        
        channel_indexes = [e   for e, s in zip(sub0['by_channel_params']['channel_indexes'], sel) if s]
        channel_names = [e  for e, s in zip(sub0['by_channel_params']['channel_names'], sel) if s]
        

        l = int(self.sampling_rate*self.buffer_length)
        self.buffer_length = (l - l%self.packet_size)/self.sampling_rate
        
        name = 'fake {} analog input'.format(self.nb_channel)
        #FIXME : name
        s = self.streamhandler.new_AnalogSignalSharedMemStream(name = name, sampling_rate = self.sampling_rate,
                                                        nb_channel = self.nb_channel, buffer_length = self.buffer_length,
                                                        packet_size = self.packet_size, dtype = np.float64,
                                                        channel_names = channel_names, channel_indexes = channel_indexes,            
                                                        )
        
        self.streams = [s, ]
        
        arr_size = s['shared_array'].shape[1]
        assert (arr_size/2)%self.packet_size ==0, 'buffer should be a multilple of pcket_size {}/2 {}'.format(arr_size, self.packet_size)
        
        
        if self.precomputed is None:
            # private precomuted array of 20s = some noise + some sinus burst
            n = int(self.sampling_rate*20./self.packet_size)*self.packet_size
            t = np.arange(n, dtype = np.float64)/self.sampling_rate
            self.precomputed = np.random.rand(self.nb_channel, n)
            for i in range(self.nb_channel):
                f1 = np.linspace(np.random.rand()*60+20. , np.random.rand()*60+20., n)
                f2 = np.linspace(np.random.rand()*1.+.1 , np.random.rand()*1.+.1, n)
                self.precomputed[i,:] += np.sin(2*np.pi*t*f1) * np.sin(np.pi*t*f2+np.random.rand()*np.pi)
                self.precomputed[i,:] += np.random.rand()*40. -20  # add random offset
                self.precomputed[i,:] *= np.random.rand()*10 # add random gain
            
            if self.last_channel_is_trig:
                self.precomputed[-1,:] = 0.
                for i in range(20):
                    self.precomputed[-1,(t>i)&(t<i+.2)] = .5
                    if np.random.rand()<.5:
                        #add  noise
                        self.precomputed[-1,(t>i+.01)&(t<i+0.015)] = 0.
                        self.precomputed[-1,(t>i+.02)&(t<i+0.025)] = 0.
                        
                    
                
            
            
        print 'FakeMultiAnalogChannel initialized:',  s['port']
    
    def start(self):
        
        self.stop_flag = mp.Value('i', 0) #flag pultiproc
        
        s = self.streams[0]
        mp_arr = s['shared_array'].mp_array
        self.process = mp.Process(target = fake_multisignal_mainLoop,  args=(self.stop_flag, s, self.precomputed) )
        self.process.start()
        
        print 'FakeMultiAnalogChannel started:'
        self.running = True
    
    def stop(self):
        self.stop_flag.value = 1
        self.process.join()
        print 'FakeMultiAnalogChannel stopped:'
        
        self.running = False
    
    def close(self):
        pass
        #TODO release stream and close the device



#####
## MultiSignals
def fake_digital_mainLoop(stop_flag, stream,  precomputed):
    import zmq
    pos = 0
    abs_pos = pos2 = 0
    
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:{}".format(stream['port']))
    
    packet_size = stream['packet_size']
    sampling_rate = stream['sampling_rate']
    np_arr = stream['shared_array'].to_numpy_array()
    half_size = np_arr.shape[1]/2
    while True:
        t1 = time.time()
        #~ print 'pos', pos, 'abs_pos', abs_pos
        #double copy
        np_arr[:,pos2:pos2+packet_size] = precomputed[:,pos:pos+packet_size] 
        np_arr[:,pos2+half_size:pos2+packet_size+half_size] = precomputed[:,pos:pos+packet_size]
        pos += packet_size
        pos = pos%precomputed.shape[1]
        abs_pos += packet_size
        pos2 = abs_pos%half_size
        socket.send(msgpack.dumps(abs_pos))
        
        if stop_flag.value:
            print 'will stop'
            break
        t2 = time.time()
        #~ time.sleep(packet_size/sampling_rate-(t2-t1))
        
        time.sleep(packet_size/sampling_rate)
        #~ gevent.sleep(packet_size/sampling_rate)



def create_digital_subdevice_param(n):
    d = {
                'type' : 'DigitalInput',
                'nb_channel' : n,
                'params' :{  }, 
                'by_channel_params' : { 
                                        'channel_indexes' : range(n),
                                        'channel_names' : [ 'DI Channel {}'.format(i) for i in range(n)],
                                        }
            }
    return d

class FakeDigital(DeviceBase):
    """
    
    
    """
    def __init__(self,  **kargs):
        DeviceBase.__init__(self, **kargs)

    def configure(self, 
                                    buffer_length= 10.,
                                    sampling_rate =1000.,
                                    packet_size = 10,
                                    subdevices =[ create_analog_subdevice_param(4) ],
                                    
                                    # if subdevices is None
                                    nb_channel = None,
                                    ):
        
        if nb_channel is not None:
            subdevices = [create_digital_subdevice_param(nb_channel), ]
        self.params = {
                                'buffer_length' : buffer_length,
                                'sampling_rate' : sampling_rate,
                                'packet_size' : packet_size,
                                'subdevices' : subdevices,
                                }
        self.__dict__.update(self.params)
        self.configured = True
        
    
    @classmethod
    def get_available_devices(cls):
        devices = OrderedDict()
        
        for n in [8, 32]:
            name = 'fake {} digital input'.format(n)
            info = {'board_name' : name,
                        'class' : 'FakeDigital',
                        'global_params' : {
                                                    'sampling_rate' : 1000.,
                                                    'buffer_length' : 60.,
                                                    'packet_size' : 10,
                                                    'nb_channel' : n,
                                                    },
                        'subdevices' : [ create_digital_subdevice_param(n)],
                        }
            devices[name] = info
        
        return devices
        
    def initialize(self, streamhandler = None):
        self.sampling_rate = float(self.sampling_rate)
        

        sub0 = self.subdevices[0]
        channel_names = sub0['by_channel_params']['channel_names']
        channel_indexes = sub0['by_channel_params']['channel_indexes']
        self.nb_channel = len(channel_names)
        
        l = int(self.sampling_rate*self.buffer_length)
        self.buffer_length = (l - l%self.packet_size)/self.sampling_rate
        
        #TODO FIXME NAME
        name = 'fake {} digital input'.format(self.nb_channel)
        s = self.streamhandler.new_DigitalSignalSharedMemStream(name = name, sampling_rate = self.sampling_rate,
                                                        nb_channel = self.nb_channel, buffer_length = self.buffer_length,
                                                        packet_size = self.packet_size, channel_names = channel_names)
        
        self.streams = [s, ]
        
        arr_size = s['shared_array'].shape[1]
        assert (arr_size/2)%self.packet_size ==0, 'buffer should be a multilple of pcket_size {}/2 {}'.format(arr_size, self.packet_size)
        
        # private precomuted array of 20s = each channel have a diffrent period
        n = int(self.sampling_rate*20./self.packet_size)*self.packet_size
        t = np.arange(n, dtype = np.float64)/self.sampling_rate
        #~ self.precomputed = np.random.rand(s['shared_array'].shape[0], n)
        self.precomputed = np.zeros((s['shared_array'].shape[0], n), dtype = np.uint8)
        for i in range(self.nb_channel):
            b = i//8
            mask =  1 << i%8
            
            cycle_size = int((i+1)*self.sampling_rate/2)
            #~ print i, b, mask, cycle_size
            period = np.concatenate([np.ones(cycle_size, dtype = np.uint8), np.zeros(cycle_size, dtype = np.uint8)] * (1+n/cycle_size/2))[:n]
            #~ self.precomputed[b, :period.size] = self.precomputed[b, :period.size] + period*mask
            self.precomputed[b, :] = self.precomputed[b, :] + period*mask
            #~ print (period*mask)[:cycle_size*2]
            #~ print self.precomputed[b,:cycle_size*2]
            #~ print n , period.size
        #~ print 'FakeDigital initialized:', self.name, s['port']
    
    def start(self):
        
        self.stop_flag = mp.Value('i', 0) #flag pultiproc
        
        s = self.streams[0]
        mp_arr = s['shared_array'].mp_array
        self.process = mp.Process(target = fake_digital_mainLoop,  args=(self.stop_flag, s, self.precomputed) )
        self.process.start()
        
        #~ print 'FakeDigital started:', self.name
        self.running = True
    
    def stop(self):
        self.stop_flag.value = 1
        self.process.join()
        #~ print 'FakeDigital stopped:', self.name
        
        self.running = False
    
    def close(self):
        pass
        #TODO release stream and close the device




def fake_multisignal_and_triggers_mainLoop(stop_flag, streams,  precomputed_sigs, precomputed_trigs, speed):
    import zmq
    pos = 0
    abs_pos = pos2 = 0
    
    
    context = zmq.Context()
    socket0 = context.socket(zmq.PUB)
    socket0.bind("tcp://*:{}".format(streams[0]['port']))
    socket0.send(msgpack.dumps(abs_pos))
    
    socket1 = context.socket(zmq.PUB)
    socket1.bind("tcp://*:{}".format(streams[1]['port']))
    
    
    packet_size = streams[0]['packet_size']
    sampling_rate = streams[0]['sampling_rate']
    np_arr = streams[0]['shared_array'].to_numpy_array()
    half_size = np_arr.shape[1]/2
    while True:
        t1 = time.time()
        #~ print 'pos', pos, 'abs_pos', abs_pos
        #double copy
        np_arr[:,pos2:pos2+packet_size] = precomputed_sigs[:,pos:pos+packet_size] 
        np_arr[:,pos2+half_size:pos2+packet_size+half_size] = precomputed_sigs[:,pos:pos+packet_size]
        pos += packet_size
        pos = pos%precomputed_sigs.shape[1]
        abs_pos += packet_size
        pos2 = abs_pos%half_size
        socket0.send(msgpack.dumps(abs_pos))
        
        sel = (precomputed_trigs['pos']>=abs_pos-packet_size) & (precomputed_trigs['pos']<abs_pos)
        if np.any(sel):
            for trigger in precomputed_trigs[sel]:
                #~ print 'send', trigger
                socket1.send(trigger.tostring())
        
        if stop_flag.value:
            print 'will stop'
            break
        t2 = time.time()
        #~ time.sleep(packet_size/sampling_rate-(t2-t1))
        
        time.sleep(packet_size/sampling_rate/speed)
        #~ gevent.sleep(packet_size/sampling_rate)

class FakeMultiSignalsAndTriggers(DeviceBase):
    """
    """
    def __init__(self,  **kargs):
        DeviceBase.__init__(self, **kargs)
    
    def configure(self, 
                                    buffer_length= 10.,
                                    sampling_rate =1000.,
                                    packet_size = 10,
                                    subdevices =[ create_analog_subdevice_param(4), create_event_subdevice_param() ],
                                    precomputed_sigs = None,
                                    precomputed_trigs = None,
                                    # if subdevices is None
                                    nb_channel = None,
                                    speed = 1.,
                                    ):
        
        if nb_channel is not None:
            subdevices = [create_analog_subdevice_param(nb_channel), create_event_subdevice_param() ]
        self.params = {
                                'buffer_length' : buffer_length,
                                'sampling_rate' : sampling_rate,
                                'packet_size' : packet_size,
                                'subdevices' : subdevices,
                                'precomputed_sigs' : precomputed_sigs,
                                'precomputed_trigs' : precomputed_trigs,
                                'speed' : speed,
                                }
        self.__dict__.update(self.params)
        self.configured = True

    @classmethod
    def get_available_devices(cls):
        devices = OrderedDict()
        
        for n in [4,]:
            name = 'fake {} analog input and one triggers'.format(n)
            info = {'board_name' : name,
                        'class' : 'FakeMultiSignals',
                        'global_params' : {'sampling_rate' : 1000.,
                                                                 'buffer_length' : 60.,
                                                                 'packet_size' : 10,
                                                                 'last_channel_is_trig': True,
                                                        },
                        'subdevices' : [ create_analog_subdevice_param(n),create_event_subdevice_param()],
                        }
            devices[name] = info
        return devices


    def initialize(self, streamhandler = None):
        self.sampling_rate = float(self.sampling_rate)
        
        sub0 = self.subdevices[0]
        sel = sub0['by_channel_params']['channel_selection']
        self.nb_channel = np.sum(sel)
        
        channel_indexes = [e   for e, s in zip(sub0['by_channel_params']['channel_indexes'], sel) if s]
        channel_names = [e  for e, s in zip(sub0['by_channel_params']['channel_names'], sel) if s]
        

        l = int(self.sampling_rate*self.buffer_length)
        self.buffer_length = (l - l%self.packet_size)/self.sampling_rate
        
        name = 'fake {} analog input'.format(self.nb_channel)
        #FIXME : name
        s0 = self.streamhandler.new_AnalogSignalSharedMemStream(name = name, sampling_rate = self.sampling_rate,
                                                        nb_channel = self.nb_channel, buffer_length = self.buffer_length,
                                                        packet_size = self.packet_size, dtype = np.float64,
                                                        channel_names = channel_names, channel_indexes = channel_indexes,            
                                                        )
        
        arr_size = s0['shared_array'].shape[1]
        assert (arr_size/2)%self.packet_size ==0, 'buffer should be a multilple of pcket_size {}/2 {}'.format(arr_size, self.packet_size)
        
        if self.precomputed_sigs is None:
            # private precomuted array of 20s = some noise + some sinus burst
            n = int(self.sampling_rate*20./self.packet_size)*self.packet_size
            t = np.arange(n, dtype = np.float64)/self.sampling_rate
            self.precomputed = np.random.rand(self.nb_channel, n)
            for i in range(self.nb_channel):
                f1 = np.linspace(np.random.rand()*60+20. , np.random.rand()*60+20., n)
                f2 = np.linspace(np.random.rand()*1.+.1 , np.random.rand()*1.+.1, n)
                self.precomputed[i,:] += np.sin(2*np.pi*t*f1) * np.sin(np.pi*t*f2+np.random.rand()*np.pi)
                self.precomputed[i,:] += np.random.rand()*40. -20  # add random offset
                self.precomputed[i,:] *= np.random.rand()*10 # add random gain
            
            if self.last_channel_is_trig:
                self.precomputed[-1,:] = 0.
                for i in range(20):
                    self.precomputed[-1,(t>i)&(t<i+.2)] = .5
                    if np.random.rand()<.5:
                        #add  noise
                        self.precomputed[-1,(t>i+.01)&(t<i+0.015)] = 0.
                        self.precomputed[-1,(t>i+.02)&(t<i+0.025)] = 0.
                        


        if self.precomputed_trigs is None:
            pass
            #TODO

        name = 'fake triggers'
        s1 = self.streamhandler.new_AsynchronusEventStream(name = name, dtype = self.precomputed_trigs.dtype)
        
                
            
        self.streams = [s0, s1 ]
        print 'FakeMultiSignalsAndTriggers initialized:',  s0['port']
    
    def start(self):
        
        self.stop_flag = mp.Value('i', 0) #flag pultiproc
        
        s = self.streams[0]
        self.process = mp.Process(target = fake_multisignal_and_triggers_mainLoop,  args=(self.stop_flag, self.streams, self.precomputed_sigs, self.precomputed_trigs, self.speed) )
        self.process.start()
        
        print 'FakeMultiSignalsAndTriggers started:'
        self.running = True
    
    def stop(self):
        self.stop_flag.value = 1
        self.process.join()
        print 'FakeMultiSignalsAndTriggers stopped:'
        
        self.running = False
    
    def close(self):
        pass
        #TODO release stream and close the device


