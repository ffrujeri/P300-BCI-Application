from pyacq.core.devices.base import DeviceBase

import zmq
import msgpack
import time
import multiprocessing
import numpy
from termcolor import cprint
from PyQt4 import QtCore
from ctypes import c_bool


class StimulatorTriggersDevice(DeviceBase):
    def __init__(self, stream_handler, eeg_stream,
                 eeg_stream_host='localhost',
                 stimulator_triggers_full_address='tcp://127.0.0.1:5556'):
        DeviceBase.__init__(self, streamhandler=stream_handler)  # args => streamhandler
        if not stimulator_triggers_full_address:
            raise Exception('No address to listen to software triggers.')
        if not eeg_stream:
            raise Exception('Need an analog stream to be synchronized with.')

        self.eeg_stream = eeg_stream
        self.eeg_stream_full_address = \
            'tcp://{}:{}'.format(eeg_stream_host, eeg_stream._params['port'])  # TODO: check for alternatives

        self.stimulator_triggers_full_address = stimulator_triggers_full_address

        self.stop_flag = multiprocessing.Value(c_bool, False)
        self.output_stream = None
        self.process = None

    def initialize(self):
        """Initialize AsynchronousEventStream output."""
        context = zmq.Context()
        triggers_pub_socket = context.socket(zmq.PUB)
        available_port = triggers_pub_socket.bind_to_random_port(
            'tcp://*', min_port=5000, max_port=10000, max_tries=100)
        triggers_pub_socket.close()

        stream_dtype = [('pos', 'int64'),   # received trigger index
                              ('code', 'int64')]  # received trigger code
        self.output_stream = self.streamhandler.new_AsynchronusEventStream(
            name='triggers_indexes', dtype=stream_dtype, port=available_port)

        self.configured = True

    def start(self):
        self.stop_flag.value = False
        self.process = multiprocessing.Process(
            target=StimulatorTriggersDevice.run,
            args=(self.output_stream, self.stop_flag,
                  self.eeg_stream_full_address,
                  self.stimulator_triggers_full_address))
        self.process.start()

    def stop(self):
        self.stop_flag.value = True
        self.process.join()

    def close(self):
        pass

    @staticmethod
    def run(output_stream, stop_flag, eeg_stream_address,
            stimulator_triggers_address):
        context = zmq.Context()

        eeg_stream_socket = context.socket(zmq.SUB)
        eeg_stream_socket.setsockopt(zmq.SUBSCRIBE, '')  # subscribe to all
        eeg_stream_socket.setsockopt(zmq.CONFLATE, 1)  # interested only on the last index
        eeg_stream_socket.connect(eeg_stream_address)

        stimulator_triggers_socket = context.socket(zmq.SUB)
        stimulator_triggers_socket.setsockopt(zmq.SUBSCRIBE, '')
        stimulator_triggers_socket.connect(stimulator_triggers_address)  # full address allows us to use IPC (interprocess protocol)

        socket_out = context.socket(zmq.PUB)
        socket_out.bind("tcp://*:{}".format(output_stream['port']))
        output_marker = numpy.empty((1, ), dtype=output_stream['dtype'])
        cprint('[{}] Sending triggers on port {}'.format(
            multiprocessing.current_process(), output_stream['port']), 'yellow')

        all_eeg_stream_pos = []
        while stop_flag.value is False:
            # listen for new triggers arriving from stimulator
            events_trigger = stimulator_triggers_socket.poll(50)
            if events_trigger == 0:
                time.sleep(.001)  # s
                continue

            trigger = stimulator_triggers_socket.recv()
            cprint('[{}] Received trigger is {}'.format(
                multiprocessing.current_process(), trigger), 'yellow')

            # fetch corresponding eeg index (emotiv)
            events_device = 0
            while events_device == 0:
                events_device = eeg_stream_socket.poll(50)
                if events_device == 0:
                    time.sleep(.001)
            eeg_stream_pos_message = eeg_stream_socket.recv()
            eeg_stream_pos = msgpack.loads(eeg_stream_pos_message)

            all_eeg_stream_pos.append(eeg_stream_pos)

            # send pair ('pos': eeg stream index, 'code': trigger)
            output_marker['pos'] = eeg_stream_pos
            output_marker['code'] = trigger
            socket_out.send(output_marker.tostring())

        stimulator_triggers_socket.close()
        eeg_stream_socket.close()
        socket_out.close()


class TriggersListenerThread(QtCore.QThread):  # TODO: eliminate
    signal_new_incoming_trig = QtCore.pyqtSignal(int)

    def __init__(self, parent=None, triggers_stream=None):
        QtCore.QThread.__init__(self, parent)
        self.running = False
        self.triggers_stream = triggers_stream

        self.context = zmq.Context()
        self.triggers_socket = self.context.socket(zmq.SUB)
        self.triggers_socket.setsockopt(zmq.SUBSCRIBE, '')
        self.triggers_socket.connect("tcp://localhost:{}".format(self.triggers_stream['port']))
        cprint(str(self) + 'listens stream type:' + str(self.triggers_stream['dtype']), 'blue')

    def run(self):
        self.running = True
        while self.running:
            # first check for incoming message on socket to avoid blocking
            events = self.triggers_socket.poll(50)
            if events == 0:
                time.sleep(.05)
                continue
            message = self.triggers_socket.recv()  # blocking function
            trigger = numpy.frombuffer(message, dtype=self.triggers_stream['dtype'])[0]  # uncompress message according to its dtype
            cprint('RECEIVED TRIGGER ' + str(message) + '*** '+ str(trigger), 'yellow', attrs=['bold'])
            self.signal_new_incoming_trig.emit(int(trigger['pos']))

    def stop(self):
        self.running = False

        # clean up
        self.triggers_socket.close()
        self.quit()
        self.wait()
