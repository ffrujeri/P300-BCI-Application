"""Real Time Electroencephalography module for dealing with real time
signal acquisition, calibration and online BCI."""

import datetime
import numpy
import os
import pyacq
from pyacq.processing import StackedChunkOnTrigger
from PyQt4 import QtCore
from stimulator import StimulatorTriggersDevice, TriggersListenerThread
from termcolor import cprint
import zmq


class RealTimeElectroencephalography(QtCore.QObject):
    RAW_DATA_DIR = '../raw_data'

    def __init__(self):
        QtCore.QObject.__init__(self, None)

        self.stream_handler = pyacq.StreamHandler()

        self.device_path = self.find_emotiv_device_path()

        self.eeg_device = pyacq.EmotivMultiSignals(streamhandler=self.stream_handler)
        self.eeg_device.configure(buffer_length=1800, device_path=self.device_path)
        self.eeg_device.initialize()

        self.bandpass_filter = pyacq.BandPassFilter(stream=self.eeg_device.streams[0],
                                                    streamhandler=self.stream_handler)
        self.bandpass_filter.set_params(f_start=.5, f_stop=20.)

        self.stimulator_triggers = StimulatorTriggersDevice(stream_handler=self.stream_handler,
                                                            eeg_stream=self.bandpass_filter.out_stream)
        self.stimulator_triggers.initialize()

        self.context_pub = zmq.Context()  # TODO: eliminate
        self.pub = self.context_pub.socket(zmq.PUB)
        self.pub.bind('tcp://127.0.0.1:6666')

        self.raw_data_recording_device = None
        self.profile_filename = None

        self.triggers_listener_thread = None
        self.epocher = None
        self.user = None

        self.start_watching_signal()

    @staticmethod
    def find_emotiv_device_path():
        """Returns first emotiv path found. Raises IOError if no dongle is connected."""
        devices = pyacq.EmotivMultiSignals.get_available_devices()  # list of connected devices (dongles)
        if not devices.values():
            raise IOError('No Emotiv EEG device found. Please check if dongle is connected.')
        return devices.values()[0]['device_path']

    def start_watching_signal(self):
        self.eeg_device.start()
        self.bandpass_filter.start()

    def stop_watching_signal(self):
        self.bandpass_filter.stop()
        self.eeg_device.stop()

    def start_recording_calibration(self, user_id):
        self.stop_watching_signal()
        self.start_watching_signal()
        self.stimulator_triggers.start()
        self.start_data_recording_device(user_id)
        cprint('Started calibration protocol for ' + user_id, 'yellow')

    def stop_recording_calibration(self):
        self.raw_data_recording_device.stop()
        self.stimulator_triggers.stop()
        self.stop_watching_signal()
        self.start_watching_signal()
        cprint('Stopped calibration protocol.', 'yellow')

    def start_data_recording_device(self, user_id):
        date_string = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
        self.profile_filename = '{}_{}'.format(user_id, date_string)
        raw_data_dir = os.path.join(self.RAW_DATA_DIR, self.profile_filename)
        os.makedirs(raw_data_dir)
        self.raw_data_recording_device = pyacq.RawDataRecording(
            [self.eeg_device.streams[0],
             self.stimulator_triggers.output_stream],
            raw_data_dir)
        self.raw_data_recording_device.start()

    def start_online_mode(self, user):
        self.user = user
        self.init_online_epoching()
        self.stop_watching_signal()
        self.start_watching_signal()
        self.stimulator_triggers.start()
        self.triggers_listener_thread.start()
        cprint('Started online mode.', 'yellow')

    def stop_online_mode(self):  # TODO: check changed order
        self.triggers_listener_thread.stop()
        self.stimulator_triggers.stop()
        self.stop_watching_signal()
        self.start_watching_signal()
        cprint('Stopped online mode.', 'yellow')

    def init_online_epoching(self):
        self.triggers_listener_thread = TriggersListenerThread(
            parent=self, triggers_stream=self.stimulator_triggers.output_stream)
        self.triggers_listener_thread.signal_new_incoming_trig.connect(
            self.on_new_trigger)

        self.epocher = StackedChunkOnTrigger(
            stream=self.bandpass_filter.out_stream,
            stack_size=1,
            left_sweep=0.,
            right_sweep=0.875)
        self.epocher.new_chunk.connect(self.on_new_epoch)

    @QtCore.pyqtSlot(int)
    def on_new_trigger(self, pos):
        self.epocher.on_trigger(pos)

    @QtCore.pyqtSlot(int)
    def on_new_epoch(self):
        stacked_chunk = self.sender()
        if stacked_chunk.total_trig == self.epocher.stack_size:
            new_epoch = stacked_chunk.stack.mean(axis=0)
            self.process_new_epoch(new_epoch)
            self.reset_epoching_stack()

    def process_new_epoch(self, new_epoch):  # TODO
        cprint('RTE process_new_epoch new_erp size = ' +
               str(numpy.size(new_epoch)), 'magenta', attrs=['bold'])  # 112 * 14 = 1568

        lf1, lf2 = self.user.compute_likelihoods(new_epoch)
        cprint('{} {}'.format(lf1, lf2), 'blue')
        self.pub.send('{} {}'.format(lf1, lf2))

    def reset_epoching_stack(self):
        cprint('RTE reset_epoching_stack', 'magenta', attrs=['bold'])
        self.epocher.reset_stack()
