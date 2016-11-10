# -*- coding: utf-8 -*-


from .fakedevices import FakeMultiSignals, FakeDigital, FakeMultiSignalsAndTriggers
device_classes = [FakeMultiSignals, FakeDigital, FakeMultiSignalsAndTriggers ]

try:
    from .measurementcomputing import MeasurementComputingMultiSignals
    device_classes += [MeasurementComputingMultiSignals]
except :
    pass

try:
    from .comedidevices import ComediMultiSignals
    device_classes += [ComediMultiSignals]
except :
    pass


try:
    from .emotiv import EmotivMultiSignals
    device_classes += [EmotivMultiSignals]
except :
    pass

from .emotiv_nonblock import EmotivNonBlocking
device_classes += [EmotivNonBlocking]

from .brainvisionsocket import BrainvisionSocket
device_classes += [BrainvisionSocket]

