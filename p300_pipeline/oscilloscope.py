
import colorsys
import pyacq.gui
import pyqtgraph
from PyQt4 import QtCore, QtGui


class CustomOscilloscope(pyacq.gui.Oscilloscope):
    def __init__(self, title, stream):
        pyacq.gui.Oscilloscope.__init__(self, stream=stream)

        self.setWindowTitle(title)
        self.mainlayout.setSpacing(0)
        self.mainlayout.setMargin(0)
        self.auto_gain_and_offset(mode=1)
        self.set_params(xsize=10.)

        # gradient color for background
        gradient = QtGui.QLinearGradient(QtCore.QRectF(self.graphicsview.rect()).topLeft(), QtCore.QRectF(self.graphicsview.rect()).bottomLeft())
        gradient.setColorAt(0, QtCore.Qt.blue)
        gradient.setColorAt(1, QtCore.Qt.black)
        self.graphicsview.setBackground(QtGui.QBrush(gradient))

        pen_color_rgb = (.2, .2, 1)
        hue, sat, val = colorsys.rgb_to_hsv(*pen_color_rgb)
        for index, curve in enumerate(self.curves):
            sat_var = (float(index)/len(self.curves))
            curve.setPen(pyqtgraph.hsvColor(hue, sat_var, val))
