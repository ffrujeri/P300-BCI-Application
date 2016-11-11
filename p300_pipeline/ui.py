import numpy
import os
import ui_qt_design
from PyQt4 import QtCore, QtGui
import pyqtgraph
from termcolor import cprint


class UserInterface(QtGui.QMainWindow, ui_qt_design.P300UI):
    def __init__(self, real_time_electroencephalography, user, parent=None):
        QtGui.QMainWindow.__init__(self, parent)  # initialize inheritance
        ui_qt_design.P300UI.__init__(self, self)
        self.rte = real_time_electroencephalography
        self.user = user

        # plot params
        pg_layout = self.graphics_view.ci
        self.p300_plot_label = pg_layout.addLabel(angle=-90)
        self.p300_plot_item = pg_layout.addPlot()

        # action handlers
        self.action_start_calibration.triggered.connect(self.start_calibration)
        self.action_stop_calibration.triggered.connect(self.stop_calibration)
        self.action_saved_p300_profile.triggered.connect(self.saved_p300_profile)
        self.action_new_p300_profile.triggered.connect(self.new_p300_profile)
        self.action_start_online_mode.triggered.connect(self.start_online_mode)
        self.action_stop_online_mode.triggered.connect(self.stop_online_mode)

    def closeEvent(self, event):
        # TODO: QtGui.QMainWindow abstract function to override (callback when user press close button)
        pass

    @QtCore.pyqtSlot()
    def start_calibration(self):
        self.rte.start_recording_calibration(self.user.id)

    @QtCore.pyqtSlot()
    def stop_calibration(self):
        self.rte.stop_recording_calibration()

    @QtCore.pyqtSlot()
    def start_online_mode(self):
        self.rte.start_online_mode(self.user)

    @QtCore.pyqtSlot()
    def stop_online_mode(self):
        self.rte.stop_online_mode()

    @QtCore.pyqtSlot()
    def new_p300_profile(self):
        self.user.compute_profile(self.rte.profile_filename)
        self.display_p300_curves()

    @QtCore.pyqtSlot()
    def saved_p300_profile(self):
        raw_data_path = str(QtGui.QFileDialog.getExistingDirectory(
            self,
            'Select raw data directory for ' + self.user.id + ' (must be on raw_data directory)',
            '../raw_data',
            QtGui.QFileDialog.ShowDirsOnly))
        self.user.fetch_profile(os.path.basename(raw_data_path))
        self.display_p300_curves()

    def display_p300_curves(self):  # TODO: check
        model = self.user.model
        self.p300_plot_label.setText(
            'Player <b>{}%</b> / {} filters'.format(round(model.accuracy),
                                                    model.optimal_num_filters))
        self.p300_plot_item.clear()

        spatial_filter_num = 0  # TODO: check
        target_mean = model.m2
        target_variance = model.v2
        target_color = (0, 0, 255)
        non_target_mean = model.m1
        non_target_variance = model.v1
        non_target_color = (255, 255, 255)
        self.mean_var_plot(non_target_mean, non_target_variance, non_target_color)
        self.mean_var_plot(target_mean, target_variance, target_color)

    def mean_var_plot(self, mean, variance, color):
        brush_color = color + (100,)
        self.p300_plot_item.plot(mean, pen=color, antialias=True)
        mean_variance_plot1 = self.p300_plot_item.plot(mean + numpy.sqrt(variance), pen=brush_color)
        mean_variance_plot2 = self.p300_plot_item.plot(mean - numpy.sqrt(variance), pen=brush_color)
        mean_variance_plot1.curve.path = mean_variance_plot1.curve.generatePath(*mean_variance_plot1.curve.getData())
        mean_variance_plot2.curve.path = mean_variance_plot2.curve.generatePath(*mean_variance_plot2.curve.getData())
        mean_variance_fill = pyqtgraph.FillBetweenItem(mean_variance_plot1, mean_variance_plot2, brush=brush_color)
        self.p300_plot_item.addItem(mean_variance_fill)
