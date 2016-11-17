from PyQt4 import QtGui
import sys
from controller import Controller
from user_profile import UserProfile
import numpy
import ui_qt_design
from PyQt4 import QtCore, QtGui
import pyqtgraph
from termcolor import cprint
import os


class CalibrationCurvesDisplay(QtGui.QMainWindow, ui_qt_design.P300UI):
    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self, parent)  # initialize inheritance
        ui_qt_design.P300UI.__init__(self, self)
        self.user = UserProfile()

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
        pass

    @QtCore.pyqtSlot()
    def stop_calibration(self):
        pass

    @QtCore.pyqtSlot()
    def start_online_mode(self):
        pass

    @QtCore.pyqtSlot()
    def stop_online_mode(self):
        pass

    @QtCore.pyqtSlot()
    def new_p300_profile(self):
        pass

    @QtCore.pyqtSlot()
    def saved_p300_profile(self):
        raw_data_path = str(QtGui.QFileDialog.getExistingDirectory(
            self,
            'Select raw data directory (must be on user_profile directory)',
            '../user_profile',
            QtGui.QFileDialog.ShowDirsOnly))
        profile_filename = os.path.basename(raw_data_path)
        self.user = UserProfile()
        # self.user.fetch_profile(profile_filename)
        self.user.compute_profile(profile_filename)
        cprint('Fetched {} user profile.'.format(profile_filename), 'yellow')
        self.user.print_model_info()
        self.display_p300_curves()

    def display_p300_curves(self):  # TODO: check
        self.p300_plot_label.setText(
            'F1-score <b>{}%</b> ({} filter(s) used)'.format(
                round(self.user.f1_score * 100), self.user.optimal_num_filters))
        self.p300_plot_item.clear()

        target_color = (0, 0, 255)
        non_target_color = (255, 255, 255)
        self.mean_var_plot(self.user.non_target_mean,
                           self.user.non_target_variance, non_target_color)
        self.mean_var_plot(self.user.target_mean,
                           self.user.target_variance, target_color)

    def mean_var_plot(self, mean, var, color):
        sampling_rate = 128
        x_range = numpy.array(range(0, mean.shape[-1])) * 1000. / sampling_rate
        brush_color = color + (100,)
        self.p300_plot_item.plot(x=x_range, y=mean, pen=color, antialias=True)
        mean_variance_plot_1 = self.p300_plot_item.plot(x=x_range, y=mean + numpy.sqrt(var), pen=brush_color)
        mean_variance_plot_2 = self.p300_plot_item.plot(x=x_range, y=mean - numpy.sqrt(var), pen=brush_color)
        mean_variance_plot_1.curve.path = mean_variance_plot_1.curve.generatePath(*mean_variance_plot_1.curve.getData())
        mean_variance_plot_2.curve.path = mean_variance_plot_2.curve.generatePath(*mean_variance_plot_2.curve.getData())
        mean_variance_fill = pyqtgraph.FillBetweenItem(mean_variance_plot_1, mean_variance_plot_2, brush=brush_color)
        self.p300_plot_item.addItem(mean_variance_fill)


def main():
    app = QtGui.QApplication([])
    ui = CalibrationCurvesDisplay()
    ui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
    # test()
