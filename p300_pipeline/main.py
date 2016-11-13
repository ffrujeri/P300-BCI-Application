from oscilloscope import CustomOscilloscope
from PyQt4 import QtGui
from rte import RealTimeElectroencephalography
import sys
from controller import Controller
from user_profile import UserProfile


def test():
    raw_data_path = '../raw_data/player1_2016_10_09_16_32'
    profile_filename = 'player1_2016_11_10_15_18'
    profile_filename = 'player1_2016_10_09_16_32'
    profile_filename = 'player1_2016_11_10_12_40'
    profile_filenames = ['player1_2016_10_09_16_32',
                         'player1_2016_11_10_12_40',
                         'player1_2016_11_10_15_18']
    # print 'profile_filename,num_triggers,classifier,num_filters,accuracy,TP,FP,TN,FN,(TP/(TP+FP),TN/(TN+FN),TP/P,TN/N'
    # for profile_filename in profile_filenames:
    user = UserProfile()
    user.compute_profile(profile_filename)
    # user.save_profile_params_file(profile_filename)

    user2 = UserProfile()
    user2.fetch_profile(profile_filename)


def main():
    app = QtGui.QApplication([])

    rte = RealTimeElectroencephalography()

    oscilloscope = CustomOscilloscope(title=rte.device_path,
                                      stream=rte.bandpass_filter.out_stream)
    oscilloscope.show()

    ui = Controller(rte)
    ui.show()
    # ui.display_p300_curves()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
    # test()
