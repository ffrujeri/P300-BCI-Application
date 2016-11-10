from oscilloscope import CustomOscilloscope
from PyQt4 import QtGui
from rte import RealTimeElectroencephalography
import sys
from ui import UserInterface
from user_model import UserProfile


def test():
    raw_data_path = '../raw_data/player1_2016_10_09_16_32'
    profile_filename = 'player1_2016_10_09_16_32'
    user = UserProfile()
    user.compute_profile(profile_filename)
    user.save_profile_params_file(profile_filename)

    user2 = UserProfile()
    user2.fetch_profile(profile_filename)


def main():
    app = QtGui.QApplication([])

    user = UserProfile()
    rte = RealTimeElectroencephalography()

    oscilloscope = CustomOscilloscope(title=user.id + rte.device_path, stream=rte.bandpass_filter.out_stream)
    oscilloscope.show()

    ui = UserInterface(rte, user)
    ui.show()
    user.fetch_profile('player1_2016_10_09_16_32')
    ui.display_p300_curves()

    sys.exit(app.exec_())


if __name__ == '__main__':
    # main()
    test()
