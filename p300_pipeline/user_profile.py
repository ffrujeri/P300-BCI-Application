import numpy
import os
import pickle
from termcolor import cprint
from user_profile_util import RawDataReader, SpatialFilters, NaiveBayesModel


class UserProfile:
    RAW_DATA_DIR = '../raw_data/'
    LEARNING_PARAMS_DIR = '../learning_params/'
    PARAMS_EXTENSION = '.params'

    def __init__(self):
        self.id = "player1"
        self.model = None

    def fetch_profile(self, profile_filename):  # TODO
        try:
            learning_params_path = self.get_learning_params_path(profile_filename)
            file_to_read = open(learning_params_path, 'r')
            params_dict = pickle.load(file_to_read)
            file_to_read.close()

            self.model = NaiveBayesModel()
            self.model.set_params(params_dict)

            cprint('Fetched profile from ' + learning_params_path, 'yellow')
        except:  # if file does not exist or problems reading file; TODO: specify exception
            self.compute_profile(profile_filename)

    def compute_profile(self, profile_filename):
        # read raw data (raw signal and triggers)
        raw_data_path = self.get_raw_data_path(profile_filename)
        reader = RawDataReader(raw_data_path)
        eeg_signal, triggers = reader.read_raw_data()

        # process signal (bandpass filter, spatial filters, epoching)
        eeg_signal.set_triggers(triggers)
        eeg_signal.apply_bandpass_filter()
        eeg_signal.apply_spatial_filters()
        eeg_signal.epoch_virtual_signal()

        # select optimal number of filters
        best_accuracy = -1
        best_model = None
        optimal_num_filters = 0
        for prior in [0.2, 0.5, 0.8]:
            for num_filters in range(1, SpatialFilters.MAX_NUM_FILTERS + 1):
                concatenated_epochs = SpatialFilters.concatenate_channels(
                    eeg_signal.epoched_signals[:num_filters])
                model = NaiveBayesModel(concatenated_epochs, eeg_signal.triggers.labels, prior)
                model.compute_params()
                accuracy = model.compute_accuracy()
                # cprint('prior {}, {} filters => {:.2f}% acc (TP={} FP={} TN={} FN={} '
                cprint('{},{},NBprior{},{},{:.2f},{},{},{},{},{:.2f},{:.2f},{:.2f},{:.2f}'.format(
                        profile_filename, len(triggers.indexes), prior, num_filters, accuracy,
                        model.true_positives, model.false_positives,
                        model.true_negatives, model.false_negatives,
                        1. * model.true_positives / (model.true_positives + model.false_positives),
                        1. * model.true_negatives / (model.true_negatives + model.false_negatives),
                        1. * model.true_positives / (model.true_positives + model.false_negatives),
                        1. * model.true_negatives / (model.true_negatives + model.false_positives)
                ), 'blue')
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    optimal_num_filters = num_filters
                    best_model = model
            # cprint('Optimal num filters = ' + str(optimal_num_filters), 'blue')
            # cprint('Accuracy = ' + str(best_accuracy), 'blue')

        self.model = best_model
        self.model.set_w_xdawn_params(eeg_signal.spatial_filters,
                                      optimal_num_filters)
        self.save_profile_params_file(profile_filename)

    @staticmethod
    def get_learning_params_path(profile_filename):
        return os.path.join(UserProfile.LEARNING_PARAMS_DIR, profile_filename) + UserProfile.PARAMS_EXTENSION

    @staticmethod
    def get_raw_data_path(profile_filename):
        return os.path.join(UserProfile.RAW_DATA_DIR, profile_filename)

    def save_profile_params_file(self, profile_filename):
        params_path = UserProfile.get_learning_params_path(profile_filename)
        self.model.save_model_params_file(params_path)
        cprint('Saved profile {} learning params'.format(profile_filename),
               'yellow')

    def compute_likelihoods(self, epoch):  # TODO
        spatial_filters = self.model.spatial_filters.transpose()[:self.model.optimal_num_filters, :]
        virtual_epoch = numpy.dot(spatial_filters, epoch)
        cprint('USER_MODEL process_new_epoch xdawned_new_erp = ' + str(numpy.size(virtual_epoch)), 'magenta', attrs=['bold'])
        virtual_epoch = virtual_epoch.reshape(-1)
        # concatenated_epochs = SpatialFilters.concatenate_channels(
        #     eeg_signal.epoched_signals[:num_filters + 1])
        return self.model.compute_likelihoods(virtual_epoch)
